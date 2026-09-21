"""Amazon main-image compliance checks.

Pure functions over a decoded image. No network, no model, no torch. The only
dependencies are Pillow (decode), numpy (arithmetic), and scipy.ndimage
(morphology: `label`, `binary_closing`, `sobel`). Deliberately no opencv — those
three operations are all that is needed and scipy already carries them.

Design rules, and the reasons they exist:

1.  **Every check reports the measured number and the rule it was measured
    against.** A caller that can only render "FAIL" is a caller that can
    fabricate. The synthesizer downstream is explicitly forbidden from
    re-deriving a verdict, so the verdict has to arrive with its evidence.

2.  **Every check carries a `tier`.** `rule_exact` is arithmetic on metadata and
    cannot be wrong. `measured` is a pixel statistic with a published
    precision/recall. `advisory` is a signal that is explicitly not a verdict.
    Shipping an `advisory` signal as a red FAIL is the failure mode this whole
    module is arranged to prevent.

3.  **Metadata checks read the original dimensions; pixel checks read a
    downscaled copy, and the output says so.** A verdict computed at 1024px on a
    4000px original is honest only if it declares the downscale factor.

4.  **Thresholds come from `rules_v1.json`, shipped beside this module, never from
    literals here.** The number the UI shows and the number the eval measures
    have to be the same number.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field, replace
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

Status = Literal["pass", "warn", "fail", "skipped"]
# `model` is a VLM judgement. It is reported and labelled, and it deliberately
# does NOT appear in the verdict tiers that `worst_status` considers, so a model
# can never produce or override a computed verdict. Groq hosts no multimodal
# model (checked 2026-09-20), so these run on Anthropic Haiku 4.5 — already
# keyed for the eval judge, and cross-family from the Groq agent so it does not
# eat the 8000 TPM the planner and synthesizer contend for.
Tier = Literal["rule_exact", "measured", "advisory", "model", "deferred"]

# Shipped as PACKAGE DATA, deliberately, and read through `importlib.resources`
# rather than a path relative to `__file__`.
#
# The previous form was `Path(__file__).resolve().parents[3] / "data" / ...`,
# which is the repo root from a source checkout and the interpreter's
# `lib/python3.11` directory from site-packages. Measured: the wheel carried 17
# entries and no data files, so `pip install .` produced a service that imported
# cleanly and then raised FileNotFoundError on its first request. `importlib`
# resolves identically either way, which is the whole point — and it is why
# `render.yaml` no longer has to pin an editable install to stay alive.
RULES_RESOURCE = "rules_v1.json"


@lru_cache(maxsize=4)
def load_rules(path: str | None = None) -> dict[str, Any]:
    """Load and cache the rule thresholds.

    Cached because a batch audit of nine images would otherwise re-read and
    re-parse the same JSON nine times.
    """
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    resource = resources.files("vislens.rules").joinpath(RULES_RESOURCE)
    return json.loads(resource.read_text(encoding="utf-8"))


# ── Results ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CheckResult:
    """One check's verdict, with the evidence that produced it."""

    check_id: str
    status: Status
    tier: Tier
    # The measured quantity, as a number where one exists. `None` for checks
    # whose result is categorical (format) or which were skipped.
    value: float | None = None
    # Human-readable statement of what `value` was compared against. This is
    # what the UI renders next to the number and what the LLM is allowed to
    # quote verbatim.
    rule: str = ""
    # Why a check was skipped, or what qualifies a warn. Never a verdict.
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def compact(self, code: str | None = None) -> dict[str, Any]:
        """One finding, in wire form. See `build_audit_payload` for the budget.

        Carries no `rule` and no `tier`: those are per-check-id, not per-image,
        so repeating them across nine images is what blew the budget (measured
        at 7012 chars). They live once in the payload legend instead.
        """
        out: dict[str, Any] = {"c": code or self.check_id, "s": self.status}
        if self.value is not None:
            out["v"] = round(self.value, 4)
        return out


@dataclass(frozen=True)
class DecodedImage:
    """A decoded, orientation-corrected, size-capped image.

    `rgb` is the *working* copy (downscaled). `orig_width`/`orig_height` are the
    true dimensions, which is what the metadata checks must use.
    """

    rgb: np.ndarray  # (H, W, 3) uint8
    orig_width: int
    orig_height: int
    image_format: str
    mode: str
    n_bytes: int
    had_alpha: bool
    sha256: str

    @property
    def work_height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def work_width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def downscale(self) -> float:
        """Working longest side ÷ original longest side. 1.0 means no downscale."""
        orig = max(self.orig_width, self.orig_height)
        work = max(self.work_width, self.work_height)
        return 1.0 if orig == 0 else work / orig


# ── Decode ────────────────────────────────────────────────────────────────────


def decode_for_audit(data: bytes, rules: dict[str, Any] | None = None) -> DecodedImage:
    """Decode image bytes with the memory caps that make this safe to serve.

    Three guards, all of them load-bearing on a 512 MB instance:

    * `MAX_IMAGE_PIXELS` is capped before any decode, so a 100 KB PNG that
      expands to 30 GB raises instead of OOM-killing the process.
    * `draft()` asks libjpeg to decode at 1/2, 1/4 or 1/8 scale *during* decode,
      so the full-size buffer is never allocated at all. For a 4000px JPEG this
      is roughly a 16x reduction, and it is the difference between fitting in
      the headroom and not.
    * `thumbnail()` finishes the job for formats `draft()` does not support.

    EXIF orientation is applied *first*. Without it a portrait phone photo
    reports landscape dimensions and every downstream geometry check is rotated.
    """
    rules = rules or load_rules()
    dcfg = rules["decode"]

    if len(data) > dcfg["max_bytes"]:
        raise ValueError(f"image is {len(data)} bytes, cap is {dcfg['max_bytes']}")

    previous_cap = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = int(dcfg["max_pixels"])
    try:
        with Image.open(io.BytesIO(data)) as im:
            image_format = (im.format or "UNKNOWN").upper()
            mode = im.mode

            # Whether the image is ACTUALLY transparent, not merely whether it
            # carries an alpha channel.
            #
            # Testing the mode alone was a false positive: an RGBA PNG whose
            # alpha is 255 everywhere is completely opaque, and that is what
            # every canvas export and most design tools produce. Caught by a
            # real browser upload — a pure-white 1200px PNG at 95% occupancy,
            # fully compliant, came back `warn` with exact_white_frac 1.0.
            has_alpha_channel = mode in ("RGBA", "LA", "PA") or "transparency" in im.info
            had_alpha = False
            if has_alpha_channel:
                try:
                    alpha = im.convert("RGBA").getchannel("A")
                    had_alpha = alpha.getextrema()[0] < 255
                except (ValueError, OSError):
                    # Unreadable alpha: assume transparency rather than assert
                    # a white background we could not verify.
                    had_alpha = True

            im = ImageOps.exif_transpose(im)
            orig_width, orig_height = im.size

            work = int(dcfg["work_longest_side"])
            # draft() is JPEG-only and a no-op elsewhere; harmless to always try.
            im.draft("RGB", (work, work))

            if has_alpha_channel:
                # Composite onto white before analysis. Amazon renders
                # transparency on white, so this matches what a shopper sees —
                # but `white_background` still reports the alpha separately,
                # because "renders as white" is not "has a white background".
                im = im.convert("RGBA")
                canvas = Image.new("RGBA", im.size, (255, 255, 255, 255))
                im = Image.alpha_composite(canvas, im)

            im = im.convert("RGB")
            if max(im.size) > work:
                im.thumbnail((work, work), Image.BILINEAR)

            rgb = np.asarray(im, dtype=np.uint8)
    finally:
        Image.MAX_IMAGE_PIXELS = previous_cap

    return DecodedImage(
        rgb=rgb,
        orig_width=orig_width,
        orig_height=orig_height,
        image_format=image_format,
        mode=mode,
        n_bytes=len(data),
        had_alpha=had_alpha,
        sha256=hashlib.sha256(data).hexdigest(),
    )


# ── Mask primitives ───────────────────────────────────────────────────────────

_STRUCT = np.ones((3, 3), dtype=bool)


def near_white_mask(rgb: np.ndarray, min_channel: int, max_spread: int) -> np.ndarray:
    """Pixels that are both bright *and* neutral.

    The neutrality term (`max - min` channel spread) is what distinguishes this
    from a luminance threshold. A faint cream (255, 253, 245) or blue-grey
    (247, 249, 255) cast passes a brightness test and fails this one, which is
    correct: Amazon's rule is pure white, not "light".
    """
    a = rgb.astype(np.int16)
    mn = a.min(axis=2)
    mx = a.max(axis=2)
    return (mn >= min_channel) & ((mx - mn) <= max_spread)


def _border_band_mask(h: int, w: int, frac: float, min_px: int) -> np.ndarray:
    band = max(int(min_px), int(round(frac * min(h, w))))
    band = min(band, max(1, min(h, w) // 2))
    m = np.zeros((h, w), dtype=bool)
    m[:band, :] = True
    m[-band:, :] = True
    m[:, :band] = True
    m[:, -band:] = True
    return m


def _central_mask(h: int, w: int, frac: float) -> np.ndarray:
    """The centred rectangle covering `frac` of each dimension."""
    ch, cw = int(round(h * frac)), int(round(w * frac))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    m = np.zeros((h, w), dtype=bool)
    m[y0 : y0 + ch, x0 : x0 + cw] = True
    return m


def _clean_components(
    fg: np.ndarray, min_area_frac: float
) -> tuple[np.ndarray, np.ndarray]:
    """Close 1px gaps, label, and drop specks below `min_area_frac`.

    Speck removal matters: a single stray JPEG artifact pixel in a corner would
    otherwise stretch the product bounding box to nearly the whole frame and
    turn a failing occupancy check into a passing one.
    """
    # border_value=1 is load-bearing, not a default worth leaving alone.
    # binary_closing erodes after dilating, and the erosion's default
    # border_value=0 treats outside-the-array as background — so it eats the
    # outermost rows of any object touching the frame edge. A compliant product
    # at >=85% occupancy touches the edge by definition, and those eaten rows
    # then survive in the background band as non-white pixels, failing
    # `white_background` on an image that is actually fine. Measured: a 100x60
    # object spanning the full height lost exactly 120 px (2 rows) at 0.
    closed = ndimage.binary_closing(fg, structure=_STRUCT, border_value=1)
    labels, n = ndimage.label(closed, structure=_STRUCT)
    if n == 0:
        return labels, np.array([], dtype=np.int64)

    counts = np.bincount(labels.ravel())
    min_area = max(1.0, min_area_frac * fg.size)
    keep = np.nonzero(counts >= min_area)[0]
    keep = keep[keep != 0]
    return labels, keep


def product_mask(
    rgb: np.ndarray, rules: dict[str, Any] | None = None
) -> tuple[np.ndarray, str]:
    """The product silhouette, and which method produced it.

    Primary method is "everything that is not near-white". The fallback exists
    for white, transparent, and glass products, where that mask is nearly empty:
    it rebuilds from gradient magnitude instead. The method name is returned
    because a verdict derived from a fallback path has to say it used one.
    """
    rules = rules or load_rules()
    occ = rules["checks"]["frame_occupancy"]
    wb = rules["checks"]["white_background"]

    fg = ~near_white_mask(rgb, wb["near_white_min_channel"], wb["near_white_max_spread"])
    labels, keep = _clean_components(fg, occ["component_min_area_frac"])
    mask = np.isin(labels, keep) if keep.size else np.zeros(fg.shape, dtype=bool)

    if mask.mean() >= occ["gradient_fallback_max_mask_frac"]:
        return mask, "near_white"

    gray = rgb.astype(np.float32).mean(axis=2)
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    edges = np.hypot(gx, gy) > float(occ["gradient_threshold"])
    filled = ndimage.binary_fill_holes(
        ndimage.binary_closing(edges, structure=_STRUCT, border_value=1)
    )
    labels, keep = _clean_components(filled, occ["component_min_area_frac"])
    grad_mask = np.isin(labels, keep) if keep.size else np.zeros(fg.shape, dtype=bool)

    if grad_mask.sum() > mask.sum():
        return grad_mask, "gradient"
    return mask, "near_white"


def _product_labels(rgb: np.ndarray, rules: dict[str, Any]) -> tuple[np.ndarray, set[int]]:
    """Label the non-white regions and identify which are the product.

    "The product" is any component that reaches the central region. This is the
    mitigation for the most common false failure in `white_background`: a
    legitimately wide product at >=85% occupancy *touches the frame edge*, so
    its pixels land in the border band and a naive band test reads them as a
    dirty background.
    """
    wb = rules["checks"]["white_background"]
    art = rules["checks"]["background_artifacts"]

    fg = ~near_white_mask(rgb, wb["near_white_min_channel"], wb["near_white_max_spread"])
    labels, keep = _clean_components(fg, art["min_component_area_frac"])
    if not keep.size:
        return labels, set()

    central = _central_mask(rgb.shape[0], rgb.shape[1], art["central_region_frac"])
    present = set(int(v) for v in np.unique(labels[central]) if v != 0)
    return labels, present & set(int(k) for k in keep)


def _modal_rgb(rgb: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    """Most common colour among masked pixels.

    Packs each RGB triple into one int so a single bincount finds the mode
    exactly, rather than approximating it with a per-channel median.
    """
    px = rgb[mask].astype(np.int32)
    packed = (px[:, 0] << 16) | (px[:, 1] << 8) | px[:, 2]
    top = int(np.bincount(packed).argmax())
    return ((top >> 16) & 255, (top >> 8) & 255, top & 255)


# ── Checks ────────────────────────────────────────────────────────────────────


def check_resolution_and_format(
    img: DecodedImage, rules: dict[str, Any] | None = None
) -> CheckResult:
    """Dimensions, format and colour mode. Tier `rule_exact`.

    Uses the *original* dimensions, not the working copy. Lead a demo with this
    check: it is the one that is trivially, verifiably right.
    """
    rules = rules or load_rules()
    cfg = rules["checks"]["resolution_and_format"]
    longest = max(img.orig_width, img.orig_height)

    if img.image_format not in cfg["accepted_formats"]:
        return CheckResult(
            "resolution_and_format",
            "fail",
            "rule_exact",
            float(longest),
            f"format must be one of {', '.join(cfg['accepted_formats'])}",
            reason=f"format is {img.image_format}",
            detail=_res_detail(img, cfg),
        )

    if longest > cfg["max_longest_side"]:
        status, reason = "fail", f"longest side {longest}px exceeds {cfg['max_longest_side']}px"
    elif longest < cfg["accepted_min_longest_side"]:
        status, reason = (
            "fail",
            f"longest side {longest}px is below the {cfg['accepted_min_longest_side']}px minimum",
        )
    elif longest < cfg["zoom_min_longest_side"]:
        status, reason = (
            "warn",
            f"{longest}px is accepted but below the {cfg['zoom_min_longest_side']}px "
            "needed for zoom",
        )
    else:
        status, reason = "pass", ""

    return CheckResult(
        "resolution_and_format",
        status,
        "rule_exact",
        float(longest),
        f"longest side >= {cfg['zoom_min_longest_side']}px for zoom "
        f"(>= {cfg['accepted_min_longest_side']}px accepted)",
        reason=reason,
        detail=_res_detail(img, cfg),
    )


def _res_detail(img: DecodedImage, cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "orig_width": img.orig_width,
        "orig_height": img.orig_height,
        "format": img.image_format,
        "mode": img.mode,
        "n_bytes": img.n_bytes,
        "zoom_eligible": max(img.orig_width, img.orig_height) >= cfg["zoom_min_longest_side"],
    }


def check_aspect_ratio(img: DecodedImage, rules: dict[str, Any] | None = None) -> CheckResult:
    """Aspect ratio. Tier `advisory` — deliberately never pass/fail.

    Amazon has no hard main-image aspect requirement for most categories; square
    is a recommendation. A pass/fail verdict here would be a fabricated finding
    with arithmetic behind it, which is the exact thing this feature replaced.
    """
    rules = rules or load_rules()
    cfg = rules["checks"]["aspect_ratio"]
    ratio = img.orig_width / img.orig_height if img.orig_height else 0.0
    delta = abs(ratio - cfg["recommended_ratio"])

    return CheckResult(
        "aspect_ratio",
        "warn" if delta > cfg["note_delta"] else "pass",
        "advisory",
        round(ratio, 4),
        "square (1:1) is recommended, not required — not a rejection criterion",
        reason=(
            f"{ratio:.2f}:1 is {delta:.2f} from square" if delta > cfg["note_delta"] else ""
        ),
        detail={"ratio": round(ratio, 4), "delta_from_square": round(delta, 4)},
    )


def check_image_count(n_supplied: int, rules: dict[str, Any] | None = None) -> CheckResult:
    """How many images were supplied. Tier `rule_exact`.

    The phrasing *is* the check. This function sees only the images handed to
    it, so it reports what was supplied and never claims to know what the live
    listing contains. The cap is category-dependent, which is what the old
    hardcoded "Image count (only 3, need 7+)" stub asserted universally and
    wrongly.
    """
    rules = rules or load_rules()
    cfg = rules["checks"]["image_count"]

    if n_supplied < cfg["recommended_min"]:
        status = "warn"
        reason = (
            f"you supplied {n_supplied}; {cfg['recommended_min']}+ is recommended. "
            f"The cap is typically {cfg['typical_cap']} but is category-dependent — "
            "verify for your category"
        )
    else:
        status, reason = "pass", ""

    return CheckResult(
        "image_count",
        status,
        "rule_exact",
        float(n_supplied),
        f"{cfg['recommended_min']}+ images recommended",
        reason=reason,
        detail={"n_images_supplied": n_supplied, "typical_cap": cfg["typical_cap"]},
    )


def check_white_background(
    img: DecodedImage, rules: dict[str, Any] | None = None
) -> CheckResult:
    """Background purity over the border band. Tier `measured`.

    Two statistics, because one is not enough: `exact_white_frac` is the literal
    rule, and `near_white_frac` (bright *and* neutral) separates "very slightly
    off-white" from "actually coloured". The modal RGB is reported so the user
    sees *what* their background is rather than only that it failed.
    """
    rules = rules or load_rules()
    cfg = rules["checks"]["white_background"]
    rgb = img.rgb
    h, w = rgb.shape[:2]

    band = _border_band_mask(h, w, cfg["border_band_frac"], cfg["border_band_min_px"])
    labels, product = _product_labels(rgb, rules)
    is_product = np.isin(labels, list(product)) if product else np.zeros((h, w), dtype=bool)

    band_px = int(band.sum())
    claimed = int((band & is_product).sum())
    claimed_frac = claimed / band_px if band_px else 0.0
    max_claim = cfg.get("max_band_product_frac", 0.9)

    if claimed_frac > max_claim:
        # Cap on the product-exclusion, and it matters. Excluding border pixels
        # that belong to a component reaching the centre is what stops a wide
        # compliant product from reading as a dirty background. But with a
        # genuinely grey or coloured background, the *whole image* is one
        # non-white component that reaches the centre, so the exclusion claims
        # the entire band and the check would report `skipped` on exactly the
        # violation it exists to catch. Past this cap the premise no longer
        # holds, so measure the raw band and let it fail.
        bg_band = band
        exclusion = "none (product claimed the whole band)"
    else:
        bg_band = band & ~is_product
        exclusion = "product components reaching the centre"

    base = {
        "downscale": round(img.downscale, 4),
        "had_alpha": img.had_alpha,
        "band_product_frac": round(claimed_frac, 4),
        "exclusion": exclusion,
    }

    if not bg_band.any():
        return CheckResult(
            "white_background",
            "skipped",
            "measured",
            None,
            "border band must be pure white",
            reason="no border band to measure",
            detail=base,
        )

    exact = ((rgb == 255).all(axis=2))[bg_band].mean()
    near = near_white_mask(rgb, cfg["near_white_min_channel"], cfg["near_white_max_spread"])[
        bg_band
    ].mean()
    modal = _modal_rgb(rgb, bg_band)

    detail = {
        **base,
        "exact_white_frac": round(float(exact), 4),
        "near_white_frac": round(float(near), 4),
        "modal_rgb": list(modal),
        "band_px": int(bg_band.sum()),
    }

    if img.had_alpha:
        # Amazon renders transparency on white so it usually *looks* compliant,
        # but a transparent background is not a white background and the
        # geometry below is measured against a composite we created.
        return CheckResult(
            "white_background",
            "warn",
            "measured",
            round(float(exact), 4),
            "background must be pure white (255,255,255)",
            reason="image has a transparent background; measured against a white composite",
            detail=detail,
        )

    near_ok = near >= cfg["near_white_pass_frac"]
    modal_is_white = modal == (255, 255, 255)

    if exact >= cfg["exact_white_pass_frac"]:
        status, reason = "pass", ""
    elif near_ok and modal_is_white and cfg.get("modal_white_passes", True):
        # A JPEG-compressed pure-white background has few *exactly* 255 pixels
        # near product edges because of ringing. Failing or warning on that
        # would be a false positive on a compliant image, so: if the modal
        # colour is pure white and essentially every band pixel is near-white,
        # this is a white background with compression noise.
        status = "pass"
        reason = (
            f"pure white with compression noise — modal RGB is (255,255,255), "
            f"{exact:.1%} of the band exactly white"
        )
    elif near_ok:
        status, reason = (
            "warn",
            f"background is near-white but not pure: modal RGB {modal}, "
            f"{exact:.1%} of the band is exactly (255,255,255)",
        )
    else:
        status, reason = (
            "fail",
            f"only {exact:.1%} of the border band is pure white; modal RGB {modal}",
        )

    return CheckResult(
        "white_background",
        status,
        "measured",
        round(float(exact), 4),
        f"≥{cfg['exact_white_pass_frac']:.0%} of the border band pure white (255,255,255)",
        reason=reason,
        detail=detail,
    )


def check_frame_occupancy(
    img: DecodedImage, rules: dict[str, Any] | None = None
) -> CheckResult:
    """How much of the frame the product fills. Tier `measured`.

    `bbox_occupancy` is the verdict. `silhouette_occupancy` is reported beside
    it and is never the verdict — see the rationale in `rules_v1.json`. Both are
    published so a reader can check the interpretation rather than trust it.
    """
    rules = rules or load_rules()
    cfg = rules["checks"]["frame_occupancy"]
    mask, method = product_mask(img.rgb, rules)
    h, w = mask.shape
    total = float(h * w)

    base = {"mask_method": method, "downscale": round(img.downscale, 4)}

    if not mask.any():
        return CheckResult(
            "frame_occupancy",
            "skipped",
            "measured",
            None,
            f"product bounding box ≥{cfg['min_bbox_occupancy']:.0%} of the frame",
            reason="no product region detected; the image may be blank",
            detail=base,
        )

    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    bbox_occ = ((y1 - y0 + 1) * (x1 - x0 + 1)) / total
    sil_occ = mask.sum() / total

    # The largest single component's bbox, reported separately: a bundle
    # photographed spread across the frame inflates the combined bbox and would
    # otherwise read as a false pass.
    labels, keep = _clean_components(mask, cfg["component_min_area_frac"])
    largest_occ = None
    if keep.size:
        counts = np.bincount(labels.ravel())
        biggest = int(keep[np.argmax(counts[keep])])
        lys, lxs = np.nonzero(labels == biggest)
        largest_occ = round(
            float(
                ((lys.max() - lys.min() + 1) * (lxs.max() - lxs.min() + 1)) / total
            ),
            4,
        )

    detail = {
        **base,
        "bbox_occupancy": round(float(bbox_occ), 4),
        "silhouette_occupancy": round(float(sil_occ), 4),
        "largest_component_bbox_occupancy": largest_occ,
        "n_components": int(keep.size),
        "bbox": [x0, y0, x1, y1],
    }

    passed = bbox_occ >= cfg["min_bbox_occupancy"]
    return CheckResult(
        "frame_occupancy",
        "pass" if passed else "fail",
        "measured",
        round(float(bbox_occ), 4),
        f"product bounding box ≥{cfg['min_bbox_occupancy']:.0%} of the frame",
        reason=(
            ""
            if passed
            else f"product bounding box fills {bbox_occ:.1%} of the frame "
            f"(silhouette {sil_occ:.1%})"
        ),
        detail=detail,
    )


def check_background_artifacts(
    img: DecodedImage, rules: dict[str, Any] | None = None
) -> CheckResult:
    """Non-white regions on the background that are not the product.

    Tier `measured`. This is the honest replacement for "text/logo/watermark
    detection": the rule is that the background must be pure white, so any
    connected non-white component that is not part of the product violates it —
    badge, watermark, border, price starburst, prop, second object. The check
    never has to classify *what* the artifact is, which is why its precision is
    high and why it needs no model.
    """
    rules = rules or load_rules()
    cfg = rules["checks"]["background_artifacts"]
    wbcfg = rules["checks"]["white_background"]
    rgb = img.rgb
    h, w = rgb.shape[:2]
    total = float(h * w)

    labels, product = _product_labels(rgb, rules)
    _, keep = _clean_components(
        ~near_white_mask(
            rgb, wbcfg["near_white_min_channel"], wbcfg["near_white_max_spread"]
        ),
        cfg["min_component_area_frac"],
    )

    artifacts: list[dict[str, Any]] = []
    for lab in (int(k) for k in keep):
        if lab in product:
            continue
        comp = labels == lab
        ys, xs = np.nonzero(comp)
        if not ys.size:
            continue
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        area_frac = float(comp.sum() / total)
        bbox_cov = ((y1 - y0 + 1) * (x1 - x0 + 1)) / total

        if (
            bbox_cov >= wbcfg["ring_bbox_coverage_frac"]
            and area_frac <= wbcfg["ring_max_area_frac"]
        ):
            hint = "border_ring"
        elif area_frac < 0.02 and _in_corner(
            (y0 + y1) / 2, (x0 + x1) / 2, h, w, cfg["corner_region_frac"]
        ):
            hint = "corner_badge"
        elif area_frac >= 0.05:
            hint = "extra_object"
        else:
            hint = "artifact"

        artifacts.append(
            {
                "hint": hint,
                "area_frac": round(area_frac, 5),
                "bbox": [x0, y0, x1, y1],
            }
        )

    artifacts.sort(key=lambda a: -a["area_frac"])
    detail = {
        "n_artifacts": len(artifacts),
        "artifacts": artifacts[:8],
        "mask_method": "near_white",
        "downscale": round(img.downscale, 4),
    }

    if not artifacts:
        return CheckResult(
            "background_artifacts",
            "pass",
            "measured",
            0.0,
            "no non-product marks on the background",
            detail=detail,
        )

    hints = ", ".join(sorted({a["hint"] for a in artifacts}))
    return CheckResult(
        "background_artifacts",
        "fail",
        "measured",
        float(len(artifacts)),
        "no non-product marks on the background",
        reason=(
            f"{len(artifacts)} non-product region(s) on the background ({hints}); "
            f"largest covers {artifacts[0]['area_frac']:.2%} of the frame"
        ),
        detail=detail,
    )


def _in_corner(cy: float, cx: float, h: int, w: int, frac: float) -> bool:
    return (cy < h * frac or cy > h * (1 - frac)) and (cx < w * frac or cx > w * (1 - frac))


# ── Orchestration ─────────────────────────────────────────────────────────────


def audit_image(
    img: DecodedImage,
    rules: dict[str, Any] | None = None,
    *,
    is_main: bool | None = None,
) -> list[CheckResult]:
    """Run every per-image check. `image_count` is per-set, so it is not here.

    `is_main` decides whether three of the checks are **verdicts or merely
    measurements**, and getting it wrong is the most damaging mistake available
    here. Amazon's pure-white-background, 85%-occupancy and
    no-marks-on-the-background requirements govern the **main image only**;
    secondary images are explicitly permitted lifestyle backgrounds, props,
    in-use scenes, text and graphics. Applying them to a whole set would fail a
    seller's perfectly compliant lifestyle photo against a rule that does not
    apply to it.

    Observed on a real listing (B08XPWDSWW): of six product images, one failed
    `white_background` at 0.5% pure white — correctly, for a main image, and
    meaninglessly for the lifestyle shot it actually is.

    * `is_main=True`  — the three checks are verdicts.
    * `is_main=False` — they are downgraded to `advisory`: still measured and
      reported, because the numbers are useful, but not claimed.
    * `is_main=None`  — unknown, treated as `False`. The scraped-page path
      cannot identify the main image, so it always lands here.
    """
    rules = rules or load_rules()
    results = [
        check_resolution_and_format(img, rules),
        check_white_background(img, rules),
        check_frame_occupancy(img, rules),
        check_background_artifacts(img, rules),
        check_aspect_ratio(img, rules),
    ]
    if is_main is True:
        return results

    main_only = set(rules.get("main_image_only", {}).get("checks", ()))
    note = (
        "main-image rule; this image is a secondary image"
        if is_main is False
        else "main-image rule; which image is the main one was not supplied"
    )
    return [
        replace(
            res,
            tier="advisory",
            reason=f"{res.reason} ({note})".strip() if res.reason else f"Measured only — {note}",
        )
        if res.check_id in main_only
        else res
        for res in results
    ]


def worst_status(results: list[CheckResult]) -> Status:
    """The set's headline status.

    Two exclusions, both deliberate:

    * `advisory` and `deferred` checks cannot drive the headline. A signal that
      is explicitly not a verdict must not be able to turn a compliant listing
      red — this is the guard that keeps `aspect_ratio` harmless.
    * `skipped` is not a verdict either. A check that could not run is missing
      information, not a violation, so it never outranks a `pass`. Only when
      *every* verdict-tier check was skipped is the headline `skipped`.
    """
    order: dict[str, int] = {"pass": 0, "warn": 1, "fail": 2}
    verdicts = [r for r in results if r.tier in ("rule_exact", "measured")]
    decided = [r for r in verdicts if r.status in order]
    if not decided:
        return "skipped"
    return max(decided, key=lambda r: order[r.status]).status


# ── Wire payload ──────────────────────────────────────────────────────────────

# Short codes keep the per-image cost down. The legend maps each back to its
# full check id, so nothing is ambiguous on the receiving end.
# 80% of ListingLens' per-tool-result cap of 3000 chars. Not 1200: that number
# came from mistakenly reading `review_qa`'s ~2239 as consuming a shared pool,
# when the cap is applied per ToolMessage.
WIRE_BUDGET_CHARS = 2400

_CHECK_CODES: dict[str, str] = {
    "resolution_and_format": "res",
    "white_background": "wbg",
    "frame_occupancy": "occ",
    "background_artifacts": "art",
    "aspect_ratio": "asp",
    "image_count": "cnt",
}


def build_audit_payload(
    per_image: list[list[CheckResult]],
    n_supplied: int,
    rules: dict[str, Any] | None = None,
    audit_id: str | None = None,
) -> dict[str, Any]:
    """Assemble the audit result for ListingLens' agent.

    The budget is the reason this is shaped the way it is. ListingLens'
    synthesizer truncates each tool result at `_TOOL_RESULT_CHARS = 3000`, and
    that cap is applied **per ToolMessage** (`synthesizer.py:97`), not against a
    shared pool — so this audit gets its own 3000 characters regardless of what
    `review_qa` spends. `WIRE_BUDGET_CHARS` leaves 20% of margin.

    Staying well under it still matters even with room to spare: the transcript
    concatenates every tool result into one prompt, and Groq's free tier allows
    8000 tokens per minute per model.

    Four decisions, measured against the naive form (every check on every
    image, each repeating its rule text) at 7444 characters for nine images:

    1.  **Images are grouped by identical findings.** This is the largest win
        and it is also the better finding: "all 9 images have a grey
        background" is what a seller needs to hear, and nine separate identical
        statements are noise. Groups separate naturally when images differ.
    2.  **Findings are enumerated; passes are counted.** The model needs what
        is wrong, not a recital of what is right. `n_pass` survives per group so
        it can still say "4 of 5 checks passed".
    3.  **The legend carries only rules that were actually breached.** A rule
        nothing violated is not evidence for anything.
    4.  **No prose notes.** The measured value plus the rule it was measured
        against is strictly better evidence than our sentence about them, and it
        is what the synthesizer is instructed to quote. Full detail — every
        passing check, artifact bounding boxes, modal RGB — stays behind
        `audit_id` for the UI.
    """
    rules = rules or load_rules()
    count_check = check_image_count(n_supplied, rules)

    legend: dict[str, dict[str, Any]] = {}
    all_results: list[CheckResult] = [count_check]
    main_index: int | None = None

    def remember(res: CheckResult) -> str:
        code = _CHECK_CODES.get(res.check_id, res.check_id)
        if code not in legend:
            # Deliberately no `tier` here. The same check is a verdict on the
            # main image and a measurement on a secondary one, so a single
            # per-code tier would mislabel one of them — which is how a
            # secondary image's occupancy number could be read as a violation.
            # Verdict-vs-measurement is carried structurally instead, by which
            # list a finding lands in.
            legend[code] = {"check": res.check_id, "rule": res.rule}
        return code

    grouped: dict[tuple, list[int]] = {}
    signatures: dict[tuple, tuple[list, list, int]] = {}

    for i, results in enumerate(per_image):
        verdicts: list[list[Any]] = []
        advisories: list[list[Any]] = []
        n_pass = 0
        for res in results:
            all_results.append(res)
            if res.status == "pass":
                n_pass += 1
                continue
            code = remember(res)
            is_verdict = res.tier in ("rule_exact", "measured")
            # Advisory findings carry NO verdict vocabulary. Measured on a live
            # run: with `"a":[["wbg","fail",0.0]]` in the payload, the
            # downstream LLM reported every advisory measurement as a
            # violation despite an explicit prompt rule forbidding exactly
            # that — the word "fail" dominated the structure. Splitting the
            # lists was necessary and not sufficient; the status string has to
            # stop reading like a verdict too.
            status = res.status if is_verdict else "measured"
            finding = [code, status] + (
                [round(res.value, 4)] if res.value is not None else []
            )
            (verdicts if is_verdict else advisories).append(finding)
        sig = (tuple(map(tuple, verdicts)), tuple(map(tuple, advisories)), n_pass)
        grouped.setdefault(sig, []).append(i)
        signatures[sig] = (verdicts, advisories, n_pass)

    # Whichever image had the main-image rules applied as verdicts.
    for i, results in enumerate(per_image):
        if any(
            r.check_id in set(rules.get("main_image_only", {}).get("checks", ()))
            and r.tier in ("rule_exact", "measured")
            for r in results
        ):
            main_index = i
            break

    # Advisory measurements are counted here, NOT enumerated.
    #
    # Three iterations to get this boundary right, each one measured against a
    # live agent run rather than assumed:
    #
    #   1. One list, with a per-code `tier` field. The tier varied per image
    #      (a rule is a verdict on the main image and a measurement on a
    #      secondary one), so one field mislabelled one of them.
    #   2. Two lists, `f` and `a`, but both carrying "fail"/"warn" statuses.
    #      The LLM reported every `a` entry as a violation anyway — the word
    #      "fail" dominated the structure, and an explicit prompt rule
    #      forbidding it did not hold.
    #   3. `a` statuses neutralised to "measured". Better — the evidence field
    #      started saying "measured" — but the prose still asserted "the
    #      background isn't pure white" and called them "key rule failures".
    #
    # The conclusion is that a number a consumer must not draw conclusions
    # from does not belong in its context at all. The agent payload now
    # carries only what the agent may legitimately assert; every measurement
    # remains in the detail record for the UI, which renders rather than
    # reasons. The count survives so nothing is hidden.
    groups = []
    for sig, idxs in grouped.items():
        verdicts, advisories, n_pass = signatures[sig]
        entry: dict[str, Any] = {"i": idxs, "n_pass": n_pass, "f": verdicts}
        if advisories:
            entry["n_measured_only"] = len(advisories)
        groups.append(entry)

    if count_check.status != "pass":
        remember(count_check)

    payload_caveat = None
    if main_index is None:
        # Unmissable, because a consumer that misses it will write "your main
        # image fails X" about an image nobody identified as the main one.
        payload_caveat = (
            "The main image was not identified, so the three main-image rules "
            "(white background, frame occupancy, background marks) were NOT "
            "evaluated and no verdict on them exists. Do not state or imply "
            "anything about whether they pass or fail. To get a verdict, the "
            "seller must upload their images and say which one is the main "
            "image."
        )

    return {
        "rules_version": rules["rules_version"],
        "audit_id": audit_id,
        "n_images": n_supplied,
        "headline": worst_status(all_results),
        "main_index": main_index,
        "caveat": payload_caveat,
        "f_schema": ["check_code", "status", "measured_value"],
        # `f` findings are verdicts against a rule. `a` findings are
        # measurements only — an advisory-tier check, or a main-image rule
        # measured on an image not known to be the main one. Their status is
        # always the literal "measured" so there is no verdict word to misread.
        "key": {
            "f": "VERDICT: this rule was broken. These are the only findings.",
            "n_measured_only": (
                "count of diagnostic measurements that are NOT rule verdicts. "
                "Their values are deliberately not included. Do not speculate "
                "about them or describe them as problems."
            ),
        },
        "set_checks": (
            [[_CHECK_CODES["image_count"], count_check.status, count_check.value]]
            if count_check.status != "pass"
            else []
        ),
        "legend": legend,
        "groups": groups,
    }
