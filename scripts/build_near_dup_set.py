"""Build the labeled near-duplicate pair set.

Downloads real product images for a list of ASINs, applies scripted mutations
with known expected outcomes, and writes a labeled pair manifest.

**The script is committed; the images are not.** `data/near_dup_set/` is
gitignored, so a clean clone rebuilds the set rather than carrying a few hundred
megabytes of other people's product photography in git history. The manifest
records the ASIN and image id of every source, so a rebuild is reproducible even
though the bytes are not stored.

Label semantics, which the build plan had backwards:

*   **positive** — a mutation of the *same source image*. This is what
    "near-duplicate" means: the same photo, reposted resized, recompressed, or
    with a badge added.
*   **hard negative** — a *different image of the same product*. Not a
    duplicate: it is a legitimate additional photo the seller should have. This
    is the discriminating case, because every image in one listing is of one
    product.
*   **easy negative** — an image of a different product.

Usage:
    python -m scripts.build_near_dup_set
    python -m scripts.build_near_dup_set --asins B08XPWDSWW,B07GZFM1ZM
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from vislens.fetch.image_fetch import fetch_image, fetch_listing_image_urls  # noqa: E402
from vislens.rules.image_rules import decode_for_audit  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "near_dup_set"

# The sibling project's supported catalogue, which keeps the two repos talking
# about the same products.
DEFAULT_ASINS = [
    "B08XPWDSWW",  # TOZO T10 earbuds
    "B07GZFM1ZM",  # Fire Stick 4K
    "B01K8B8YA8",  # Echo Dot 2nd gen
    "B07PXGQC1Q",  # AirPods 2nd gen
    "B00N2ZDXW2",  # Ring Video Doorbell
    "B08RLW7918",  # WYZE Cam v2
]

MIN_SIDE = 700
ASPECT_BAND = (0.7, 1.4)


def _is_plausible_product_image(width: int, height: int) -> bool:
    """Drop site chrome. Measured: the banners are ~14:1 and under 100px tall."""
    if min(width, height) < MIN_SIDE:
        return False
    return ASPECT_BAND[0] <= width / height <= ASPECT_BAND[1]


# ── Mutations ─────────────────────────────────────────────────────────────────


def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def mut_resize(img: Image.Image, factor: float) -> Image.Image:
    size = (max(1, int(img.width * factor)), max(1, int(img.height * factor)))
    return img.resize(size, Image.Resampling.LANCZOS)


def mut_badge(img: Image.Image) -> Image.Image:
    """A corner promo badge — the single most common real modification."""
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    side = int(min(out.width, out.height) * 0.20)
    draw.rectangle([10, 10, 10 + side, 10 + side // 2], fill=(220, 30, 40))
    draw.text((22, 16 + side // 8), "50% OFF", fill=(255, 255, 255))
    return out


def mut_watermark(img: Image.Image) -> Image.Image:
    """A low-contrast centre watermark."""
    base = img.copy().convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.text((base.width // 3, base.height // 2), "BRANDNAME", fill=(0, 0, 0, 46))
    return Image.alpha_composite(base, layer).convert("RGB")


def mut_crop(img: Image.Image, keep: float = 0.95) -> Image.Image:
    dx = int(img.width * (1 - keep) / 2)
    dy = int(img.height * (1 - keep) / 2)
    return img.crop((dx, dy, img.width - dx, img.height - dy))


def mut_brighten(img: Image.Image, delta: int = 12) -> Image.Image:
    a = np.asarray(img.convert("RGB"), dtype=np.int16) + delta
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


MUTATIONS: dict[str, object] = {
    "resize_75": lambda im: mut_resize(im, 0.75),
    "resize_50": lambda im: mut_resize(im, 0.50),
    "jpeg_q60": lambda im: _jpeg(im, 60),
    "jpeg_q30": lambda im: _jpeg(im, 30),
    "badge": mut_badge,
    "watermark": mut_watermark,
    "crop_95": mut_crop,
    "brighten": mut_brighten,
}


def build(asins: list[str]) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    sources: list[dict] = []

    for asin in asins:
        page = fetch_listing_image_urls(asin)
        if page.status not in ("ok", "partial"):
            print(f"  {asin}: {page.status} — {page.reason[:70]}")
            continue

        kept = 0
        for url in page.image_urls:
            if kept >= 4:
                break
            result = fetch_image(url)
            if not result.ok or result.data is None:
                continue
            try:
                img = decode_for_audit(result.data)
            except Exception:  # noqa: BLE001
                continue
            if not _is_plausible_product_image(img.orig_width, img.orig_height):
                continue

            image_id = url.split("/I/")[1]
            path = OUT / f"{asin}__{image_id}"
            path.write_bytes(result.data)
            sources.append(
                {
                    "asin": asin,
                    "image_id": image_id,
                    "file": path.name,
                    "url": url,
                    "width": img.orig_width,
                    "height": img.orig_height,
                }
            )
            kept += 1
        print(f"  {asin}: kept {kept} product images ({page.status})")

    # Mutations of every source.
    mutants: list[dict] = []
    for src in sources:
        with Image.open(OUT / src["file"]) as im:
            base = im.convert("RGB")
            for name, fn in MUTATIONS.items():
                out_name = f"mut__{name}__{src['file']}"
                fn(base).save(OUT / out_name, "JPEG", quality=92)  # type: ignore[operator]
                mutants.append({"of": src["file"], "mutation": name, "file": out_name})

    manifest = {
        "asins": asins,
        "n_sources": len(sources),
        "n_mutants": len(mutants),
        "mutations": sorted(MUTATIONS),
        "sources": sources,
        "mutants": mutants,
        "_labels": {
            "positive": "a mutant and its own source — the same image, modified",
            "hard_negative": (
                "two different sources with the same ASIN — a different photo "
                "of the same product, which is NOT a duplicate"
            ),
            "easy_negative": "two sources with different ASINs",
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asins", default=",".join(DEFAULT_ASINS))
    args = parser.parse_args()

    asins = [a.strip() for a in args.asins.split(",") if a.strip()]
    print(f"Building near-duplicate set for {len(asins)} ASINs into {OUT}")
    manifest = build(asins)
    print(
        f"\n{manifest['n_sources']} sources, {manifest['n_mutants']} mutants, "
        f"{len(manifest['mutations'])} mutation types"
    )
    print(f"manifest: {OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
