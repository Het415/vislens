"""Compliance-check tests.

Every geometric assertion here is **analytic**, not eyeballed: a white canvas
with a black rectangle of known size has an exactly computable bounding box and
occupancy, so the expected values are arithmetic rather than observed output
pasted back in. PNG is used throughout because it is lossless, so the colour
assertions are exact too.

Canvas is 1000x1000, which is under the 1024px working cap, so `downscale` is
1.0 and pixel geometry is not confounded by resizing.
"""

from __future__ import annotations

import copy
import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vislens.rules.image_rules import (
    WIRE_BUDGET_CHARS,
    CheckResult,
    audit_image,
    build_audit_payload,
    check_aspect_ratio,
    check_background_artifacts,
    check_frame_occupancy,
    check_image_count,
    check_resolution_and_format,
    check_white_background,
    decode_for_audit,
    describe_colour,
    load_rules,
    worst_status,
)
from vislens.rules.seller_guidance import fix_for

SIZE = 1000
AREA = SIZE * SIZE


@pytest.fixture
def rules() -> dict:
    return copy.deepcopy(load_rules())


def canvas(size: int = SIZE, bg: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    return np.full((size, size, 3), bg, dtype=np.uint8)


def png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def centered_square(side: int, size: int = SIZE, bg=(255, 255, 255)) -> np.ndarray:
    """A black square of exactly `side` px, centred. bbox occupancy = side^2/size^2."""
    a = canvas(size, bg)
    off = (size - side) // 2
    a[off : off + side, off : off + side] = 0
    return a


def decode(arr: np.ndarray, rules: dict | None = None):
    return decode_for_audit(png(arr), rules)


# ── resolution_and_format (tier rule_exact: exact thresholds) ─────────────────


@pytest.mark.parametrize(
    ("side", "expected"),
    [
        (1200, "pass"),  # above the 1000px zoom threshold
        (1000, "pass"),  # exactly at it
        (999, "warn"),   # one pixel below -> accepted but not zoom-eligible
        (500, "warn"),   # exactly at the accepted minimum
        (499, "fail"),   # one pixel below the accepted minimum
    ],
)
def test_resolution_thresholds_are_exact(side, expected, rules):
    img = decode(canvas(side), rules)
    assert check_resolution_and_format(img, rules).status == expected


def test_resolution_reports_original_dimensions_not_the_working_copy(rules):
    """A 2000px image is downscaled to 1024 for pixel work, but the verdict
    must be about the 2000px original or it is meaningless."""
    img = decode(centered_square(1900, size=2000), rules)
    res = check_resolution_and_format(img, rules)
    assert res.detail["orig_width"] == 2000
    assert res.value == 2000.0
    assert img.work_width == 1024
    assert img.downscale == pytest.approx(1024 / 2000)


def test_zoom_eligibility_is_reported_as_a_fact(rules):
    assert check_resolution_and_format(decode(canvas(1200), rules), rules).detail[
        "zoom_eligible"
    ]
    assert not check_resolution_and_format(decode(canvas(800), rules), rules).detail[
        "zoom_eligible"
    ]


# ── white_background (tier measured) ──────────────────────────────────────────


def test_pure_white_background_passes(rules):
    res = check_white_background(decode(centered_square(400), rules), rules)
    assert res.status == "pass"
    assert res.detail["exact_white_frac"] == 1.0
    assert res.detail["modal_rgb"] == [255, 255, 255]


def test_off_white_background_warns_and_names_the_colour(rules):
    """248,248,248 is the case that a 250 threshold made unreachable.

    It must warn — not fail (it is near-white and neutral) and not pass (it is
    not pure white) — and the user has to be told the actual colour.
    """
    res = check_white_background(
        decode(centered_square(400, bg=(248, 248, 248)), rules), rules
    )
    assert res.status == "warn"
    assert res.detail["modal_rgb"] == [248, 248, 248]
    assert res.detail["near_white_frac"] == 1.0
    assert res.detail["exact_white_frac"] == 0.0
    assert "248" in res.reason


def test_grey_background_fails(rules):
    res = check_white_background(
        decode(centered_square(400, bg=(200, 200, 200)), rules), rules
    )
    assert res.status == "fail"


def test_neutral_but_tinted_background_fails_on_the_spread_term(rules):
    """(255,250,240) is bright enough to pass a luminance test and must still
    fail: channel spread 15 exceeds the neutrality tolerance of 8. This is the
    cream/blue-grey cast the second statistic exists to catch."""
    res = check_white_background(
        decode(centered_square(400, bg=(255, 250, 240)), rules), rules
    )
    assert res.status == "fail"
    assert res.detail["near_white_frac"] == 0.0


def test_pure_white_with_compression_noise_passes(rules):
    """JPEG ringing leaves few *exactly* 255 pixels. If the mode is pure white
    and the band is near-white throughout, that is a white background with
    noise, and warning on it would be a false positive on a compliant image."""
    a = centered_square(400)
    rng = np.random.default_rng(0)
    band = rng.integers(250, 256, size=(12, SIZE, 3), dtype=np.uint8)
    a[:12] = band  # noisy top band, mode still 255 by construction of the range
    img = decode(a, rules)
    res = check_white_background(img, rules)
    assert res.status == "pass"
    assert res.detail["exact_white_frac"] < 0.99
    assert res.detail["modal_rgb"] == [255, 255, 255]


def test_wide_product_touching_the_frame_edge_does_not_fail_the_background(rules):
    """The most common false failure this check has to avoid.

    A compliant product at >=85% occupancy touches the frame edge, so its
    pixels land in the border band. Excluding components that reach the centre
    is what keeps that from reading as a dirty background.
    """
    a = canvas()
    a[:, :] = 255
    a[0:SIZE, 20:980] = 0  # spans the full height, touching top and bottom edges
    res = check_white_background(decode(a, rules), rules)
    assert res.status in ("pass", "skipped")


# ── frame_occupancy (tier measured) ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("side", "expected_occ", "expected"),
    [
        (950, 0.9025, "pass"),  # 950^2/1000^2, above 0.85
        (923, 0.851929, "pass"),  # just above
        (900, 0.81, "fail"),  # 900^2/1000^2, below 0.85
        (400, 0.16, "fail"),
    ],
)
def test_bbox_occupancy_is_analytic(side, expected_occ, expected, rules):
    res = check_frame_occupancy(decode(centered_square(side), rules), rules)
    assert res.value == pytest.approx(expected_occ, abs=0.004)
    assert res.status == expected


def test_silhouette_is_reported_but_never_the_verdict(rules):
    """The regression test for the interpretation error that would make this
    check wrong more often than right.

    An L-shaped product spans a 940x940 bbox (0.8836 of the frame -> compliant)
    while filling only ~0.178 of it by area. A silhouette-area reading would
    fail this compliant listing. bbox must decide; silhouette must be visible.
    """
    a = canvas()
    a[30:970, 30:130] = 0   # vertical bar
    a[870:970, 30:970] = 0  # horizontal bar
    res = check_frame_occupancy(decode(a, rules), rules)

    assert res.status == "pass"
    assert res.detail["bbox_occupancy"] == pytest.approx(0.8836, abs=0.004)
    assert res.detail["silhouette_occupancy"] == pytest.approx(0.178, abs=0.01)
    assert res.detail["silhouette_occupancy"] < 0.85  # would have failed on silhouette


def test_blank_image_is_skipped_not_failed(rules):
    """No product detected is missing information, not a violation."""
    res = check_frame_occupancy(decode(canvas(), rules), rules)
    assert res.status == "skipped"
    assert res.value is None


def test_speck_does_not_inflate_the_bounding_box(rules):
    """A stray dark pixel in a corner would otherwise stretch the bbox to the
    whole frame and turn a failing occupancy into a passing one."""
    a = centered_square(400)
    a[2:4, 2:4] = 0  # 4px speck, below the 500px component floor
    res = check_frame_occupancy(decode(a, rules), rules)
    assert res.status == "fail"
    assert res.value == pytest.approx(0.16, abs=0.01)


def test_white_product_uses_the_gradient_fallback_and_says_so(rules):
    """A near-white product is invisible to the near-white mask, so the mask is
    rebuilt from gradient magnitude. A verdict from a fallback path has to
    declare which path it used.

    245 on 255 is a 10-level edge, which is what a real white product against a
    white background actually looks like. (A 3-level edge sits right on the
    sobel threshold, breaks at the corners, and correctly does *not* trigger the
    fallback — that is noise rejection, not a bug.)
    """
    a = canvas()
    a[100:900, 100:900] = 245  # near-white block, inside the 240 band
    img = decode(a, rules)
    res = check_frame_occupancy(img, rules)
    assert res.detail["mask_method"] == "gradient"
    # 800x800 of 1000x1000 -> the fallback recovers the true geometry exactly.
    assert res.detail["silhouette_occupancy"] == pytest.approx(0.64, abs=0.01)


def test_subtle_noise_does_not_trigger_the_gradient_fallback(rules):
    """The other side of the same threshold: a 3-level variation is JPEG noise,
    not a product, and must not be promoted into a silhouette."""
    a = canvas()
    a[100:900, 100:900] = 252
    res = check_frame_occupancy(decode(a, rules), rules)
    assert res.detail["mask_method"] == "near_white"
    assert res.status == "skipped"


# ── frame_occupancy on a non-white backdrop ───────────────────────────────────

GREY = (200, 200, 200)


@pytest.mark.parametrize(
    ("bg", "product", "rows", "cols", "expected"),
    [
        (GREY, (0, 0, 0), (200, 800), (150, 650), "fail"),  # 600x500 -> 0.30
        ((180, 140, 90), (0, 0, 0), (25, 975), (25, 975), "pass"),  # 950^2 -> 0.9025
        ((60, 90, 200), (0, 0, 0), (300, 700), (300, 700), "fail"),  # 400^2 -> 0.16
        # Touches the top and bottom edges: 1000x900 -> 0.90, and the product
        # now covers 46% of the border, so the backdrop only just has the mode.
        (GREY, (0, 0, 0), (0, 1000), (50, 950), "pass"),
        # Near-white is the product here, not the background. Against white
        # this square would be invisible; against the measured grey it is not.
        (GREY, (255, 255, 255), (300, 700), (300, 700), "fail"),  # 0.16
    ],
)
def test_bbox_occupancy_on_a_non_white_backdrop_is_analytic(
    bg, product, rows, cols, expected, rules
):
    """The false pass this path closes, measured exactly.

    Against white, everything on a grey or coloured backdrop is "not
    near-white", so the bbox was the whole frame and occupancy read 1.0 for a
    product of any size. Cut from the measured backdrop colour instead, a known
    rectangle has an exactly computable bbox again.
    """
    (y0, y1), (x0, x1) = rows, cols
    a = canvas(bg=bg)
    a[y0:y1, x0:x1] = product
    res = check_frame_occupancy(decode(a, rules), rules)

    assert res.detail["mask_method"] == "background_colour"
    assert res.detail["background_rgb"] == list(bg)
    assert res.detail["bbox"] == [x0, y0, x1 - 1, y1 - 1]
    assert res.value == pytest.approx((y1 - y0) * (x1 - x0) / AREA)
    assert res.status == expected


def test_seller_hears_about_size_and_background_from_one_audit(rules):
    """The photo that exposed the false pass, as a seller would upload it.

    JPEG, not PNG, deliberately: it is the reported case, and JPEG noise is
    what `background_match_tolerance` has to absorb. Before, the main-image
    audit failed only `white_background` and passed occupancy at 1.0; the
    seller reshot on white and was then failed on occupancy, a problem the
    first audit could have reported. Both have to arrive together.
    """
    a = np.full((600, 800, 3), (226, 226, 222), dtype=np.uint8)
    a[200:420, 300:520] = (40, 60, 90)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG")
    img = decode_for_audit(buf.getvalue(), rules)
    by_id = {r.check_id: r for r in audit_image(img, rules, is_main=True)}

    assert by_id["white_background"].status == "fail"
    occ = by_id["frame_occupancy"]
    assert occ.status == "fail"
    assert occ.detail["mask_method"] == "background_colour"
    # 220x220 of 800x600; JPEG ringing may move an edge by a pixel.
    assert occ.value == pytest.approx(220 * 220 / (800 * 600), abs=0.003)


def test_blank_grey_photo_is_skipped_not_passed(rules):
    """A grey photo with nothing on it used to read as a product filling the
    whole frame, and pass. Cut from the measured grey it is what it is: blank,
    which is missing information rather than a verdict either way."""
    res = check_frame_occupancy(decode(canvas(bg=GREY), rules), rules)
    assert res.status == "skipped"
    assert res.detail["mask_method"] == "background_colour"


def test_off_white_backdrop_stays_on_the_near_white_mask(rules):
    """The boundary of the new path. (248,248,248) is near-white, so the
    original mask already separates the product from it, and switching
    methods there would change verdicts nobody reported as wrong."""
    res = check_frame_occupancy(
        decode(centered_square(400, bg=(248, 248, 248)), rules), rules
    )
    assert res.detail["mask_method"] == "near_white"
    assert "background_rgb" not in res.detail
    assert res.value == pytest.approx(0.16, abs=0.004)


def test_dark_product_filling_a_white_frame_is_not_mistaken_for_a_backdrop(rules):
    """The regression the band cap guards against.

    A black product spanning the full height and all but 15px of the width
    covers 81% of the border, so black is the border's modal colour. Cutting
    from black would find only the white strip, a ~1.5% "product", on a photo
    that is compliant. The white strip is 19% of the border, which is more
    than `white_background`'s own cap allows a non-white backdrop, so the
    near-white mask stays.
    """
    a = canvas()
    a[:, 15:] = 0
    img = decode(a, rules)
    res = check_frame_occupancy(img, rules)

    assert res.detail["mask_method"] == "near_white"
    assert res.status == "pass"
    assert res.value == pytest.approx(985 / 1000)
    assert check_white_background(img, rules).status == "pass"


def test_cut_that_finds_only_edge_slivers_is_not_kept(rules):
    """The regression the centre rule guards against.

    The same dark product on a grey backdrop makes black the border's mode,
    and nothing near-white is left to say otherwise. Cutting from black then
    finds the grey strip at the edge, which never reaches the centre where a
    product is, and would report it as a 1.5% product.
    """
    a = canvas(bg=GREY)
    a[:, 15:] = 0
    res = check_frame_occupancy(decode(a, rules), rules)
    assert res.detail["mask_method"] == "near_white"
    assert res.status == "pass"

    # Shown, not assumed: a central region covering the whole frame disables
    # the rule, and the confident wrong number comes back.
    rules["checks"]["background_artifacts"]["central_region_frac"] = 1.0
    unguarded = check_frame_occupancy(decode(a, rules), rules)
    assert unguarded.detail["background_rgb"] == [0, 0, 0]
    assert unguarded.value == pytest.approx(15 / 1000)


def test_border_with_no_dominant_colour_is_not_treated_as_a_backdrop(rules):
    """Cutting from a colour that is not the background would be inventing the
    evidence. Random colour on every pixel has no one backdrop, so its most
    common pixel covers almost none of the border and the method stays
    near-white."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, 200, size=(400, 400, 3), dtype=np.uint8)
    res = check_frame_occupancy(decode(a, rules), rules)
    assert res.detail["mask_method"] == "near_white"

    rules["checks"]["frame_occupancy"]["background_min_band_frac"] = 0.0
    unguarded = check_frame_occupancy(decode(a, rules), rules)
    assert unguarded.detail["mask_method"] == "background_colour"


# ── background_artifacts (tier measured) ──────────────────────────────────────


def test_clean_image_has_no_artifacts(rules):
    res = check_background_artifacts(decode(centered_square(400), rules), rules)
    assert res.status == "pass"
    assert res.detail["n_artifacts"] == 0


def test_corner_badge_is_found_and_located(rules):
    a = centered_square(400)
    a[30:70, 30:70] = 0  # 1600px badge, above the 500px floor
    res = check_background_artifacts(decode(a, rules), rules)
    assert res.status == "fail"
    assert res.detail["n_artifacts"] == 1
    assert res.detail["artifacts"][0]["hint"] == "corner_badge"


def test_border_ring_is_classified_as_a_border(rules):
    a = centered_square(400)
    a[:4, :] = 0
    a[-4:, :] = 0
    a[:, :4] = 0
    a[:, -4:] = 0
    res = check_background_artifacts(decode(a, rules), rules)
    assert res.status == "fail"
    assert any(x["hint"] == "border_ring" for x in res.detail["artifacts"])


def test_artifact_check_never_classifies_what_the_mark_is(rules):
    """The check's honesty rests on being definitional: any non-product mark on
    the background violates the white-background rule regardless of what it is.
    It must not claim to recognise text, logos, or watermarks."""
    a = centered_square(400)
    a[30:70, 30:70] = 0
    res = check_background_artifacts(decode(a, rules), rules)
    blob = json.dumps(res.detail) + res.reason + res.rule
    for word in ("text", "logo", "watermark", "OCR", "promotional"):
        assert word.lower() not in blob.lower()


# ── aspect_ratio (tier advisory: must never be a verdict) ─────────────────────


@pytest.mark.parametrize("size", [(1000, 1000), (2000, 500), (500, 2000), (1600, 900)])
def test_aspect_ratio_never_fails(size, rules):
    """Amazon has no hard aspect requirement for most categories, so a fail
    here would be a fabricated verdict with arithmetic behind it."""
    a = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
    res = check_aspect_ratio(decode(a, rules), rules)
    assert res.status in ("pass", "warn")
    assert res.tier == "advisory"


def test_aspect_ratio_rule_text_says_it_is_not_a_rejection_criterion(rules):
    res = check_aspect_ratio(decode(canvas(), rules), rules)
    assert "not a rejection criterion" in res.rule


# ── image_count (tier rule_exact: the phrasing IS the check) ──────────────────


def test_image_count_claims_only_what_was_supplied(rules):
    """The old stub asserted 'Image count (only 3, need 7+)' about a listing it
    had never seen. This check sees only what it was handed and must say so."""
    res = check_image_count(3, rules)
    assert res.status == "warn"
    assert "you supplied 3" in res.reason
    assert "category-dependent" in res.reason
    assert res.detail["n_images_supplied"] == 3
    for phrase in ("your listing has", "need 7+"):
        assert phrase not in res.reason


def test_image_count_passes_at_the_recommended_minimum(rules):
    assert check_image_count(rules["checks"]["image_count"]["recommended_min"], rules).status == (
        "pass"
    )


# ── Tier discipline ───────────────────────────────────────────────────────────


def test_advisory_checks_cannot_drive_the_headline_status():
    """The single most important guard in the module: an advisory signal must
    never be able to turn a compliant listing red."""
    results = [
        CheckResult("resolution_and_format", "pass", "rule_exact"),
        CheckResult("white_background", "pass", "measured"),
        CheckResult("aspect_ratio", "fail", "advisory"),
        CheckResult("off_product_text", "fail", "deferred"),
    ]
    assert worst_status(results) == "pass"


def test_headline_takes_the_worst_verdict_tier_result():
    results = [
        CheckResult("resolution_and_format", "pass", "rule_exact"),
        CheckResult("white_background", "warn", "measured"),
        CheckResult("frame_occupancy", "fail", "measured"),
    ]
    assert worst_status(results) == "fail"


def test_skipped_does_not_count_as_a_failure():
    results = [
        CheckResult("white_background", "skipped", "measured"),
        CheckResult("resolution_and_format", "pass", "rule_exact"),
    ]
    assert worst_status(results) == "pass"


def test_off_product_text_is_not_shipped(rules):
    """Specified, gated, and absent from the shipped run.

    It is now a `model`-tier check rather than a heuristic, because overlaid
    text and text printed on the product are visually identical and only the
    geometry relative to the product distinguishes them. It still does not run
    until its agreement against hand labels clears the gate stored beside it.
    """
    cfg = rules["checks"]["off_product_text"]
    assert cfg["enabled"] is False
    assert cfg["tier"] == "model"
    assert cfg["promote_at_agreement"] >= 0.9
    ran = audit_image(decode(canvas(), rules), rules)
    assert all(r.check_id != "off_product_text" for r in ran)


def test_no_model_check_runs_in_the_deterministic_path(rules):
    """`audit_image` must stay pure: no network, no API key, no variance. That
    is what lets the compliance eval run in CI in seconds with no noise floor,
    unlike the judged agent benchmark."""
    model_checks = {
        name for name, cfg in rules["checks"].items() if cfg.get("tier") == "model"
    }
    assert model_checks  # the VLM checks are specified
    ran = {r.check_id for r in audit_image(decode(centered_square(400), rules), rules)}
    assert ran.isdisjoint(model_checks)


# ── Decode guards ─────────────────────────────────────────────────────────────


def test_exif_orientation_is_applied_before_any_geometry(rules):
    """Without this a portrait phone photo reports landscape and every
    downstream geometry check is rotated."""
    im = Image.new("RGB", (600, 400), "white")
    exif = im.getexif()
    exif[274] = 6  # Orientation: rotate 90
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)

    img = decode_for_audit(buf.getvalue(), rules)
    assert (img.orig_width, img.orig_height) == (400, 600)


def test_decompression_bomb_cap_is_enforced(rules):
    rules["decode"]["max_pixels"] = 100
    with pytest.raises(Image.DecompressionBombError):
        decode_for_audit(png(canvas(1000)), rules)


def test_byte_cap_is_enforced_before_decode(rules):
    rules["decode"]["max_bytes"] = 128
    with pytest.raises(ValueError, match="cap is 128"):
        decode_for_audit(png(canvas(1000)), rules)


def test_transparent_background_warns_rather_than_passing_silently(rules):
    """Amazon renders transparency on white so it usually *looks* compliant, but
    a transparent background is not a white background, and the geometry below
    was measured against a composite we created."""
    rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    rgba[400:600, 400:600] = [0, 0, 0, 255]
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")

    img = decode_for_audit(buf.getvalue(), rules)
    assert img.had_alpha
    res = check_white_background(img, rules)
    assert res.status == "warn"
    assert "transparent" in res.reason


def test_decode_is_deterministic_by_content_hash(rules):
    data = png(centered_square(400))
    assert decode_for_audit(data, rules).sha256 == decode_for_audit(data, rules).sha256


# ── Payload budget ────────────────────────────────────────────────────────────


def test_nine_image_audit_fits_the_synthesizer_budget(rules):
    """ListingLens caps each tool result at 3000 chars, applied per ToolMessage
    rather than against a shared pool, so this audit has its own 3000 and
    WIRE_BUDGET_CHARS keeps 20% of margin.

    The naive form — every check on every image, each repeating its rule text —
    measures 7444 for nine images, which is what forced the legend design.
    """
    img = decode(centered_square(400), rules)
    payload = build_audit_payload([audit_image(img, rules)] * 9, 9, rules)
    wire = json.dumps(payload, separators=(",", ":"))
    assert len(wire) < WIRE_BUDGET_CHARS, f"payload is {len(wire)} chars"


def test_worst_case_audit_still_fits_the_budget(rules):
    """The budget has to hold when everything fails, not just on a clean set.

    Nine images, every verdict-tier check flagged on each, plus the set-level
    count check — built directly rather than photographed, so the worst case is
    guaranteed rather than hoped for.
    """
    # Rule and reason text taken from the real checks, not typed here, so the
    # budget is measured against the wording that actually ships.
    tiny = decode(centered_square(400, size=400, bg=(200, 200, 200)), rules)
    real = {r.check_id: r for r in audit_image(tiny, rules, is_main=True)}
    flagged_everything = [
        replace(real["resolution_and_format"], status="fail", value=400.0),
        replace(real["white_background"], status="fail", value=0.0),
        replace(real["frame_occupancy"], status="fail", value=0.16),
        replace(
            real["background_artifacts"], status="fail", value=3.0,
        ),
        CheckResult(
            "aspect_ratio", "warn", "advisory", 4.0,
            real["aspect_ratio"].rule,
            reason="4.00:1 is 3.00 from square",
        ),
    ]
    payload = build_audit_payload([flagged_everything] * 9, 1, rules)
    wire = json.dumps(payload, separators=(",", ":"))

    assert len(wire) < WIRE_BUDGET_CHARS, f"worst-case payload is {len(wire)} chars"
    assert payload["headline"] == "fail"
    # Five per-image checks plus the set-level count check, which also warns
    # here because the call passes n_supplied=1.
    assert set(payload["legend"]) == {"res", "wbg", "occ", "art", "asp", "cnt"}
    # All nine images fail identically, so they collapse into one group — which
    # is also the better finding: "all 9 images" beats nine identical lines.
    assert len(payload["groups"]) == 1
    group = payload["groups"][0]
    assert group["i"] == list(range(9))
    # Four verdicts, and the advisory aspect-ratio finding kept out of them.
    assert len(group["f"]) == 4
    # The advisory aspect-ratio finding is counted, not enumerated.
    assert group["n_measured_only"] == 1
    # Every broken rule carries a fix; the advisory one must not, because a
    # remedy beside it would assert a problem nobody found.
    assert {c for c, e in payload["legend"].items() if "fix" in e} == {
        "res", "wbg", "occ", "art", "cnt",
    }


def test_payload_carries_no_prose_notes(rules):
    """The measured value plus the rule it was measured against is strictly
    better evidence than our sentence about them, and it is what the
    synthesizer is instructed to quote. Prose would also blow the budget."""
    img = decode(centered_square(400), rules)
    payload = build_audit_payload([audit_image(img, rules, is_main=True)], 1, rules)
    for group in payload["groups"]:
        for finding in group["f"]:
            assert len(finding) <= 3
            assert all(not isinstance(x, str) or len(x) < 12 for x in finding)


def test_payload_enumerates_findings_and_counts_passes(rules):
    """The model needs what is wrong, not a recital of what is right — but the
    denominator has to survive so it can still say '4 of 5 passed'."""
    img = decode(centered_square(400), rules)  # occupancy 0.16 -> fails
    payload = build_audit_payload([audit_image(img, rules, is_main=True)], 1, rules)

    assert payload["headline"] == "fail"
    (group,) = payload["groups"]
    assert [f[0] for f in group["f"]] == ["occ"]
    assert group["f"][0][1] == "fail"
    assert group["f"][0][2] == pytest.approx(0.16, abs=0.01)
    assert group["n_pass"] == 4
    # image_count is a set-level check and warns at 1 supplied image
    assert payload["set_checks"][0][0] == "cnt"
    # The positional finding form is self-describing via the schema key.
    assert payload["f_schema"] == ["check_code", "status", "measured_value"]


def test_payload_legend_carries_only_breached_rules(rules):
    """A rule nothing violated is not evidence for anything, so it does not
    consume budget."""
    img = decode(centered_square(400), rules)
    payload = build_audit_payload([audit_image(img, rules, is_main=True)], 6, rules)

    assert set(payload["legend"]) == {"occ"}
    entry = payload["legend"]["occ"]
    assert entry["check"] == "frame_occupancy"
    assert "85%" in entry["rule"]
    # No `tier` in the legend, on purpose: the same check is a verdict on the
    # main image and a measurement on a secondary one, so one per-code tier
    # would mislabel one of them.
    assert "tier" not in entry


# ── Seller guidance ───────────────────────────────────────────────────────────


def test_broken_rules_carry_a_title_and_a_reshoot_fix(rules):
    """The defect that motivated `seller_guidance`: given only auditor rule
    text, the agent told a seller to "crop or adjust the shot so the product
    fills ≥ 85 % of the frame", which neither the seller nor the developer
    could act on. The fix has to arrive as data beside the verdict."""
    img = decode(centered_square(400), rules)  # occupancy 0.16 -> fails
    payload = build_audit_payload([audit_image(img, rules, is_main=True)], 6, rules)

    entry = payload["legend"]["occ"]
    assert entry["title"] == "Product size in the photo"
    assert "closer" in entry["fix"]
    assert "fix" in payload["key"]


def test_measured_only_rules_get_no_fix(rules):
    """A secondary photo's occupancy is measured, not judged. Its legend entry
    exists, but a fix beside it would tell the seller to repair a lifestyle
    shot that breaks no rule."""
    img = decode(centered_square(400), rules)
    payload = build_audit_payload([audit_image(img, rules, is_main=False)], 6, rules)

    assert payload["groups"][0]["f"] == []
    assert all("fix" not in entry for entry in payload["legend"].values())
    assert "fix" not in payload["key"]


def test_fix_numbers_come_from_the_rules_file(rules):
    rules["checks"]["resolution_and_format"]["zoom_min_longest_side"] = 1600
    assert "1,600 pixels" in fix_for("resolution_and_format", rules)


def test_aspect_ratio_has_no_fix(rules):
    """It is a recommendation, never a verdict, so there is nothing to fix."""
    assert fix_for("aspect_ratio", rules) is None


AUDITOR_JARGON = ("bounding box", "border band", "modal", "silhouette", "px", "frame")


def test_seller_facing_text_has_no_auditor_jargon(rules):
    """Everything a seller reads for a failing main image: the title, the rule,
    the per-image reason and the fix. Each of these words reached a seller
    before and meant nothing to them."""
    # Two photos, because a mark on a grey background merges with it into one
    # non-white region, so no single image fails both the background check and
    # the marks check.
    small = centered_square(200, size=600)
    small[10:40, 10:40] = (255, 0, 0)  # a mark in the corner
    grey = centered_square(500, size=600, bg=(200, 200, 200))
    per_image = [
        audit_image(decode(small, rules), rules, is_main=True),
        audit_image(decode(grey, rules), rules, is_main=True),
    ]
    payload = build_audit_payload(per_image, 2, rules)

    broken = {f[0] for g in payload["groups"] for f in g["f"]}
    assert {"res", "wbg", "occ", "art"} <= broken
    seller_text = " ".join(
        [e["title"] + " " + e["rule"] + " " + e.get("fix", "") for e in payload["legend"].values()]
        + [r.reason for results in per_image for r in results if r.status != "pass"]
    ).lower()
    for word in AUDITOR_JARGON:
        assert word not in seller_text, f"{word!r} reached the seller"


@pytest.mark.parametrize(
    ("rgb", "name"),
    [
        ((248, 248, 248), "off-white"),
        ((200, 200, 200), "light grey"),
        ((128, 128, 128), "grey"),
        ((20, 20, 20), "dark grey or black"),
        ((240, 220, 190), "a light reddish or warm colour"),
        ((60, 90, 200), "a bluish colour"),
    ],
)
def test_background_colour_is_named_for_a_seller(rgb, name):
    assert describe_colour(rgb) == name


def test_payload_headline_ignores_advisory_checks(rules):
    """A wide image warns on aspect ratio; that must not reach the headline."""
    a = np.full((500, 2000, 3), 255, dtype=np.uint8)
    a[20:480, 40:1960] = 0
    payload = build_audit_payload(
        [audit_image(decode(a, rules), rules, is_main=True)], 6, rules
    )
    verdicts = {f[0] for g in payload["groups"] for f in g["f"]}

    # A 4:1 image does warn on aspect ratio, but that is advisory, so it must
    # not appear as a finding at all — only in the measured-only count.
    assert "asp" not in verdicts
    assert sum(g.get("n_measured_only", 0) for g in payload["groups"]) >= 1
    assert "NOT rule verdicts" in payload["key"]["n_measured_only"]


def test_every_check_reports_the_rule_it_was_measured_against(rules):
    """A caller that can only render 'FAIL' is a caller that can fabricate."""
    for res in audit_image(decode(centered_square(400), rules), rules):
        assert res.rule, f"{res.check_id} has no rule text"
        assert res.tier in ("rule_exact", "measured", "advisory")


def test_full_bleed_nonwhite_background_fails_rather_than_skipping(rules):
    """The regression test for the exclusion cap.

    With a grey background the whole image is one non-white component that
    reaches the centre, so the product-exclusion would claim the entire border
    band and report `skipped` — on exactly the violation this check exists to
    catch. Past the cap the raw band is measured instead.
    """
    res = check_white_background(
        decode(centered_square(400, bg=(200, 200, 200)), rules), rules
    )
    assert res.status == "fail"
    assert res.detail["band_product_frac"] > 0.9
    assert res.detail["exclusion"].startswith("none")


def test_exclusion_still_applies_when_the_background_is_white(rules):
    """The cap must not disable the mitigation it guards. A wide product on a
    genuinely white background still gets its border pixels excluded."""
    a = canvas()
    a[0:SIZE, 20:980] = 0
    res = check_white_background(decode(a, rules), rules)
    assert res.status == "pass"
    assert res.detail["band_product_frac"] <= 0.9
    assert res.detail["exclusion"] == "product components reaching the centre"


def test_model_tier_cannot_drive_the_headline(rules):
    """A VLM judgement is reported, never a verdict.

    Same guard as `aspect_ratio`: `worst_status` considers only `rule_exact`
    and `measured`, so no model output can turn a compliant listing red or
    override a computed check.
    """
    results = [
        CheckResult("resolution_and_format", "pass", "rule_exact"),
        CheckResult("white_background", "pass", "measured"),
        CheckResult("off_product_text", "fail", "model"),
        CheckResult("prohibited_content", "fail", "model"),
    ]
    assert worst_status(results) == "pass"


def test_model_checks_are_not_enabled_until_agreement_is_measured(rules):
    """Each VLM check ships disabled with an agreement gate stored beside it,
    so the gate cannot drift from the eval that measures it."""
    for name in ("off_product_text", "image_role", "prohibited_content"):
        cfg = rules["checks"][name]
        assert cfg["enabled"] is False
        assert cfg["tier"] == "model"
        assert cfg["promote_at_agreement"] >= 0.85
        # Local pinned ONNX, not a hosted model: determinism is the property
        # that motivated computing the rest of this module, and a hosted model
        # would forfeit it (plus cost per call and an LLM rate limit).
        assert cfg["runtime"] == "local-onnx"


# ── Main-image scope: the three rules that only govern the main image ────────


def test_main_only_rules_are_verdicts_on_the_main_image(rules):
    img = decode(centered_square(400), rules)  # occupancy 0.16
    by_id = {r.check_id: r for r in audit_image(img, rules, is_main=True)}

    assert by_id["frame_occupancy"].tier == "measured"
    assert by_id["frame_occupancy"].status == "fail"
    assert worst_status(list(by_id.values())) == "fail"


def test_main_only_rules_are_advisory_on_a_secondary_image(rules):
    """A lifestyle photo with a coloured background and a small product is
    perfectly compliant as a secondary image. Failing it would be a verdict
    against a rule that does not govern it.

    Observed on a real listing (B08XPWDSWW): one of six product images measured
    0.5% pure white — correct for a main image, meaningless for the lifestyle
    shot it actually is.
    """
    a = centered_square(300, bg=(180, 140, 90))  # warm lifestyle backdrop
    by_id = {r.check_id: r for r in audit_image(decode(a, rules), rules, is_main=False)}

    for check in ("white_background", "frame_occupancy", "background_artifacts"):
        assert by_id[check].tier == "advisory", check
        assert "secondary image" in by_id[check].reason

    # ...and therefore cannot drag the headline down.
    assert worst_status(list(by_id.values())) in ("pass", "warn")


def test_unknown_role_measures_but_does_not_claim(rules):
    """The scraped-page path cannot identify the main image, so it lands here.
    The measurement is still reported — it is useful — but it is not a claim."""
    by_id = {r.check_id: r for r in audit_image(decode(centered_square(400), rules), rules)}

    occ = by_id["frame_occupancy"]
    assert occ.tier == "advisory"
    assert occ.value == pytest.approx(0.16, abs=0.01)  # still measured
    assert "not supplied" in occ.reason
    assert worst_status(list(by_id.values())) != "fail"


def test_universal_rules_apply_regardless_of_role(rules):
    """Resolution and format are not main-image-specific: a 400px secondary
    image is just as unusable as a 400px main image."""
    small = decode(canvas(400), rules)
    for is_main in (True, False, None):
        by_id = {r.check_id: r for r in audit_image(small, rules, is_main=is_main)}
        assert by_id["resolution_and_format"].tier == "rule_exact"
        assert by_id["resolution_and_format"].status == "fail"


def test_main_only_set_is_declared_in_the_rules_file(rules):
    assert set(rules["main_image_only"]["checks"]) == {
        "white_background",
        "frame_occupancy",
        "background_artifacts",
    }


def test_advisory_findings_are_structurally_separated_from_verdicts(rules):
    """The bug this closes, found on live data.

    The legend keys by check code, but the same check is a verdict on the main
    image and a measurement on a secondary one. A single per-code `tier` field
    therefore mislabelled one of them — a real listing's secondary-image
    occupancy number was being presented under `tier: measured`, i.e. as a
    violation. Splitting `f` (verdicts) from `a` (measurements) makes the
    distinction structural, so a consumer cannot misread it by forgetting to
    check a field.
    """
    main = audit_image(decode(centered_square(400), rules), rules, is_main=True)
    secondary = audit_image(decode(centered_square(400), rules), rules, is_main=False)
    payload = build_audit_payload([main, secondary], 6, rules)

    main_group = next(g for g in payload["groups"] if 0 in g["i"])
    secondary_group = next(g for g in payload["groups"] if 1 in g["i"])

    # Same measured value, opposite standing.
    assert [f[0] for f in main_group["f"]] == ["occ"]
    assert main_group.get("n_measured_only", 0) == 0
    # Same measured value, opposite standing: a verdict on the main image, and
    # on the secondary one not a finding at all — only a count.
    assert secondary_group["f"] == []
    assert secondary_group["n_measured_only"] == 1

    assert payload["main_index"] == 0
    assert payload["headline"] == "fail"  # driven by the main image alone


def test_advisory_findings_carry_no_verdict_vocabulary(rules):
    """The defect a live agent run exposed.

    With `"a":[["wbg","fail",0.0]]` in the payload, the downstream LLM
    reported every advisory measurement as a violation — despite an explicit
    prompt rule forbidding exactly that. The word "fail" dominated the
    structure. Splitting `f` from `a` was necessary and not sufficient: the
    status string inside `a` must not read like a verdict either.
    """
    a = np.full((500, 2000, 3), 255, dtype=np.uint8)
    a[20:480, 40:1960] = 0
    payload = build_audit_payload([audit_image(decode(a, rules), rules)], 6, rules)

    # Not merely neutralised — absent. A number a consumer must not draw
    # conclusions from does not belong in its context.
    assert all("a" not in g for g in payload["groups"])
    assert any(g.get("n_measured_only", 0) > 0 for g in payload["groups"])
    wire = json.dumps(payload)
    assert '"fail"' not in wire or payload["groups"][0]["f"]


def test_unidentified_main_image_carries_an_explicit_caveat(rules):
    """Without it, a consumer writes "your main image fails X" about an image
    nobody identified as the main one — which is what happened."""
    results = audit_image(decode(centered_square(400), rules), rules)
    payload = build_audit_payload([results], 6, rules)

    assert payload["main_index"] is None
    assert payload["caveat"]
    assert "NOT evaluated" in payload["caveat"]
    assert "no verdict on them exists" in payload["caveat"]
    # And it names the remedy rather than leaving the consumer to invent one.
    assert "upload" in payload["caveat"]


def test_an_identified_main_image_has_no_caveat(rules):
    payload = build_audit_payload(
        [audit_image(decode(centered_square(400), rules), rules, is_main=True)], 6, rules
    )
    assert payload["main_index"] == 0
    assert payload["caveat"] is None


def test_fully_opaque_rgba_is_not_treated_as_transparent(rules):
    """The false positive a real browser upload exposed.

    An RGBA PNG whose alpha is 255 everywhere is completely opaque — and that
    is what canvas exports and most design tools produce. Testing the mode
    alone made every such file warn: a pure-white 1200px PNG at 95% occupancy,
    fully compliant, came back `warn` with exact_white_frac 1.0.
    """
    rgba = np.full((SIZE, SIZE, 4), 255, dtype=np.uint8)
    rgba[300:700, 300:700] = [0, 0, 0, 255]
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")

    img = decode_for_audit(buf.getvalue(), rules)
    assert img.mode == "RGBA"
    assert img.had_alpha is False, "opaque RGBA must not read as transparent"

    res = check_white_background(img, rules)
    assert res.status == "pass"
    assert "transparent" not in res.reason


def test_partially_transparent_rgba_still_warns(rules):
    """The guard must not be so loose that it misses real transparency."""
    rgba = np.full((SIZE, SIZE, 4), 255, dtype=np.uint8)
    rgba[:, :, 3] = 0  # fully transparent background
    rgba[300:700, 300:700] = [0, 0, 0, 255]
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")

    img = decode_for_audit(buf.getvalue(), rules)
    assert img.had_alpha is True
    assert "transparent" in check_white_background(img, rules).reason


# ── Packaging: where the thresholds come from ─────────────────────────────────


def test_both_rule_files_ship_inside_the_package():
    """The bug this guards was invisible until the first request in production.

    Both files used to sit at the repo root under `data/` and be resolved with
    `Path(__file__).resolve().parents[3]` — the repo root from a source
    checkout, the interpreter's `lib/python3.11` from site-packages. The wheel
    carried neither, so `pip install .` built green, imported cleanly, and
    raised FileNotFoundError the moment an audit ran. Only an editable install
    worked, which is a deployment constraint nobody would guess from the code.

    CI installs non-editable and loads both files, which is the real guard. This
    one is cheaper and catches the likelier regression: someone reintroducing a
    copy under `data/` and leaving two definitions of the same thresholds.
    """
    from importlib import resources

    from vislens.rules import image_dedup, image_rules

    package = resources.files("vislens.rules")
    for name in (image_rules.RULES_RESOURCE, image_dedup.THRESHOLDS_RESOURCE):
        assert package.joinpath(name).is_file(), name

    # Asserted on FILES, not on the directories. `git mv` leaves the old
    # directories behind on a machine that had them, and an empty directory is
    # harmless; a reintroduced JSON is the regression.
    root = Path(image_rules.__file__).resolve().parents[3]
    for stale in (root / "data" / "image_rules", root / "data" / "near_duplicate"):
        found = sorted(stale.glob("*.json")) if stale.is_dir() else []
        assert not found, f"{found} is back — two definitions of one threshold"


def test_the_loaders_read_the_packaged_copy_by_default():
    """And an explicit path still overrides it, which the calibration script and
    the rule-provenance tests both rely on."""
    from importlib import resources

    from vislens.rules.image_dedup import THRESHOLDS_RESOURCE, load_thresholds
    from vislens.rules.image_rules import RULES_RESOURCE

    packaged = json.loads(resources.files("vislens.rules").joinpath(RULES_RESOURCE).read_text())
    assert load_rules() == packaged

    override = json.loads(
        resources.files("vislens.rules").joinpath(THRESHOLDS_RESOURCE).read_text()
    )
    assert load_thresholds() == override
