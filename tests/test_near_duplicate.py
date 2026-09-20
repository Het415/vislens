"""Perceptual hashing and near-duplicate detection.

The robustness assertions here are regression tests for the calibration in
`eval/reports/2026-09-20-near-dup-calibration.md`, not independent discoveries:
the calibration measured which method to ship and at what threshold, and these
pin the properties that decision rests on. If a change breaks one, the
calibration needs re-running rather than the test needs relaxing.
"""

from __future__ import annotations

import copy
import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from vislens.rules.image_dedup import (
    calibrated_method,
    find_group_mismatches,
    find_near_duplicates,
    load_thresholds,
    pairwise_distances,
)
from vislens.rules.image_hash import (
    HASH_BITS,
    compute_hashes,
    detail_std,
    hamming,
    normalised_distance,
    tile_agreement,
)

SIDE = 512


@pytest.fixture
def thresholds() -> dict:
    return copy.deepcopy(load_thresholds())


def textured(seed: int, side: int = SIDE) -> np.ndarray:
    """A synthetic image with real contrast structure.

    Every hash here encodes gradient or DCT signs, so a flat image is a
    degenerate input (see the featureless tests below). Structure is required
    for any of these assertions to mean anything.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    img = Image.fromarray(base).resize((side, side), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(img)
    draw.ellipse([side // 4, side // 4, 3 * side // 4, 3 * side // 4], fill=(20, 20, 20))
    draw.rectangle([side // 8, side // 2, side // 3, 7 * side // 8], fill=(240, 240, 10))
    return np.asarray(img, dtype=np.uint8)


def as_pil(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb)


def jpeg_roundtrip(rgb: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    as_pil(rgb).save(buf, "JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


# ── Hash primitives ───────────────────────────────────────────────────────────


def test_hashes_are_deterministic():
    rgb = textured(1)
    assert compute_hashes(rgb) == compute_hashes(rgb)


def test_identical_images_are_at_distance_zero():
    a = compute_hashes(textured(1))
    b = compute_hashes(textured(1))
    assert hamming(a.dhash, b.dhash) == 0
    assert hamming(a.phash, b.phash) == 0
    assert tile_agreement(a.tiles, b.tiles, 0) == len(a.tiles)


def test_unrelated_images_are_far_apart():
    a, b = compute_hashes(textured(1)), compute_hashes(textured(999))
    assert hamming(a.dhash, b.dhash) > 8
    assert normalised_distance(a.dhash, b.dhash) > 0.12


def test_hamming_is_a_bit_count():
    assert hamming(0b1011, 0b1001) == 1
    assert hamming(0, 2**64 - 1) == HASH_BITS
    assert hamming(5, 5) == 0


def test_tile_agreement_rejects_mismatched_tile_counts():
    with pytest.raises(ValueError, match="tile counts differ"):
        tile_agreement((1, 2), (1, 2, 3), 0)


def test_phash_drops_the_dc_term():
    """The DC coefficient encodes mean brightness, so keeping it would make the
    hash react to an exposure shift that leaves the image recognisable."""
    rgb = textured(2)
    brighter = np.clip(rgb.astype(np.int16) + 20, 0, 255).astype(np.uint8)
    assert hamming(compute_hashes(rgb).phash, compute_hashes(brighter).phash) <= 4


# ── Robustness, per the calibration ───────────────────────────────────────────


@pytest.mark.parametrize("factor", [0.75, 0.5, 0.25])
def test_dhash_survives_rescaling(factor):
    rgb = textured(3)
    small = np.asarray(
        as_pil(rgb).resize(
            (int(SIDE * factor), int(SIDE * factor)), Image.Resampling.LANCZOS
        ),
        dtype=np.uint8,
    )
    assert hamming(compute_hashes(rgb).dhash, compute_hashes(small).dhash) <= 8


@pytest.mark.parametrize("quality", [60, 30])
def test_dhash_survives_recompression(quality):
    rgb = textured(4)
    assert (
        hamming(compute_hashes(rgb).dhash, compute_hashes(jpeg_roundtrip(rgb, quality)).dhash)
        <= 8
    )


def test_dhash_survives_a_pasted_badge():
    """The modification that decided the calibration.

    A 20%-wide promo badge is a large, high-contrast paste. It moves pHash's
    low-frequency coefficients enough to miss at its own threshold (measured:
    48% recall, median distance 12 against a cutoff of 10), while dHash compares
    gradient *signs* on a coarse thumbnail, so the badge flips only a couple of
    bits.
    """
    rgb = textured(5)
    img = as_pil(rgb).copy()
    draw = ImageDraw.Draw(img)
    side = int(SIDE * 0.20)
    draw.rectangle([10, 10, 10 + side, 10 + side // 2], fill=(220, 30, 40))
    badged = np.asarray(img, dtype=np.uint8)

    assert hamming(compute_hashes(rgb).dhash, compute_hashes(badged).dhash) <= 8


def test_tiled_hash_does_not_claim_crop_robustness():
    """Documented before it was measured, and then measured.

    A crop shifts content across cell boundaries, so cell *i* stops
    corresponding to cell *i*. The calibration put tiled dHash at 22% recall on
    a 5% crop against dHash's 96%, which is the structural limit rather than a
    tuning problem — so the docstring's caveat is pinned here.
    """
    rgb = textured(6)
    keep = 0.95
    dx = int(SIDE * (1 - keep) / 2)
    cropped = np.asarray(
        as_pil(rgb).crop((dx, dx, SIDE - dx, SIDE - dx)), dtype=np.uint8
    )

    a, b = compute_hashes(rgb), compute_hashes(cropped)
    tiles_disagreeing = len(a.tiles) - tile_agreement(a.tiles, b.tiles, 10)
    dhash_distance = hamming(a.dhash, b.dhash)

    assert dhash_distance <= 8, "dhash should survive a 5% crop"
    assert tiles_disagreeing >= 2, "tiled hashing is expected to degrade on a crop"


# ── The featureless degenerate case ───────────────────────────────────────────


def test_flat_images_hash_to_zero_and_are_therefore_incomparable():
    """The bug the guard exists for.

    A flat white square and a flat grey square both hash to
    0x0000000000000000 under dHash *and* pHash, so the distance between two
    visibly different images is 0. Without the guard the detector would report
    them as the same image.
    """
    white = compute_hashes(np.full((SIDE, SIDE, 3), 255, np.uint8))
    grey = compute_hashes(np.full((SIDE, SIDE, 3), 128, np.uint8))

    assert white.dhash == 0 and grey.dhash == 0
    assert hamming(white.dhash, grey.dhash) == 0
    assert hamming(white.phash, grey.phash) == 0
    assert white.detail_std < 1.0 and grey.detail_std < 1.0


def test_featureless_images_are_skipped_not_called_duplicates(thresholds):
    hashes = [
        compute_hashes(np.full((SIDE, SIDE, 3), 255, np.uint8)),
        compute_hashes(np.full((SIDE, SIDE, 3), 128, np.uint8)),
    ]
    report = find_near_duplicates(hashes, thresholds)

    assert report.pairs == []
    assert report.skipped_featureless == [0, 1]
    assert "contrast" in report.reason


def test_textured_images_are_not_skipped(thresholds):
    """The guard must not be so broad that it excludes real photographs."""
    hashes = [compute_hashes(textured(i)) for i in (7, 8, 9)]
    report = find_near_duplicates(hashes, thresholds)
    assert report.skipped_featureless == []


def test_detail_std_separates_flat_from_textured():
    assert detail_std(np.full((SIDE, SIDE, 3), 255, np.uint8)) < 1.0
    assert detail_std(textured(10)) > 10.0


# ── Detection, clustering, refusal ────────────────────────────────────────────


def test_a_modified_copy_is_found_as_a_duplicate(thresholds):
    rgb = textured(11)
    hashes = [
        compute_hashes(rgb),
        compute_hashes(jpeg_roundtrip(rgb, 40)),
        compute_hashes(textured(12)),  # unrelated
    ]
    report = find_near_duplicates(hashes, thresholds)

    assert report.method == "dhash"
    assert [(p.a, p.b) for p in report.pairs] == [(0, 1)]
    assert report.clusters == [[0, 1]]


def test_clustering_is_transitive(thresholds):
    """A seller wants "these three are the same photo", not three pair rows —
    so if A matches B and B matches C, all three are one cluster even when A
    and C fall outside the threshold."""
    from vislens.rules.image_dedup import DuplicatePair, _cluster

    pairs = [
        DuplicatePair(a=0, b=1, method="dhash", distance=2),
        DuplicatePair(a=1, b=2, method="dhash", distance=2),
        DuplicatePair(a=4, b=5, method="dhash", distance=1),
    ]
    assert _cluster(6, pairs) == [[0, 1, 2], [4, 5]]


def test_unrelated_images_produce_no_pairs(thresholds):
    hashes = [compute_hashes(textured(i)) for i in (20, 21, 22, 23)]
    report = find_near_duplicates(hashes, thresholds)
    assert report.pairs == []
    assert report.clusters == []


def test_detector_refuses_rather_than_guessing_a_threshold(thresholds):
    """Before calibration the honest answer is "I have not measured this", not
    a plausible-looking cutoff."""
    thresholds["calibrated"] = False
    thresholds["selected_method"] = None

    report = find_near_duplicates([compute_hashes(textured(1))] * 2, thresholds)
    assert report.method is None
    assert report.calibrated is False
    assert report.pairs == []
    assert "calibrated" in report.reason


def test_a_method_with_no_measured_distance_is_refused(thresholds):
    thresholds["methods"]["phash"]["max_distance"] = None
    report = find_near_duplicates([compute_hashes(textured(1))] * 2, thresholds, method="phash")
    assert report.pairs == []
    assert "no measured max_distance" in report.reason


def test_the_shipped_threshold_is_the_calibrated_one():
    """Pins the calibration's output so a silent edit to the config is caught."""
    cfg = load_thresholds()
    assert cfg["calibrated"] is True
    assert cfg["selected_method"] == "dhash"
    assert cfg["methods"]["dhash"]["max_distance"] == 8
    assert cfg["calibration_report"].startswith("eval/reports/")
    # And the measurement that justified it.
    assert cfg["measured"]["held_out_precision"] == 1.0
    assert cfg["measured"]["auc_hard_only"] == 1.0
    assert calibrated_method() == "dhash"


def test_pairwise_distances_covers_every_pair_once():
    hashes = [compute_hashes(textured(i)) for i in (30, 31, 32, 33)]
    dists = pairwise_distances(hashes, "dhash")
    assert set(dists) == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}


# ── SKU group mismatch ────────────────────────────────────────────────────────


def test_group_mismatch_flags_an_image_nearest_to_another_sku(thresholds):
    """The achievable form of "wrong image on a variation".

    Image 2 is tagged as SKU-B but is a recompressed copy of SKU-A's photo, so
    its nearest neighbour is in the wrong group.
    """
    a_rgb = textured(40)
    hashes = [
        compute_hashes(a_rgb),                        # SKU-A
        compute_hashes(textured(41)),                 # SKU-B
        compute_hashes(jpeg_roundtrip(a_rgb, 50)),     # tagged SKU-B, actually A's
    ]
    groups = ["SKU-A", "SKU-B", "SKU-B"]

    mismatches = find_group_mismatches(hashes, groups, thresholds)
    flagged = {m.index: m for m in mismatches}

    assert 2 in flagged
    assert flagged[2].own_group == "SKU-B"
    assert flagged[2].nearest_group == "SKU-A"
    assert flagged[2].nearest_index == 0
    assert flagged[2].distance_to_nearest < (flagged[2].distance_within_own_group or 99)


def test_correctly_grouped_images_are_not_flagged(thresholds):
    a = textured(50)
    hashes = [
        compute_hashes(a),
        compute_hashes(jpeg_roundtrip(a, 60)),
        compute_hashes(textured(51)),
        compute_hashes(jpeg_roundtrip(textured(51), 60)),
    ]
    groups = ["SKU-A", "SKU-A", "SKU-B", "SKU-B"]
    assert find_group_mismatches(hashes, groups, thresholds) == []


def test_untagged_images_are_skipped_not_guessed(thresholds):
    hashes = [compute_hashes(textured(i)) for i in (60, 61)]
    assert find_group_mismatches(hashes, [None, None], thresholds) == []


def test_group_mismatch_needs_a_calibrated_method(thresholds):
    thresholds["calibrated"] = False
    thresholds["selected_method"] = None
    hashes = [compute_hashes(textured(i)) for i in (70, 71)]
    assert find_group_mismatches(hashes, ["A", "B"], thresholds) == []


def test_group_mismatch_is_gated_at_the_duplicate_threshold(thresholds):
    """The bug this closes, found on real data.

    Without a distance gate, a mismatch was reported whenever an image's
    nearest neighbour happened to sit in another group — at any distance. On
    seven real product photos every pairwise dHash distance fell between 19 and
    35 bits, so the nearest-group ordering was arbitrary and 5 of 7 images were
    flagged. Unrelated images must produce no mismatches however they are
    tagged.
    """
    hashes = [compute_hashes(textured(80 + i)) for i in range(6)]
    groups = ["SKU-A", "SKU-A", "SKU-A", "SKU-B", "SKU-B", "SKU-B"]

    # Sanity: these are all genuinely far apart, so nothing here is a duplicate.
    dists = pairwise_distances(hashes, "dhash", thresholds)
    assert min(dists.values()) > thresholds["methods"]["dhash"]["max_distance"]

    assert find_group_mismatches(hashes, groups, thresholds) == []


def test_a_genuine_cross_sku_duplicate_still_reports(thresholds):
    """The gate must not suppress the finding it exists to sharpen."""
    a = textured(90)
    hashes = [
        compute_hashes(a),
        compute_hashes(textured(91)),
        compute_hashes(jpeg_roundtrip(a, 50)),
    ]
    mismatches = find_group_mismatches(hashes, ["SKU-A", "SKU-B", "SKU-B"], thresholds)

    flagged = {m.index for m in mismatches}
    assert 2 in flagged
    near = next(m for m in mismatches if m.index == 2)
    assert near.distance_to_nearest <= thresholds["methods"]["dhash"]["max_distance"]
