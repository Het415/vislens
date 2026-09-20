"""Near-duplicate detection and catalogue hygiene over a small image set.

Operates on the images a user handed us — typically 3 to 12 — which is why this
is plain pairwise comparison over a normalised matrix and not an index. **Say
that out loud rather than reaching for FAISS:** at under a hundred vectors an
exact all-pairs comparison is both faster and less code than building an index,
and an index here would be FAISS-for-show. The recall/latency tradeoff table
that justifies an approximate index belongs in the benchmark half of this repo,
where the corpus is six figures.

Two findings, and they are different questions:

*   **Near-duplicates.** Which of these images are the same image? Sellers
    repost a photo resized, recompressed, or with a badge added, and Amazon
    penalises duplicate images across a listing. Answered by perceptual hash
    distance.
*   **Group mismatches.** Given images tagged with which SKU they belong to,
    is any image closer to a *different* SKU's images than to its own? That is
    the achievable form of "wrong image on a variation" — see the docstring on
    `find_group_mismatches` for why the obvious form is not available.

Thresholds are loaded, never inlined, and the loader refuses to hand back an
uncalibrated threshold as though it were measured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from .image_hash import ImageHashes, hamming, tile_agreement

_THRESHOLDS_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "near_duplicate" / "thresholds_v1.json"
)

Method = Literal["phash", "dhash", "tiles"]


@lru_cache(maxsize=4)
def load_thresholds(path: str | None = None) -> dict[str, Any]:
    p = Path(path) if path else _THRESHOLDS_PATH
    with open(p) as f:
        return json.load(f)


def calibrated_method(thresholds: dict[str, Any] | None = None) -> str | None:
    """The method whose threshold has actually been measured, or None.

    Returning None is the honest answer before calibration, and callers are
    expected to degrade rather than fall back to a guessed number. A detector
    reporting duplicates at an unmeasured threshold is a detector reporting
    whatever its author felt like.
    """
    thresholds = thresholds or load_thresholds()
    if not thresholds.get("calibrated"):
        return None
    return thresholds.get("selected_method")


@dataclass(frozen=True)
class DuplicatePair:
    a: int
    b: int
    method: str
    distance: int
    # Populated for the tiled method, where the natural reading is a count of
    # agreeing regions rather than a bit distance.
    tiles_agreeing: int | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "a": self.a,
            "b": self.b,
            "method": self.method,
            "distance": self.distance,
        }
        if self.tiles_agreeing is not None:
            out["tiles_agreeing"] = self.tiles_agreeing
        return out


@dataclass(frozen=True)
class GroupMismatch:
    index: int
    own_group: str
    nearest_group: str
    nearest_index: int
    distance_to_nearest: int
    distance_within_own_group: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "own_group": self.own_group,
            "nearest_group": self.nearest_group,
            "nearest_index": self.nearest_index,
            "distance_to_nearest": self.distance_to_nearest,
            "distance_within_own_group": self.distance_within_own_group,
        }


@dataclass(frozen=True)
class DedupReport:
    method: str | None
    threshold: int | None
    calibrated: bool
    pairs: list[DuplicatePair] = field(default_factory=list)
    clusters: list[list[int]] = field(default_factory=list)
    mismatches: list[GroupMismatch] = field(default_factory=list)
    # Indices excluded for having no contrast structure to hash. Reported
    # rather than silently dropped: "I could not compare these two" is a
    # different statement from "these two are not duplicates".
    skipped_featureless: list[int] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "threshold": self.threshold,
            "calibrated": self.calibrated,
            "reason": self.reason,
            "pairs": [p.as_dict() for p in self.pairs],
            "clusters": self.clusters,
            "mismatches": [m.as_dict() for m in self.mismatches],
            "skipped_featureless": self.skipped_featureless,
        }


def _distance(a: ImageHashes, b: ImageHashes, method: str, thresholds: dict[str, Any]) -> int:
    if method == "phash":
        return hamming(a.phash, b.phash)
    if method == "dhash":
        return hamming(a.dhash, b.dhash)
    if method == "tiles":
        per_tile = int(thresholds["methods"]["tiles"]["per_tile_max_bits"])
        # Expressed as a distance so every method is comparable: the number of
        # cells that DISAGREE.
        return len(a.tiles) - tile_agreement(a.tiles, b.tiles, per_tile)
    raise ValueError(f"unknown method {method!r}")


def pairwise_distances(
    hashes: list[ImageHashes], method: str, thresholds: dict[str, Any] | None = None
) -> dict[tuple[int, int], int]:
    """Every pair's distance. O(n^2), which is correct at this size."""
    thresholds = thresholds or load_thresholds()
    out: dict[tuple[int, int], int] = {}
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            out[(i, j)] = _distance(hashes[i], hashes[j], method, thresholds)
    return out


def _cluster(n: int, pairs: list[DuplicatePair]) -> list[list[int]]:
    """Connected components over the duplicate pairs, via union-find.

    Transitivity is deliberate: if A matches B and B matches C, all three are
    one duplicate set even when A and C fall just outside the threshold. A
    seller wants "these four images are the same photo", not six pair rows.
    """
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pair in pairs:
        ra, rb = find(pair.a), find(pair.b)
        if ra != rb:
            parent[rb] = ra

    buckets: dict[int, list[int]] = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(i)
    return sorted((sorted(v) for v in buckets.values() if len(v) > 1), key=lambda g: g[0])


def find_near_duplicates(
    hashes: list[ImageHashes],
    thresholds: dict[str, Any] | None = None,
    *,
    method: str | None = None,
    threshold: int | None = None,
) -> DedupReport:
    """Duplicate pairs and clusters, or an honest refusal.

    With no calibrated threshold available this returns an empty report whose
    `calibrated` flag is False and whose `reason` says why, rather than
    inventing a cutoff. The service surfaces that as an unavailable section, in
    the same spirit as labelling a degraded tool result.
    """
    thresholds = thresholds or load_thresholds()
    chosen = method or calibrated_method(thresholds)

    if chosen is None:
        return DedupReport(
            method=None,
            threshold=None,
            calibrated=False,
            reason=(
                "no calibrated threshold: run the calibration against a labeled "
                "pair set before reporting duplicates"
            ),
        )

    cfg = thresholds["methods"][chosen]
    cutoff = threshold if threshold is not None else cfg.get("max_distance")
    if cutoff is None:
        return DedupReport(
            method=chosen,
            threshold=None,
            calibrated=False,
            reason=f"method {chosen!r} has no measured max_distance",
        )

    # Exclude images with no contrast structure. Every hash here encodes
    # gradient or DCT signs, so a featureless image hashes to all-zero: a flat
    # white square and a flat grey square both come out at distance 0, which
    # would report two visibly different images as the same one.
    min_std = float(thresholds.get("featureless_guard", {}).get("min_detail_std", 0.0))
    featureless = [i for i, h in enumerate(hashes) if h.detail_std < min_std]

    pairs: list[DuplicatePair] = []
    for (i, j), dist in pairwise_distances(hashes, chosen, thresholds).items():
        if i in featureless or j in featureless:
            continue
        if dist > cutoff:
            continue
        agreeing = None
        if chosen == "tiles":
            agreeing = len(hashes[i].tiles) - dist
        pairs.append(
            DuplicatePair(a=i, b=j, method=chosen, distance=dist, tiles_agreeing=agreeing)
        )

    pairs.sort(key=lambda p: (p.distance, p.a, p.b))
    return DedupReport(
        method=chosen,
        threshold=int(cutoff),
        calibrated=bool(thresholds.get("calibrated")) and method is None,
        pairs=pairs,
        clusters=_cluster(len(hashes), pairs),
        skipped_featureless=featureless,
        reason=(
            f"{len(featureless)} image(s) had too little contrast to hash and "
            "were not compared"
            if featureless
            else ""
        ),
    )


def find_group_mismatches(
    hashes: list[ImageHashes],
    groups: list[str | None],
    thresholds: dict[str, Any] | None = None,
    *,
    method: str | None = None,
) -> list[GroupMismatch]:
    """Images whose nearest neighbour belongs to a different SKU group.

    **Why this shape and not "wrong image on a variation".** That phrasing needs
    parent/child variation data, and there is none to be had here: the caller
    supplies loose images with no family structure, and the sibling project's
    catalogue is a flat ASIN list with no parent links. Asking the user to tag
    each image with its SKU turns an unanswerable question into a precise one.

    **The distance gate is load-bearing, and its absence was a bug.** The first
    version reported a mismatch whenever an image's nearest neighbour sat in
    another group, *at any distance*. For genuinely unrelated product photos
    that is pure noise: measured on seven real images, every pairwise dHash
    distance fell between 19 and 35 bits, so which group happened to contain
    the nearest one was arbitrary — and 5 of 7 images were flagged. The finding
    only means something when the neighbour is actually close, so it is gated
    at the *calibrated duplicate threshold*. The claim is therefore precise:
    "this image is a near-duplicate of an image you tagged to a different SKU."

    Images with no group tag are skipped, not guessed at.
    """
    thresholds = thresholds or load_thresholds()
    chosen = method or calibrated_method(thresholds)
    if chosen is None or len(hashes) != len(groups):
        return []

    dists = pairwise_distances(hashes, chosen, thresholds)

    def between(i: int, j: int) -> int:
        return dists[(i, j)] if i < j else dists[(j, i)]

    cutoff = thresholds["methods"].get(chosen, {}).get("max_distance")

    min_std = float(thresholds.get("featureless_guard", {}).get("min_detail_std", 0.0))
    featureless = {i for i, h in enumerate(hashes) if h.detail_std < min_std}

    out: list[GroupMismatch] = []
    for i, own in enumerate(groups):
        if own is None or i in featureless:
            continue
        others = [
            (between(i, j), j)
            for j in range(len(hashes))
            if j != i and groups[j] and j not in featureless
        ]
        if not others:
            continue
        nearest_dist, nearest = min(others, key=lambda t: (t[0], t[1]))
        if groups[nearest] == own:
            continue
        if cutoff is not None and nearest_dist > cutoff:
            # Nearest, but not close. See the docstring: an ordering over large
            # distances carries no information about group membership.
            continue

        same_group = [d for d, j in others if groups[j] == own]
        out.append(
            GroupMismatch(
                index=i,
                own_group=own,
                nearest_group=str(groups[nearest]),
                nearest_index=nearest,
                distance_to_nearest=nearest_dist,
                distance_within_own_group=min(same_group) if same_group else None,
            )
        )
    return out
