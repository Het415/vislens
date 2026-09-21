"""Perceptual hashes for near-duplicate detection. No model, no network.

Three hashes, because they fail differently and the point is to *measure* which
one to ship rather than assert it:

*   **dHash** — horizontal gradient signs on a 9x8 grayscale. Cheap, and very
    sensitive to any global change (a resize with a different filter shifts
    bits). Good at exact-ish reposts.
*   **pHash** — a DCT on a 32x32 grayscale, keeping the low-frequency 8x8 block
    (minus DC) against its median. Robust to rescaling and recompression
    because those barely touch low frequencies. The standard choice.
*   **Tiled dHash** — a dHash per cell of a 4x4 grid, matched by counting how
    many cells agree. This is robust to **localized** edits: a badge pasted in
    one corner destroys one or two cells and leaves fourteen intact, where a
    global hash smears the change across every bit.

    A note on the last one, because it is easy to claim too much for it: tiled
    hashing does **not** buy crop robustness. A crop shifts content across cell
    boundaries, so cell *i* of one image no longer corresponds to cell *i* of
    the other. What it buys is exactly the seller-relevant case — the same photo
    reposted with a "50% OFF" badge or a logo added.

All three return 64-bit ints (tiled returns sixteen of them), compared by
Hamming distance. Nothing here picks a threshold: thresholds are calibrated
against a labeled pair set and ship as package data in `thresholds_v1.json`,
because a
threshold asserted without measurement is a number made up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import fft

# Pinned, and it matters: PIL, cv2 and torchvision-without-antialias produce
# different pixels for the same target size, which shifts hash bits. A hash
# computed with one filter is not comparable to a hash computed with another,
# so this constant is part of the hash's identity.
RESAMPLE = Image.Resampling.LANCZOS

DHASH_SIDE = 8          # 9x8 sampled -> 8x8 comparisons -> 64 bits
PHASH_IMAGE_SIDE = 32   # DCT input
PHASH_BLOCK = 8         # low-frequency block kept -> 64 bits
TILE_GRID = 4           # 4x4 = 16 tiles
HASH_BITS = 64


@dataclass(frozen=True)
class ImageHashes:
    """Every hash for one image, plus the identity of how they were computed."""

    dhash: int
    phash: int
    tiles: tuple[int, ...]
    # Standard deviation of the 32x32 grayscale. Carried because every hash here
    # encodes *contrast structure*, so an image with none produces a degenerate
    # hash. Measured: a flat white square and a flat grey square both hash to
    # 0x0000000000000000 under dHash *and* pHash, so two visibly different
    # images come out at distance 0 — a confident wrong answer. Callers use this
    # to exclude featureless images rather than report a false duplicate.
    detail_std: float = 0.0
    resample: str = RESAMPLE.name

    def as_dict(self) -> dict[str, object]:
        return {
            "dhash": f"{self.dhash:016x}",
            "phash": f"{self.phash:016x}",
            "tiles": [f"{t:016x}" for t in self.tiles],
            "detail_std": round(self.detail_std, 3),
            "resample": self.resample,
        }


def _gray(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb).convert("L")


def _bits_to_int(bits: np.ndarray) -> int:
    """Pack a boolean array into an int, most significant bit first."""
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def dhash(rgb: np.ndarray) -> int:
    """Sign of the horizontal gradient on a 9x8 grayscale thumbnail."""
    small = _gray(rgb).resize((DHASH_SIDE + 1, DHASH_SIDE), RESAMPLE)
    px = np.asarray(small, dtype=np.int16)
    return _bits_to_int(px[:, 1:] > px[:, :-1])


def phash(rgb: np.ndarray) -> int:
    """Low-frequency DCT coefficients against their median.

    The DC term is dropped. It encodes mean brightness, so keeping it would make
    the hash react to an exposure change that leaves the image recognisably the
    same — and would also dominate the median.
    """
    small = _gray(rgb).resize((PHASH_IMAGE_SIDE, PHASH_IMAGE_SIDE), RESAMPLE)
    px = np.asarray(small, dtype=np.float64)
    coeffs = fft.dctn(px, norm="ortho")[:PHASH_BLOCK, :PHASH_BLOCK]

    flat = coeffs.ravel()[1:]  # drop DC
    median = np.median(flat)
    bits = np.concatenate(([False], flat > median))
    return _bits_to_int(bits)


def tiled_dhash(rgb: np.ndarray, grid: int = TILE_GRID) -> tuple[int, ...]:
    """One dHash per cell of a `grid` x `grid` split.

    Cells are cut from the full-resolution array rather than from a thumbnail,
    so a small pasted badge stays confined to the cell it occupies instead of
    being resampled across several.
    """
    height, width = rgb.shape[:2]
    ys = np.linspace(0, height, grid + 1).astype(int)
    xs = np.linspace(0, width, grid + 1).astype(int)

    hashes: list[int] = []
    for row in range(grid):
        for col in range(grid):
            cell = rgb[ys[row] : ys[row + 1], xs[col] : xs[col + 1]]
            if cell.size == 0:
                hashes.append(0)
                continue
            hashes.append(dhash(cell))
    return tuple(hashes)


def detail_std(rgb: np.ndarray) -> float:
    """Contrast structure, as the std of a 32x32 grayscale.

    The quantity every hash here depends on. Near zero means the hashes carry
    no information about the image.
    """
    small = _gray(rgb).resize((PHASH_IMAGE_SIDE, PHASH_IMAGE_SIDE), RESAMPLE)
    return float(np.asarray(small, dtype=np.float64).std())


def compute_hashes(rgb: np.ndarray) -> ImageHashes:
    return ImageHashes(
        dhash=dhash(rgb),
        phash=phash(rgb),
        tiles=tiled_dhash(rgb),
        detail_std=detail_std(rgb),
    )


def hamming(a: int, b: int) -> int:
    """Bit distance. `int.bit_count` is exact and needs no table."""
    return (a ^ b).bit_count()


def tile_agreement(a: tuple[int, ...], b: tuple[int, ...], per_tile_max: int) -> int:
    """How many cells agree within `per_tile_max` bits.

    Returns a count, not a distance, because that is the quantity with a natural
    reading: "14 of 16 regions of these two images are the same".
    """
    if len(a) != len(b):
        raise ValueError(f"tile counts differ: {len(a)} vs {len(b)}")
    return sum(1 for x, y in zip(a, b, strict=True) if hamming(x, y) <= per_tile_max)


def normalised_distance(a: int, b: int) -> float:
    """Hamming distance as a fraction of the hash width, for reporting."""
    return hamming(a, b) / HASH_BITS
