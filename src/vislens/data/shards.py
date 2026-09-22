"""Reading the packed WebDataset shards.

`scripts/pack_shards.py` wrote them; this reads them back. The contract it
relies on is stated there and repeated here because breaking it is silent:
every sample is a group of consecutive tar members sharing a key, the image is
`.jpg` (two of 397,170 are `.png`), the metadata is `.json`, and a `pairs`
sample additionally carries the title in `.txt`.

Decoding is OURS, not WebDataset's. `.decode("pil")` would hand back an image
resized by whatever happens to be installed; `vislens.data.transforms` is the
single definition (`CLAUDE.md` section 9) and the whole point is that training
and serving cannot disagree about it. So the pipeline takes raw bytes and calls
`preprocess_bytes`.

Three fields travel beside the pixels, and each is load-bearing:

*   `product_id` and `title_hash` — the false-negative mask. Two rows for the
    same ASIN across marketplaces, or two different ASINs with an identical
    title, are not negatives; counted as such they put a ceiling on the metric
    that no amount of training removes. The build plan does not mention this
    and it is the single easiest way to get an uninterpretable result.
*   `key` — the join back to the precomputed text embeddings, which are keyed
    by the shard's own sample key precisely so no second key-derivation exists
    to drift from `pack_shards`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import webdataset as wds

from .transforms import preprocess_bytes

ROLES = ("pairs", "index")
SPLITS = ("train", "val", "test")


def shard_paths(shards_dir: str | Path, role: str, split: str) -> list[str]:
    """The shards for one series, in a stable order.

    Sorted, so a run is reproducible and two processes reading the same series
    agree about what shard 0 is. WebDataset shuffles shard ORDER when asked;
    it should never depend on the filesystem's.
    """
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}, expected one of {ROLES}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    found = sorted(Path(shards_dir).glob(f"{role}-{split}-*.tar"))
    return [str(p) for p in found]


def title_hash(title: str) -> int:
    """A stable 63-bit hash of a normalised title, for false-negative masking.

    Normalised on case and whitespace only. Anything cleverer (stripping
    punctuation, stemming) would start merging titles that are genuinely
    different products, which turns a mask that prevents a ceiling into one
    that hides real negatives.
    """
    normalised = " ".join(title.lower().split())
    digest = hashlib.sha1(normalised.encode("utf-8")).digest()
    # 63 bits, so it always fits a signed int64 tensor without wrapping.
    return int.from_bytes(digest[:8], "big") >> 1


def _to_sample(item: dict) -> dict:
    """One WebDataset dict to the fields the training loop wants."""
    image_bytes = item.get("jpg") or item.get("png")
    if image_bytes is None:  # pragma: no cover - defensive
        raise KeyError(f"sample {item.get('__key__')!r} carries no jpg or png")

    meta = json.loads(item["json"])
    title = item.get("txt", b"").decode("utf-8") if "txt" in item else ""

    product_id = meta.get("product_id", "")
    return {
        "pixels": preprocess_bytes(image_bytes),
        "key": item["__key__"],
        "product_id": product_id,
        # Hashed here, not in the collate: the mask is a tensor comparison and
        # the alternative is a Python loop over strings once per batch.
        "product_hash": title_hash(product_id) if product_id else 0,
        "title": title,
        "title_hash": title_hash(title) if title else 0,
    }


def make_dataset(
    shards_dir: str | Path,
    role: str = "pairs",
    split: str = "train",
    *,
    shuffle_buffer: int = 0,
    seed: int = 0,
) -> wds.WebDataset:
    """A WebDataset pipeline over one series.

    `shuffle_buffer` is 0 for val and test on purpose: a deterministic order is
    what makes two eval runs comparable. For train, a buffer of a few thousand
    is plenty — the shards are already content-hash ordered, so a shard is
    itself a random sample of the corpus rather than a block of one product
    type, and the shuffle is topping that up rather than doing the whole job.
    """
    urls = shard_paths(shards_dir, role, split)
    if not urls:
        raise FileNotFoundError(f"no {role}-{split}-*.tar under {shards_dir}")

    dataset = wds.WebDataset(
        urls,
        shardshuffle=bool(shuffle_buffer),
        # `detshuffle` so a seed reproduces a run. Non-deterministic shuffling
        # is how two runs of "the same" config stop being comparable — the same
        # class of bug as the catalog split not being reproducible.
        detshuffle=bool(shuffle_buffer),
        seed=seed,
        handler=wds.warn_and_continue,
    )
    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer)
    return dataset.map(_to_sample, handler=wds.warn_and_continue)


def iter_samples(
    shards_dir: str | Path,
    role: str = "pairs",
    split: str = "train",
    limit: int | None = None,
) -> Iterator[dict]:
    """Plain iteration, for the precompute pass and for tests."""
    for n, sample in enumerate(make_dataset(shards_dir, role, split)):
        if limit is not None and n >= limit:
            return
        yield sample
