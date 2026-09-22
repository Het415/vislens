"""Precompute the frozen text embeddings, one row per shard sample.

The text tower does not train (see `vislens/models/towers.py`), so encoding its
titles once and reading them back is both correct and the thing that makes
batch 256 fit a free T4 — there is no text forward pass in the training step at
all.

**Reads the shards, not `pairs.parquet`.** The row key has to match the key
WebDataset will yield at training time, and `pack_shards.py` owns how that key
is built. Deriving it a second time here would be a duplicate definition of a
rule that has already bitten this repo once: the embeddings would load happily
and be attached to the wrong images.

Identical titles are encoded once. Roughly one title in six repeats across the
catalogue ("Amazon Brand - Solimo Designer …" variants), and a frozen encoder
maps identical strings to identical vectors, so encoding them separately is
pure waste. The output still stores one row per key — a consumer should not
have to understand the deduplication to use the file.

Needs network on first run: it downloads CLIP's weights. Never run in CI.

Usage:
    python -m scripts.precompute_text                      # all three splits
    python -m scripts.precompute_text --split val
    python -m scripts.precompute_text --limit 500          # local smoke
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vislens.data.shards import iter_samples  # noqa: E402
from vislens.data.text_emb import embeddings_path, save_text_embeddings  # noqa: E402

SHARDS = pathlib.Path(os.getenv("VISLENS_SHARDS_DIR", ROOT / "data" / "shards"))
OUT = pathlib.Path(os.getenv("VISLENS_TEXT_EMB_DIR", ROOT / "data" / "text_emb"))

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
BATCH = 256


def collect(split: str, limit: int | None) -> tuple[list[str], list[str]]:
    """Every (key, title) in a split's pairs shards, in shard order."""
    keys: list[str] = []
    titles: list[str] = []
    for sample in iter_samples(SHARDS, role="pairs", split=split, limit=limit):
        if not sample["title"]:
            # A pairs sample without a title should not exist — the series is
            # defined as the rows that have one. Loud, because a silent skip
            # would put the key/row alignment off by one for everything after.
            raise ValueError(f"{sample['key']} is in a pairs shard with no title")
        keys.append(sample["key"])
        titles.append(sample["title"])
    return keys, titles


def encode(titles: list[str], encoder, tokenizer, device: str) -> np.ndarray:
    """Encode unique titles once, then expand back to one row per input."""
    import torch

    unique = sorted(set(titles))
    index = {title: i for i, title in enumerate(unique)}
    rows = []
    started = time.time()
    for start in range(0, len(unique), BATCH):
        chunk = unique[start : start + BATCH]
        with torch.no_grad():
            tokens = tokenizer(chunk).to(device)
            features = encoder(tokens)
            # L2-normalised at rest. The zero-shot baseline consumes these
            # directly, and a normalised store means the baseline and the
            # training input cannot disagree about whether it was done.
            features = features / features.norm(dim=-1, keepdim=True)
        rows.append(features.float().cpu().numpy())
        done = min(start + BATCH, len(unique))
        if done % (BATCH * 20) == 0 or done == len(unique):
            rate = done / max(time.time() - started, 1e-9)
            print(f"    {done:,}/{len(unique):,} unique titles ({rate:,.0f}/s)")

    matrix = np.concatenate(rows, axis=0)
    return matrix[np.array([index[t] for t in titles], dtype=np.int64)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), action="append")
    parser.add_argument("--limit", type=int, default=None, help="cap samples, for a local smoke")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    splits = args.split or ["train", "val", "test"]

    import torch

    from vislens.models.towers import load_clip

    device = args.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"loading {MODEL_NAME}/{PRETRAINED} on {device} ...")
    model, tokenizer = load_clip(MODEL_NAME, PRETRAINED, device=device)

    report = {}
    for split in splits:
        print(f"\n{split}:")
        keys, titles = collect(split, args.limit)
        if not keys:
            print("  no samples, skipping")
            continue
        print(f"  {len(keys):,} samples, {len(set(titles)):,} unique titles")
        matrix = encode(titles, model.encode_text, tokenizer, device)

        path = embeddings_path(OUT, split)
        save_text_embeddings(
            path,
            keys,
            matrix,
            meta={
                "model": MODEL_NAME,
                "pretrained": PRETRAINED,
                "normalized": True,
                "built": dt.date.today().isoformat(),
                "limit": args.limit,
            },
        )
        report[split] = {
            "samples": len(keys),
            "unique_titles": len(set(titles)),
            "dim": int(matrix.shape[1]),
            "megabytes": round(path.stat().st_size / 1e6, 1),
        }
        print(f"  wrote {path} ({report[split]['megabytes']} MB)")

    print()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
