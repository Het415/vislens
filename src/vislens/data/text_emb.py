"""Reading and writing the frozen text embeddings.

The text tower is frozen (see `models/towers.py`), so its output is computed
once and reused for every epoch. That is what removes the text forward pass
from the training step and buys the batch size that contrastive learning
actually needs.

**Keyed by the shard's own sample key**, never by anything re-derived from the
parquet. `pack_shards.py` builds a key as `{product_id}-{occurrence:02d}`, and
a second implementation of that rule living here is exactly the kind of
duplicate definition that drifts silently — the embeddings would still load,
just for the wrong rows. So the precompute reads the shards.

Stored as float16. These are inputs to a trainable projection, not weights
being optimised, so the precision is not doing any work — and it halves a
124 MB file to 62 MB, which matters when it has to be uploaded to Kaggle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def embeddings_path(text_emb_dir: str | Path, split: str) -> Path:
    return Path(text_emb_dir) / f"{split}.npz"


def save_text_embeddings(
    path: str | Path,
    keys: list[str],
    embeddings: np.ndarray,
    meta: dict | None = None,
) -> Path:
    if len(keys) != embeddings.shape[0]:
        raise ValueError(f"{len(keys)} keys against {embeddings.shape[0]} rows")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        keys=np.array(keys, dtype=object),
        emb=embeddings.astype(np.float16),
        # The model and tag are recorded so a run cannot silently mix
        # embeddings from one text encoder with a checkpoint trained against
        # another — they would load fine and quietly mean nothing.
        meta=np.array([str(meta or {})], dtype=object),
    )
    return out


def load_text_embeddings(path: str | Path):
    """Returns `(lookup, matrix)` — key to row, and the rows as float32.

    Cast to float32 on load rather than stored that way: fp16 halves the file
    on disk and the cast is free next to the forward pass.
    """
    import torch

    with np.load(Path(path), allow_pickle=True) as data:
        keys = [str(k) for k in data["keys"]]
        matrix = torch.from_numpy(data["emb"].astype(np.float32))
    return {key: i for i, key in enumerate(keys)}, matrix
