"""Run configuration, and the hash that identifies it.

`CLAUDE.md` section 11: no Weights & Biases. A run is identified by the hash of
its own config and its record is committed as CSV/JSON, so the numbers sit
where a reviewer reads them rather than behind a login. The hash covers every
field that can change a result — change one and you get a new run directory
rather than quietly overwriting the old one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class TrainConfig:
    shards_dir: str = "data/shards"
    text_emb_dir: str = "data/text_emb"
    runs_dir: str = "src/vislens/runs"

    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    projection_dim: int = 256

    # 256 because the text tower is frozen and precomputed, which is what makes
    # it fit a free T4. Contrastive negatives come from the batch, so this is a
    # quality parameter, not just a throughput one.
    batch_size: int = 256
    epochs: int = 4
    trunk_lr: float = 1e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.1
    warmup_steps: int = 200
    grad_clip: float = 1.0

    seed: int = 0
    num_workers: int = 2
    shuffle_buffer: int = 4000

    mask_false_negatives: bool = True
    amp: bool = True

    # Kaggle kills a "Save & Run All" commit at 12 hours. Exiting at 11h20m
    # means the run terminates on its own terms, having written a checkpoint,
    # rather than being killed mid-write.
    max_minutes: float = 680.0
    subset_batches: int | None = None

    notes: str = ""
    excluded_from_hash: tuple[str, ...] = field(
        # Paths and worker counts change where a run happens, not what it
        # computes. Including them would give the same experiment two hashes.
        default=("shards_dir", "text_emb_dir", "runs_dir", "num_workers", "notes", "max_minutes"),
        repr=False,
    )

    def hashable(self) -> dict:
        data = asdict(self)
        data.pop("excluded_from_hash", None)
        for key in self.excluded_from_hash:
            data.pop(key, None)
        return data

    @property
    def run_id(self) -> str:
        blob = json.dumps(self.hashable(), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
