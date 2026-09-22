"""The training loop.

Everything here that looks defensive is in response to a specific way a
headless 12-hour Kaggle commit fails, where the only evidence afterwards is a
log and whatever was written to disk:

*   **fp16 + `GradScaler`, never bf16.** A T4 is `sm_75`; bf16 needs `sm_80`+.
    AMP is enabled only on CUDA — on MPS and CPU it is off, so the local smoke
    path exercises the same code in fp32 rather than a different branch.
*   **`logit_scale` clamped after every step**, not inside forward. Unclamped
    it grows until fp16 logits overflow.
*   **The loss is computed in fp32** — see `loss.py`.
*   **Checkpoints are atomic**: `.tmp` then `os.replace`. A commit killed
    mid-write must not leave a corrupt checkpoint, because the next run
    resumes from it.
*   **`--max-minutes` exits cleanly** before Kaggle's 12-hour kill, so the run
    ends having written a checkpoint and a metrics row rather than being
    terminated mid-epoch.
*   **Best-on-val, not last-epoch.** The build plan asks for this explicitly.

Metrics go to a CSV that is appended after every validation, so a run killed
anyway still leaves everything it had measured up to that point.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data.shards import make_dataset
from ..data.transforms import describe as describe_preprocess
from .config import TrainConfig
from .loss import info_nce


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_collate(text_lookup: dict[str, int], text_matrix: torch.Tensor):
    """Batch the shard samples and attach their precomputed text embeddings.

    Samples whose key is missing from the lookup are DROPPED, loudly in the
    count rather than silently: a missing embedding means the precompute and
    the shards disagree, and quietly training on a zero vector would be a
    slow, invisible corruption of the target space.

    A batch that shrinks below two usable samples is dropped ENTIRELY, and that
    is not fastidiousness. InfoNCE over a 1x1 similarity matrix has no
    negatives: the only logit is the positive, so cross-entropy is exactly 0.0
    and accuracy exactly 1.0. The first smoke run produced rows reading
    `loss=0.0, batch_acc_i2t=1.0` and they look like a model that has solved
    the task. A metric that reports perfection when it has measured nothing is
    worse than a crash.
    """

    def collate(samples: list[dict]) -> dict | None:
        usable = [s for s in samples if s["key"] in text_lookup]
        if len(usable) < 2:
            return None
        rows = torch.tensor([text_lookup[s["key"]] for s in usable], dtype=torch.long)
        return {
            "pixels": torch.from_numpy(np.stack([s["pixels"] for s in usable])),
            "text": text_matrix[rows],
            "product_hash": torch.tensor([s["product_hash"] for s in usable], dtype=torch.long),
            "title_hash": torch.tensor([s["title_hash"] for s in usable], dtype=torch.long),
            "dropped": len(samples) - len(usable),
        }

    return collate


def build_loader(
    config: TrainConfig,
    split: str,
    collate,
    shuffle: bool,
) -> DataLoader:
    dataset = make_dataset(
        config.shards_dir,
        role="pairs",
        split=split,
        shuffle_buffer=config.shuffle_buffer if shuffle else 0,
        seed=config.seed,
    )
    return DataLoader(
        dataset.batched(config.batch_size, collation_fn=collate, partial=False),
        batch_size=None,
        num_workers=config.num_workers,
    )


def lr_at(step: int, base: float, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay, the CLIP-standard schedule."""
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    if total <= warmup:
        return base
    progress = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))


def atomic_save(state: dict, path: Path) -> None:
    """`.tmp` then `os.replace`. The replace is atomic on POSIX."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, config: TrainConfig) -> dict:
    model.eval()
    totals = {"loss": 0.0, "batch_acc_i2t": 0.0}
    seen = 0
    for batch in loader:
        if batch is None:
            continue
        image, text, scale = model(batch["pixels"].to(device), batch["text"].to(device))
        _, stats = info_nce(
            image,
            text,
            scale,
            product_ids=batch["product_hash"].to(device) if config.mask_false_negatives else None,
            title_hashes=batch["title_hash"].to(device) if config.mask_false_negatives else None,
        )
        for key in totals:
            totals[key] += stats[key]
        seen += 1
    model.train()
    if not seen:
        return {"val_loss": float("nan"), "val_batch_acc": float("nan"), "val_batches": 0}
    return {
        "val_loss": totals["loss"] / seen,
        "val_batch_acc": totals["batch_acc_i2t"] / seen,
        "val_batches": seen,
    }


def train(
    model: nn.Module,
    config: TrainConfig,
    text_lookup: dict[str, int],
    text_matrix: torch.Tensor,
    device: str | None = None,
) -> dict:
    device = device or pick_device()
    seed_everything(config.seed)
    model = model.to(device)
    model.train()

    run_dir = Path(config.runs_dir) / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({**asdict(config), "preprocess": describe_preprocess()}, indent=2, default=str)
        + "\n"
    )

    collate = make_collate(text_lookup, text_matrix)
    train_loader = build_loader(config, "train", collate, shuffle=True)
    val_loader = build_loader(config, "val", collate, shuffle=False)

    # Two parameter groups: the pretrained trunk must move far more slowly than
    # a randomly initialised projection head, and a single global LR cannot be
    # right for both.
    head_params = [p for n, p in model.named_parameters() if "trunk" not in n]
    trunk_params = [p for n, p in model.named_parameters() if "trunk" in n]
    optimizer = torch.optim.AdamW(
        [
            {"params": trunk_params, "lr": config.trunk_lr},
            {"params": head_params, "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )

    use_amp = config.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    metrics_path = run_dir / "metrics.csv"
    new_file = not metrics_path.exists()
    metrics_file = metrics_path.open("a", newline="")
    writer = csv.DictWriter(
        metrics_file,
        fieldnames=[
            "step",
            "epoch",
            "loss",
            "batch_acc_i2t",
            "logit_scale",
            "false_negatives_masked",
            "lr_trunk",
            "val_loss",
            "val_batch_acc",
            "minutes",
        ],
    )
    if new_file:
        writer.writeheader()

    started = time.time()
    step = 0
    best_val = float("inf")
    stopped = "completed"
    total_steps = (config.subset_batches or 1000) * config.epochs

    for epoch in range(config.epochs):
        for batch in train_loader:
            if batch is None:
                continue
            minutes = (time.time() - started) / 60
            if minutes >= config.max_minutes:
                stopped = "max_minutes"
                break
            if config.subset_batches and step >= config.subset_batches * (epoch + 1):
                break

            lr_trunk = lr_at(step, config.trunk_lr, config.warmup_steps, total_steps)
            optimizer.param_groups[0]["lr"] = lr_trunk
            optimizer.param_groups[1]["lr"] = lr_at(
                step, config.head_lr, config.warmup_steps, total_steps
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                image, text, scale = model(batch["pixels"].to(device), batch["text"].to(device))
            loss, stats = info_nce(
                image,
                text,
                scale,
                product_ids=batch["product_hash"].to(device)
                if config.mask_false_negatives
                else None,
                title_hashes=batch["title_hash"].to(device)
                if config.mask_false_negatives
                else None,
            )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            # After the step, never inside forward — see towers.py.
            model.clamp_logit_scale()

            step += 1
            if step % 50 == 0 or config.subset_batches:
                writer.writerow(
                    {
                        "step": step,
                        "epoch": epoch,
                        **{
                            k: stats[k]
                            for k in (
                                "loss",
                                "batch_acc_i2t",
                                "logit_scale",
                                "false_negatives_masked",
                            )
                        },
                        "lr_trunk": lr_trunk,
                        "val_loss": "",
                        "val_batch_acc": "",
                        "minutes": round(minutes, 2),
                    }
                )
                metrics_file.flush()

        val = evaluate(model, val_loader, device, config)
        writer.writerow(
            {
                "step": step,
                "epoch": epoch,
                "loss": "",
                "batch_acc_i2t": "",
                "logit_scale": "",
                "false_negatives_masked": "",
                "lr_trunk": "",
                "val_loss": val["val_loss"],
                "val_batch_acc": val["val_batch_acc"],
                "minutes": round((time.time() - started) / 60, 2),
            }
        )
        metrics_file.flush()

        # Best on val, not last epoch.
        if val["val_loss"] == val["val_loss"] and val["val_loss"] < best_val:
            best_val = val["val_loss"]
            atomic_save(
                {
                    "model": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch,
                    "step": step,
                },
                run_dir / "best.pt",
            )
        atomic_save(
            {"model": model.state_dict(), "config": asdict(config), "epoch": epoch, "step": step},
            run_dir / "last.pt",
        )
        if stopped == "max_minutes":
            break

    metrics_file.close()
    summary = {
        "run_id": config.run_id,
        "stopped": stopped,
        "steps": step,
        "best_val_loss": best_val if best_val < float("inf") else None,
        "minutes": round((time.time() - started) / 60, 2),
        "device": device,
        "amp": use_amp,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
