"""The training half: preprocessing, shard reading, the loss, and the loop.

Everything here runs on CPU with a stub trunk. CI has no GPU and no network,
so nothing may download CLIP weights — which is exactly why `ImageTower` takes
an injected trunk rather than constructing one.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

# torch lives in the `train` extra only (CLAUDE.md section 2), so the default
# CI job — which asserts torch is ABSENT from a serving install — must not fail
# collecting this file. The dedicated training job installs `[train]` and
# asserts torch imports before calling pytest, so a silently skipped suite
# cannot hide behind this.
torch = pytest.importorskip("torch", reason="requires the [train] extra")

from PIL import Image  # noqa: E402
from torch import nn  # noqa: E402

from vislens.data.shards import make_dataset, shard_paths, title_hash  # noqa: E402
from vislens.data.transforms import (  # noqa: E402
    IMAGE_SIZE,
    preprocess,
    preprocess_config,
    write_preprocess_json,
)  # noqa: E402
from vislens.models.towers import (  # noqa: E402
    MAX_LOGIT_SCALE,
    ImageTower,
    TextProjection,
    TwoTower,
)
from vislens.train.config import TrainConfig  # noqa: E402
from vislens.train.loop import lr_at, make_collate, train  # noqa: E402
from vislens.train.loss import false_negative_mask, info_nce  # noqa: E402

# ── Preprocessing: the single definition ──────────────────────────────────────


@pytest.mark.parametrize("size", [(800, 600), (256, 256), (92, 65), (1200, 300), (64, 2000)])
def test_any_aspect_ratio_gives_the_model_input_shape(size):
    out = preprocess(Image.new("RGB", size, (200, 100, 50)))
    assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_transparency_is_composited_onto_white_not_dropped():
    """A transparent PNG converted straight to RGB goes BLACK where it was
    transparent. On product photography that is a black silhouette where the
    background belongs — the same trap the audit's `had_alpha` check exists
    to surface, and it would poison training silently."""
    rgba = Image.new("RGBA", (300, 300), (255, 0, 0, 0))
    out = preprocess(rgba)
    assert out.min() > 0.99, "transparent pixels should composite to white"


def test_normalization_is_not_applied_here():
    """It belongs inside the ONNX graph so a consumer cannot forget it — and
    must therefore NOT also happen here, or it happens twice."""
    cfg = preprocess_config()
    assert cfg.normalization_in_graph is True
    out = preprocess(Image.new("RGB", (224, 224), (255, 255, 255)))
    assert out.max() == pytest.approx(1.0), "white should be 1.0, not mean-subtracted"


def test_preprocess_json_carries_what_a_consumer_needs(tmp_path):
    written = json.loads(write_preprocess_json(tmp_path / "preprocess.json").read_text())
    for key in ("image_size", "interpolation", "mean", "std", "l2_normalize_in_graph"):
        assert key in written
    assert written["interpolation"] == "bicubic"


# ── Shard reading ─────────────────────────────────────────────────────────────


def _write_shard(path: Path, n: int = 6) -> None:
    """A tar in exactly the layout `scripts/pack_shards.py` writes."""
    with tarfile.open(path, "w") as tf:
        for i in range(n):
            key = f"pairs/PROD{i:03d}-00"
            buf = io.BytesIO()
            Image.new("RGB", (300, 200), (i * 30 % 255, 80, 120)).save(buf, "JPEG")
            payloads = {
                f"{key}.jpg": buf.getvalue(),
                f"{key}.json": json.dumps(
                    {"product_id": f"PROD{i:03d}", "split": "val", "path": f"ab/{i}.jpg"}
                ).encode(),
                f"{key}.txt": f"Product number {i}".encode(),
            }
            for name, payload in payloads.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))


def test_shard_paths_are_sorted_and_validated(tmp_path):
    for n in (1, 0):
        _write_shard(tmp_path / f"pairs-val-00000{n}.tar")
    assert [Path(p).name for p in shard_paths(tmp_path, "pairs", "val")] == [
        "pairs-val-000000.tar",
        "pairs-val-000001.tar",
    ]
    with pytest.raises(ValueError):
        shard_paths(tmp_path, "nonsense", "val")


def test_samples_carry_the_fields_the_loss_needs(tmp_path):
    _write_shard(tmp_path / "pairs-val-000000.tar")
    sample = next(iter(make_dataset(tmp_path, "pairs", "val")))
    assert sample["pixels"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert sample["product_id"].startswith("PROD")
    assert sample["product_hash"] and sample["title_hash"]
    assert sample["key"].startswith("pairs/")


def test_title_hash_normalises_case_and_whitespace_only():
    assert title_hash("Echo  Dot") == title_hash("echo dot")
    assert title_hash("Echo Dot") != title_hash("Echo Dots")


# ── The loss, and the masking the build plan omits ────────────────────────────


def test_a_perfect_model_scores_zero():
    e = torch.nn.functional.normalize(torch.randn(8, 32), dim=-1)
    _, stats = info_nce(e, e, torch.tensor(20.0))
    assert stats["loss"] == pytest.approx(0.0, abs=1e-4)
    assert stats["batch_acc_i2t"] == pytest.approx(1.0)


def test_the_diagonal_is_never_masked():
    """It is the positive. Masking it removes the numerator."""
    ids = torch.zeros(5, dtype=torch.long)  # every row the same product
    mask = false_negative_mask(ids, torch.arange(1, 6))
    assert not mask.diagonal().any()
    assert mask.sum() == 20  # everything off-diagonal


def test_absent_titles_do_not_merge():
    """`index`-role samples hash to 0. Two zeros are not the same title."""
    mask = false_negative_mask(torch.arange(4), torch.zeros(4, dtype=torch.long))
    assert mask.sum() == 0


def test_masking_a_duplicate_lowers_the_loss():
    """Because an impossible target has been removed, not because the model
    improved. If this ever stops holding, the mask is not being applied."""
    torch.manual_seed(0)
    img = torch.nn.functional.normalize(torch.randn(8, 32), dim=-1)
    txt = torch.nn.functional.normalize(torch.randn(8, 32), dim=-1)
    duplicated = torch.tensor([0, 0, 2, 3, 4, 5, 6, 7])
    plain, _ = info_nce(img, txt, torch.tensor(20.0))
    masked, stats = info_nce(
        img, txt, torch.tensor(20.0), product_ids=duplicated, title_hashes=torch.arange(1, 9)
    )
    assert masked < plain
    assert stats["false_negatives_masked"] == 2


def test_the_loss_is_fp32_even_when_the_inputs_are_half():
    half = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1).half()
    loss, _ = info_nce(half, half, torch.tensor(100.0).half())
    assert loss.dtype == torch.float32


# ── Model ─────────────────────────────────────────────────────────────────────


def _stub_model(text_dim: int = 512, dim: int = 32) -> TwoTower:
    trunk = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(3 * 16, 64))
    return TwoTower(ImageTower(trunk, 64, dim), TextProjection(text_dim, dim))


def test_both_towers_are_l2_normalised():
    model = _stub_model()
    image, text, _ = model(torch.randn(4, 3, 224, 224), torch.randn(4, 512))
    assert torch.allclose(image.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(text.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_logit_scale_is_clamped_to_clips_ceiling():
    """Unclamped it grows until fp16 logits overflow, and the loss NaNs hours
    into a headless run."""
    model = _stub_model()
    model.logit_scale.data.fill_(50.0)
    model.clamp_logit_scale()
    assert float(model.logit_scale.detach()) == pytest.approx(MAX_LOGIT_SCALE)


# ── Config identity ───────────────────────────────────────────────────────────


def test_run_id_ignores_where_a_run_happens_but_not_what_it_computes():
    base = TrainConfig()
    assert base.run_id == TrainConfig(num_workers=16, shards_dir="/elsewhere").run_id
    assert base.run_id != TrainConfig(batch_size=128).run_id
    assert base.run_id != TrainConfig(mask_false_negatives=False).run_id


def test_warmup_then_cosine():
    assert lr_at(0, 1.0, 10, 100) < lr_at(9, 1.0, 10, 100)
    assert lr_at(10, 1.0, 10, 100) == pytest.approx(1.0)
    assert lr_at(99, 1.0, 10, 100) < 0.01


# ── The loop ──────────────────────────────────────────────────────────────────


def test_a_batch_with_one_usable_sample_is_dropped():
    """The regression that matters most here.

    InfoNCE over a 1x1 matrix has no negatives: cross-entropy is exactly 0.0
    and accuracy exactly 1.0. The first smoke run emitted rows reading
    `loss=0.0, batch_acc_i2t=1.0`, which look like a solved task. A metric that
    reports perfection when it measured nothing is worse than a crash.
    """
    collate = make_collate({"a": 0, "b": 1}, torch.randn(2, 8))
    sample = lambda key: {  # noqa: E731
        "key": key,
        "pixels": np.zeros((3, 224, 224), dtype=np.float32),
        "product_hash": 1,
        "title_hash": 2,
    }
    assert collate([sample("a"), sample("missing")]) is None
    assert collate([sample("missing")]) is None
    batch = collate([sample("a"), sample("b")])
    assert batch is not None and batch["pixels"].shape[0] == 2


def test_train_writes_a_complete_run_record(tmp_path):
    """Config, metrics and both checkpoints — the run record IS the result,
    since there is no Weights & Biases to fall back on."""
    shards = tmp_path / "shards"
    shards.mkdir()
    for split in ("train", "val"):
        _write_shard(shards / f"pairs-{split}-000000.tar", n=8)

    keys = [f"pairs/PROD{i:03d}-00" for i in range(8)]
    lookup = {k: i for i, k in enumerate(keys)}
    matrix = torch.nn.functional.normalize(torch.randn(len(keys), 512), dim=-1)

    config = TrainConfig(
        shards_dir=str(shards),
        runs_dir=str(tmp_path / "runs"),
        batch_size=4,
        epochs=1,
        subset_batches=2,
        num_workers=0,
        shuffle_buffer=0,
        warmup_steps=1,
    )
    summary = train(_stub_model(), config, lookup, matrix, device="cpu")

    run = Path(config.runs_dir) / config.run_id
    assert summary["stopped"] == "completed"
    for artifact in ("config.json", "metrics.csv", "best.pt", "last.pt", "summary.json"):
        assert (run / artifact).exists(), artifact

    rows = (run / "metrics.csv").read_text().strip().splitlines()
    assert len(rows) > 1
    # The preprocessing config travels with the run, so a checkpoint can never
    # be replayed under a different resize than it was trained on.
    assert "preprocess" in json.loads((run / "config.json").read_text())


# ── Text embeddings ───────────────────────────────────────────────────────────


def test_text_embeddings_round_trip_through_fp16(tmp_path):
    """Stored as fp16 to halve a 124 MB file that has to reach Kaggle; loaded
    as fp32 because the cast is free next to a forward pass."""
    from vislens.data.text_emb import load_text_embeddings, save_text_embeddings

    keys = [f"pairs/PROD{i:03d}-00" for i in range(5)]
    original = np.random.RandomState(0).randn(5, 512).astype(np.float32)
    original /= np.linalg.norm(original, axis=1, keepdims=True)

    path = save_text_embeddings(tmp_path / "val.npz", keys, original, meta={"model": "ViT-B-32"})
    lookup, matrix = load_text_embeddings(path)

    assert lookup == {k: i for i, k in enumerate(keys)}
    assert matrix.dtype == torch.float32
    # fp16 keeps ~3 decimal digits; enough for an input to a trained projection.
    assert torch.allclose(matrix, torch.from_numpy(original), atol=1e-3)
    assert torch.allclose(matrix.norm(dim=-1), torch.ones(5), atol=1e-2)


def test_mismatched_keys_and_rows_are_refused(tmp_path):
    """Silent misalignment would attach every embedding to the wrong image."""
    from vislens.data.text_emb import save_text_embeddings

    with pytest.raises(ValueError):
        save_text_embeddings(tmp_path / "bad.npz", ["a", "b"], np.zeros((3, 4), dtype=np.float32))


def test_encode_deduplicates_but_returns_one_row_per_input():
    """A frozen encoder maps identical strings to identical vectors, so
    encoding repeats is waste — but the file must still be one row per key, or
    every consumer has to understand the deduplication."""
    from scripts.precompute_text import encode

    seen: list[list[str]] = []

    def tokenizer(chunk):
        seen.append(list(chunk))
        return torch.arange(len(chunk))

    def encoder(tokens):
        # Distinct in DIRECTION, not just magnitude: `encode` L2-normalises, so
        # two constant vectors of different scale would normalise to the same
        # point and the test would be asserting nothing.
        return torch.stack([torch.eye(8)[int(t) % 8] + 0.1 for t in tokens])

    titles = ["red shoe", "blue sofa", "red shoe", "red shoe", "blue sofa"]
    out = encode(titles, encoder, tokenizer, device="cpu")

    assert out.shape == (5, 8)
    assert sum(len(c) for c in seen) == 2, "only the unique titles should be encoded"
    # Equal titles must come back as equal rows.
    assert np.allclose(out[0], out[2]) and np.allclose(out[0], out[3])
    assert np.allclose(out[1], out[4])
    assert not np.allclose(out[0], out[1])


def test_a_pairs_sample_without_a_title_is_an_error(tmp_path, monkeypatch):
    """Not a skip. Skipping would shift every key/row pair after it by one,
    and the embeddings would load fine attached to the wrong images."""
    import scripts.precompute_text as pre

    monkeypatch.setattr(
        pre, "iter_samples", lambda *a, **k: iter([{"key": "pairs/X-00", "title": ""}])
    )
    with pytest.raises(ValueError, match="no title"):
        pre.collect("val", None)
