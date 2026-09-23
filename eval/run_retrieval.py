"""Text-to-image retrieval eval, with every baseline computed at runtime.

    python -m eval.run_retrieval --limit 2000            # local smoke
    python -m eval.run_retrieval --split test            # the real thing
    python -m eval.run_retrieval --checkpoint runs/<id>/best.pt

Produces a dated markdown report plus the per-query ranks as JSONL, so every
number in the README is traceable to a committed artifact (`CLAUDE.md`
section 7).

**The index is the whole corpus; the queries are held out** (section 6). The
corpus is every catalog row with an image, including the rows the model
trained on. That is the harder and the correct setup: the distractors a real
lookup faces are exactly those rows, and indexing only the held-out split
would report a much easier task.

**Every baseline is recomputed here, from the same loaded corpus** (section 3).
None of them is a constant in this file.

⚠️ **BM25 on this task is a leakage detector, not a baseline.** The query IS
the target's own title, and the target's title is in the corpus, so BM25 is
asked "find the item whose title matches this title" — measured at a perfect
1.000 on a synthetic exact-match set. It is reported because omitting an
inconvenient baseline is worse than reporting it with its caveat, but it is
NOT a bar the model is failing to clear. The bar is frozen CLIP.
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

from vislens.index.baselines import BM25, class_prior_ranks, random_ranks  # noqa: E402
from vislens.index.metrics import paired_bootstrap, summarise  # noqa: E402
from vislens.index.retrieval import build_index, ranks_from_embeddings  # noqa: E402

CATALOG = pathlib.Path(os.getenv("VISLENS_CATALOG_DIR", ROOT / "data" / "catalog"))
SHARDS = pathlib.Path(os.getenv("VISLENS_SHARDS_DIR", ROOT / "data" / "shards"))
REPORTS = ROOT / "eval" / "reports"
HEADLINE = "recall@1"


def load_corpus(limit: int | None):
    """Every catalog row with a resolvable main image. No split filter."""
    import duckdb

    sql = f"""
        SELECT product_id, title, product_type, main_image_path, split
        FROM '{(CATALOG / "catalog.parquet").as_posix()}'
        WHERE main_image_path IS NOT NULL
        ORDER BY product_id, main_image_path
    """
    if limit:
        sql += f" LIMIT {limit}"
    return duckdb.sql(sql).df()


def embed_images(paths, encode_fn, batch: int = 64):
    """Embed the corpus images in ONE streaming pass, holding no pixel buffer.

    The obvious implementation decodes every image into a list and then embeds
    the list. At the real corpus size that list is 145,614 x 3 x 224 x 224
    float32 — **87 GB** — so it works perfectly on a `--limit 3000` smoke run
    and dies on the only run that matters. Decoded pixels are therefore
    accumulated one batch at a time and discarded as soon as they are embedded;
    what survives is 145,614 x 512 floats, about 300 MB.

    path -> LIST of slots, not one. Catalog rows share main images (467 of the
    first 3,000), and a plain dict keeps only the last slot for a repeated
    path, leaves the earlier ones unfilled, then reports them as absent from
    the archive — an error blaming the data for a bug in the lookup.
    """
    import tarfile

    from vislens.data.transforms import preprocess_bytes

    wanted: dict[str, list[int]] = {}
    for i, path in enumerate(paths):
        wanted.setdefault(path, []).append(i)

    out: np.ndarray | None = None
    filled = np.zeros(len(paths), dtype=bool)
    pending_pixels: list[np.ndarray] = []
    pending_slots: list[list[int]] = []
    started = time.time()
    done = 0

    def flush():
        nonlocal out, pending_pixels, pending_slots, done
        if not pending_pixels:
            return
        embedded = encode_fn(np.stack(pending_pixels))
        if out is None:
            out = np.zeros((len(paths), embedded.shape[1]), dtype=np.float32)
        for row, slots in zip(embedded, pending_slots, strict=True):
            for slot in slots:
                out[slot] = row
                filled[slot] = True
        done += len(pending_pixels)
        pending_pixels = []
        pending_slots = []
        if done % (batch * 40) == 0:
            rate = done / max(time.time() - started, 1e-9)
            remaining = (len(wanted) - done) / max(rate, 1e-9) / 60
            print(f"    {done:,}/{len(wanted):,} images ({rate:,.0f}/s, ~{remaining:.1f} min left)")

    source = ROOT / "data" / "abo" / "abo-images-small.tar"
    with tarfile.open(source, "r|") as tf:
        for member in tf:
            if not member.isfile() or not member.name.startswith("images/small/"):
                continue
            slots = wanted.get(member.name[len("images/small/") :])
            if not slots:
                continue
            handle = tf.extractfile(member)
            if handle is None:  # pragma: no cover - defensive
                continue
            pending_pixels.append(preprocess_bytes(handle.read()))
            pending_slots.append(slots)
            if len(pending_pixels) >= batch:
                flush()
    flush()

    if out is None or not filled.all():
        raise FileNotFoundError(f"{int((~filled).sum())} corpus images absent from {source.name}")
    return out


def build_encoders(checkpoint: str | None, device: str):
    """Returns (encode_images, encode_texts, label).

    With no checkpoint this is frozen zero-shot CLIP — which is both a
    baseline and the fallback, so the harness is runnable before any training
    has happened.
    """
    import torch

    from vislens.models.towers import load_clip

    clip_model, tokenizer = load_clip(device=device)

    def encode_images(arrays: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            feats = clip_model.encode_image(torch.from_numpy(arrays).to(device))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy()

    def encode_texts(texts: list[str]) -> np.ndarray:
        with torch.no_grad():
            feats = clip_model.encode_text(tokenizer(texts).to(device))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy()

    if checkpoint is None:
        return encode_images, encode_texts, "frozen CLIP ViT-B/32 (zero-shot)"

    raise NotImplementedError(
        "No trained checkpoint exists yet. Run without --checkpoint for the "
        "zero-shot baseline; the fine-tuned path lands with the first run."
    )


def markdown_report(rows, deltas, meta) -> str:
    lines = [
        f"# Retrieval eval — {meta['date']}",
        "",
        f"Task **text-to-image**. Corpus **{meta['corpus']:,}** catalog rows (the whole catalog — "
        f"the held-out thing is the queries). Queries **{meta['queries']:,}** "
        f"from `{meta['split']}`.",
        f"Model under test: **{meta['label']}**.",
        "",
    ]
    if meta.get("limit"):
        lines += [
            f"> ⚠️ **SMOKE RUN — `--limit {meta['limit']:,}`.** The real corpus is the whole",
            "> catalog (145,614 rows). A smaller corpus has fewer distractors, so every number",
            "> here is optimistic and none of them is comparable to a full run. This report",
            "> exists to show the harness works, not to report a result.",
            "",
        ]
    lines += [
        "Retrieval eval is **deterministic given a checkpoint** — repeating an identical run",
        "measures nothing. This is unlike ListingLens' judged agent benchmark and its ~37%",
        "run-to-run noise floor; do not assume that convention here. The variance that exists is",
        "query-sampling and training-seed, so deltas below use a **paired bootstrap over queries**",
        f"({meta['resamples']:,} resamples, 95% CI).",
        "",
        f"Analytic random floor at R@50 is `50/{meta['corpus']:,}` = "
        f"**{50 / meta['corpus']:.6f}** — about {50 / meta['corpus'] * meta['queries']:.0f} hits "
        f"across {meta['queries']:,} queries. The sampled `random` row below is that draw, and at "
        "single-digit hit counts it swings two- or three-fold on the seed alone. Read the analytic "
        "value, not the sampled one.",
        "",
        f"**{meta['duplicate_targets']:,} of {meta['corpus']:,} corpus rows share a main image "
        "with another row.** Ties count against the system being scored, so a target whose "
        "identical twin is indexed cannot rank 1st. That is a ceiling on R@1 for every row in "
        "this table, model and baselines alike.",
        "",
        "| system | R@1 | R@10 | R@50 | NDCG@10 | MRR | median rank |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in rows:
        lines.append(
            f"| {name} | {m['recall@1']:.4f} | {m['recall@10']:.4f} | {m['recall@50']:.4f} "
            f"| {m['ndcg@10']:.4f} | {m['mrr']:.4f} | {m['median_rank']:.0f} |"
        )
    lines += [
        "",
        f"## Deltas against each baseline ({HEADLINE})",
        "",
        "| baseline | model | baseline | delta [95% CI] | claimed |",
        "|---|---|---|---|---|",
    ]
    for d in deltas:
        lines.append(
            f"| {d.metric} | {d.model:.4f} | {d.baseline:.4f} | {d.format_ci()} "
            f"| {'yes' if d.claimed else '**no**'} |"
        )
    lines += [
        "",
        "A delta whose interval crosses zero is **not claimed** (`CLAUDE.md` section 4).",
        "",
        "## On the BM25 row",
        "",
        "BM25 here is a **leakage detector, not a baseline to beat**. The query is the target's",
        "own title and that title is in the corpus, so BM25 is asked to find the item whose title",
        "matches this title. It is reported because quietly dropping an inconvenient baseline is",
        "worse than reporting it with its caveat. The bar that decides whether fine-tuning was",
        "worth anything is the frozen-CLIP row.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--limit", type=int, default=None, help="cap the corpus, for a smoke run")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()

    import torch

    device = args.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    started = time.time()

    print("loading corpus ...")
    corpus = load_corpus(args.limit)
    print(f"  {len(corpus):,} rows")

    # Duplicate images in the index are a property of the catalogue, not a
    # bug, and they bound the metric: under pessimistic tie-breaking a target
    # whose identical twin is also indexed can never rank 1st. Measured and
    # published so a reader is not left wondering why R@1 has a ceiling.
    duplicate_targets = int(len(corpus) - corpus["main_image_path"].nunique())

    query_rows = corpus[corpus["split"] == args.split]
    if args.max_queries:
        query_rows = query_rows.head(args.max_queries)
    if query_rows.empty:
        sys.exit(f"no {args.split} rows in the loaded corpus — raise --limit")
    targets = np.asarray(query_rows.index, dtype=np.int64)
    queries = list(query_rows["title"].fillna(""))
    print(f"  {len(queries):,} held-out queries from {args.split}")

    encode_images, encode_texts, label = build_encoders(args.checkpoint, device)

    print(f"embedding the corpus on {device} ...")
    corpus_emb = embed_images(list(corpus["main_image_path"]), encode_images)
    index = build_index(corpus_emb)
    assert index.ntotal == len(corpus), "the index must cover the whole corpus"

    print("embedding queries ...")
    query_emb = np.concatenate(
        [encode_texts(queries[i : i + 256]) for i in range(0, len(queries), 256)], axis=0
    )

    print("scoring ...")
    ranks = {
        label: ranks_from_embeddings(index, query_emb, targets),
        "random": random_ranks(len(queries), len(corpus), seed=args.seed),
        "class-prior": class_prior_ranks(list(corpus["product_type"]), targets, seed=args.seed),
        "BM25 over titles (leakage detector)": BM25(list(corpus["title"].fillna(""))).ranks(
            queries, targets
        ),
    }

    rows = [(name, summarise(r)) for name, r in ranks.items()]
    from vislens.index.metrics import per_query_metrics

    model_metric = per_query_metrics(ranks[label])[HEADLINE]
    deltas = [
        paired_bootstrap(
            model_metric,
            per_query_metrics(r)[HEADLINE],
            metric=f"vs {name}",
            resamples=args.resamples,
            seed=args.seed,
        )
        for name, r in ranks.items()
        if name != label
    ]

    REPORTS.mkdir(parents=True, exist_ok=True)
    date = dt.date.today().isoformat()
    stem = REPORTS / f"{date}-retrieval-{args.split}"
    meta = {
        "date": date,
        "corpus": len(corpus),
        "queries": len(queries),
        "split": args.split,
        "label": label,
        "limit": args.limit,
        "resamples": args.resamples,
        "duplicate_targets": duplicate_targets,
        "seconds": round(time.time() - started, 1),
    }
    stem.with_suffix(".md").write_text(markdown_report(rows, deltas, meta))
    with stem.with_suffix(".jsonl").open("w") as f:
        for name, r in ranks.items():
            f.write(json.dumps({"system": name, "ranks": r.tolist()}) + "\n")

    print()
    print(markdown_report(rows, deltas, meta))
    print(f"wrote {stem}.md and {stem}.jsonl")


if __name__ == "__main__":
    main()
