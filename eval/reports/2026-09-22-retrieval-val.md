# Retrieval eval — 2026-09-22

Task **text-to-image**. Corpus **3,000** catalog rows (the whole catalog — the held-out thing is the queries). Queries **150** from `val`.
Model under test: **frozen CLIP ViT-B/32 (zero-shot)**.

> ⚠️ **SMOKE RUN — `--limit 3,000`.** The real corpus is the whole
> catalog (145,614 rows). A smaller corpus has fewer distractors, so every number
> here is optimistic and none of them is comparable to a full run. This report
> exists to show the harness works, not to report a result.

Retrieval eval is **deterministic given a checkpoint** — repeating an identical run
measures nothing. This is unlike ListingLens' judged agent benchmark and its ~37%
run-to-run noise floor; do not assume that convention here. The variance that exists is
query-sampling and training-seed, so deltas below use a **paired bootstrap over queries**
(2,000 resamples, 95% CI).

**467 of 3,000 corpus rows share a main image with another row.** Ties count against the system being scored, so a target whose identical twin is indexed cannot rank 1st. That is a ceiling on R@1 for every row in this table, model and baselines alike.

| system | R@1 | R@10 | R@50 | NDCG@10 | MRR | median rank |
|---|---|---|---|---|---|---|
| frozen CLIP ViT-B/32 (zero-shot) | 0.0667 | 0.2667 | 0.4733 | 0.1524 | 0.1300 | 64 |
| random | 0.0000 | 0.0067 | 0.0267 | 0.0020 | 0.0028 | 1628 |
| class-prior | 0.0000 | 0.0000 | 0.0067 | 0.0000 | 0.0014 | 1814 |
| BM25 over titles (leakage detector) | 0.8200 | 0.8267 | 0.8267 | 0.8242 | 0.8234 | 1 |

## Deltas against each baseline (recall@1)

| baseline | model | baseline | delta [95% CI] | claimed |
|---|---|---|---|---|
| vs random | 0.0667 | 0.0000 | +0.0667 [+0.0333, +0.1067] | yes |
| vs class-prior | 0.0667 | 0.0000 | +0.0667 [+0.0333, +0.1067] | yes |
| vs BM25 over titles (leakage detector) | 0.0667 | 0.8200 | -0.7533 [-0.8200, -0.6800] | yes |

A delta whose interval crosses zero is **not claimed** (`CLAUDE.md` section 4).

## On the BM25 row

BM25 here is a **leakage detector, not a baseline to beat**. The query is the target's
own title and that title is in the corpus, so BM25 is asked to find the item whose title
matches this title. It is reported because quietly dropping an inconvenient baseline is
worse than reporting it with its caveat. The bar that decides whether fine-tuning was
worth anything is the frozen-CLIP row.
