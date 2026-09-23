# Retrieval eval — 2026-09-22

Task **text-to-image**. Corpus **145,614** catalog rows (the whole catalog — the held-out thing is the queries). Queries **14,072** from `val`.
Model under test: **frozen CLIP ViT-B/32 (zero-shot)**.

Retrieval eval is **deterministic given a checkpoint** — repeating an identical run
measures nothing. This is unlike ListingLens' judged agent benchmark and its ~37%
run-to-run noise floor; do not assume that convention here. The variance that exists is
query-sampling and training-seed, so deltas below use a **paired bootstrap over queries**
(10,000 resamples, 95% CI).

**22,110 of 145,614 corpus rows share a main image with another row.** Ties count against the system being scored, so a target whose identical twin is indexed cannot rank 1st. That is a ceiling on R@1 for every row in this table, model and baselines alike.

| system | R@1 | R@10 | R@50 | NDCG@10 | MRR | median rank |
|---|---|---|---|---|---|---|
| frozen CLIP ViT-B/32 (zero-shot) | 0.0194 | 0.0807 | 0.1510 | 0.0473 | 0.0417 | 1980 |
| random | 0.0000 | 0.0001 | 0.0009 | 0.0001 | 0.0001 | 73130 |
| class-prior | 0.0000 | 0.0001 | 0.0003 | 0.0000 | 0.0001 | 70156 |
| BM25 over titles (leakage detector) | 0.7555 | 0.8217 | 0.8313 | 0.7929 | 0.7837 | 1 |

## Deltas against each baseline (recall@1)

| baseline | model | baseline | delta [95% CI] | claimed |
|---|---|---|---|---|
| vs random | 0.0194 | 0.0000 | +0.0194 [+0.0171, +0.0217] | yes |
| vs class-prior | 0.0194 | 0.0000 | +0.0194 [+0.0171, +0.0217] | yes |
| vs BM25 over titles (leakage detector) | 0.0194 | 0.7555 | -0.7361 [-0.7432, -0.7286] | yes |

A delta whose interval crosses zero is **not claimed** (`CLAUDE.md` section 4).

## On the BM25 row

BM25 here is a **leakage detector, not a baseline to beat**. The query is the target's
own title and that title is in the corpus, so BM25 is asked to find the item whose title
matches this title. It is reported because quietly dropping an inconvenient baseline is
worse than reporting it with its caveat. The bar that decides whether fine-tuning was
worth anything is the frozen-CLIP row.
