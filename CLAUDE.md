# CLAUDE.md — conventions for `vislens`

Read this before writing code here. These rules were carried over from ListingLens, where they
live scattered across `README.md` and an 88 KB `HANDOFF.md`. This file is the version that does
not drift.

---

## 1. What this repo is

Two halves that share one encoder:

- **The audit service** (`src/vislens/rules/`, `src/vislens/fetch/`, `service/`) — product-image
  compliance checks and near-duplicate detection over **user-supplied images**. This is the
  product. ListingLens calls it over HTTP and holds only a thin client.
- **The benchmark** (`src/vislens/{data,models,train,onnx,index}/`, `eval/`) — whether in-domain
  fine-tuning beats zero-shot CLIP at product retrieval, and at what data scale.

**The benchmark half is a benchmark, not a service.** There is no deployment for it, because there
is no consumer and a p99 without a workload would be a number we made up. If a number has no real
workload behind it, it does not get published.

**All vision code lives here.** ListingLens deliberately bans torch (`requirements.txt:36-53`:
torch + sentence-transformers cost ~490 MiB RSS before reading a weight and were OOM-killed under
a 512 MB cap). Nothing in this repo may assume it can add a dependency to ListingLens.

---

## 2. torch is training-only

`torch` appears **only** in the `train` optional-dependency group. Anything that serves a request
— the audit service, the ONNX exports, the parity tests — uses `onnxruntime` and nothing else.

This is not stylistic. ListingLens peaks at 469.8 / 512 MiB today, and the failure signal for
exceeding it is a silent `exit 137` with no traceback. `src/onnx_embeddings.py` in that repo is the
template: hand-written ONNX inference, pinned model revision, `intra_op_num_threads` from env, and
a parity test asserting cosine ≥ 0.9999 against the torch reference.

---

## 3. Baselines are mandatory, and computed at runtime

Every metric is published **next to the baseline it has to beat**, and the baselines are computed
from the loaded gold set and the loaded corpus **inside the same process** that computes the
model's numbers. Never hardcode a floor.

The reason, from `listinglens/eval/run_eval.py:262-270`: the gold set keeps changing, so a
hardcoded baseline "would quietly start lying," and a `--limit` run would report the full set's
floor against a subset's accuracy.

Retrieval baselines, per task, with N/A marked honestly rather than omitted:

| Baseline | text2image | image2image |
|---|---|---|
| Random | yes | yes |
| Class-prior (corpus ranked by descending global `product_type` frequency, fixed-seed ties) | yes | yes |
| BM25 over titles | yes | **N/A** — no text query |
| **Frozen CLIP ViT-B/32 zero-shot** | yes | yes |

The frozen-CLIP row is the one that decides whether the fine-tune was worth anything. It is not
optional and it is not a formality.

Compliance-check baselines: always-pass, **plus a naive competitor per check** — the
four-corner-pixel test for white background, the silhouette-area ratio for frame occupancy. The
delta over the naive version is the entire justification for the real implementation.

---

## 4. Deltas need confidence intervals

Retrieval eval is **deterministic** given a checkpoint, so repeated identical runs measure
nothing. This differs from ListingLens' judged agent benchmark, which has an empirically measured
~11-row (~37%) run-to-run noise floor. **Say so in the report** — the mechanism is different and a
reader who knows the ListingLens convention will otherwise assume repeated runs.

The variance that actually exists here is query-sampling and training-seed. So:

- **Paired bootstrap over queries**, 10,000 resamples, 95% CI, on the per-query difference between
  model and baseline. Paired is much tighter than two independent CIs and is the correct test for
  "did we beat CLIP."
- **A delta whose CI crosses zero is not claimed.** Print the CI in every table, always.
- **Three seeds** on any headline config. A delta smaller than the seed spread is not a result.

---

## 5. Split by `product_id`, never by image

A product with five images must have all five in one split, or every retrieval number is inflated.

Splits are **hash-based, not a stored shuffle**:

```python
split = (["train"] * 8 + ["val"] + ["test"])[
    int(hashlib.sha1(product_id.encode()).hexdigest(), 16) % 10
]
```

Order-independent, reproducible from a clean clone with no split file to lose, and stable when
rows are added later.

Three tests enforce it, and **a test ships with every data-transform commit**:
1. Zero `product_id` overlap across splits.
2. Every `image_id` maps to exactly one split.
3. **Every packed shard is split-pure** — which makes leakage structurally impossible at the
   shard level rather than merely tested for.

---

## 6. The index is the full catalog. The queries are held out.

`build_index(catalog_df)` takes **no split argument**, has a docstring saying why, and a test
asserting `index.ntotal == len(catalog)`.

The held-out thing in retrieval eval is the *queries*, not the corpus. Reviewers assume the wrong
thing by default, so this has to be explicit in the code and not only in the README.

---

## 7. Honest reporting

- **Disproven prior claims get struck through, not quietly edited.** The model to copy is
  ListingLens' README: *"⚠️ The scored number rose from 60% to 69.2%, and that is not the agent
  getting better."*
- Confounds get a `⚠️` marker and a sentence naming the confound.
- **Every number in the README is traceable** to a committed `src/vislens/runs/<id>/metrics.csv`
  or an `eval/reports/` file. No number lives only in prose.
- Dated reports: `eval/reports/YYYY-MM-DD-<tag>.md` (human summary) + `.jsonl` (per-query raw).
- **If the fine-tune loses to frozen CLIP, the loss is the headline.** That is why the data-scaling
  sweep (5/25/50/100% of pairs) is mandatory and pre-committed: it turns a loss into a curve with a
  slope and an extrapolated crossover, which is a finding. "We fine-tuned and it was worse" is not.

---

## 8. No committed data or weights

Download scripts plus `.gitignore`. Model artifacts ship as GitHub Release assets.

The one exception, and it is deliberate: `src/vislens/rules/rules_v1.json` and its sibling
`thresholds_v1.json` **are** committed, and they live inside the package rather than under
`data/` so a wheel carries them (2026-09-21: a `pip install .` without them imported fine and
then raised `FileNotFoundError` on the first request). Rule
thresholds are configuration with a provenance record (`verified_against`, `verified_on`), not
derived data, and the eval report has to be able to name the exact thresholds it measured. A
threshold change is a dated, reviewable diff.

---

## 9. Preprocessing has exactly one definition

`src/vislens/data/transforms.py`, imported by training, eval, ONNX export, and the parity
fixtures, and serialized into `preprocess.json` for consumers. **A second definition anywhere is
the bug** — it silently degrades the product instead of failing.

Interpolation is pinned to PIL `BICUBIC`. PIL, cv2, and torchvision-without-antialias produce
different pixels for the same resize.

Mean/std normalization and L2-normalization go **inside the ONNX graph**, so a consumer owns only
resize + crop + scale and cannot forget to normalize. A test asserts `‖output‖ = 1`.

---

## 10. Notebooks are launchers, never logic

A Kaggle notebook is ~15 lines: `pip install -q git+https://github.com/<you>/vislens@<sha>`, then
`from vislens.train import main; main(cfg, resume_from=...)`. **Pin the SHA** — that is what makes
a run reproducible. Notebooks live in `notebooks/` and are version-controlled, so the exact cell
that produced a number is in the repo.

---

## 11. Compute reality

- **Select `NvidiaTeslaT4` explicitly.** Kaggle's P100 no longer runs Kaggle's own PyTorch — the
  cu128 build's arch list excludes `sm_60`, and the PR to restore it was rejected because Google
  Cloud is deprecating P100s. On a P100 you get `no kernel image is available for execution`.
- T4 is `sm_75`: **fp16 + `GradScaler`, not bf16** (bf16 needs `sm_80`+). Keep `logit_scale` in log
  space, clamp to `log(100)`, and cast logits to fp32 before cross-entropy — otherwise the
  contrastive loss NaNs several hundred steps into a headless run.
- **9 hours interactive, 12 hours for a "Save & Run All" commit.** Ride the commit path; it is
  headless and survives closing the browser. `--max-minutes` exits cleanly at ~11h20m so a run
  terminates on its own terms instead of being killed.
- **≤500 notebook output files.** Never write per-step artifacts; keep exactly two checkpoints.
  Image data is packed into tar shards, not loose files.
- Checkpoint writes are **atomic** (`.tmp` then `os.replace`). A commit killed mid-write must not
  leave a corrupt checkpoint.
- **No Weights & Biases**, and that is a feature: run records committed as CSV/JSON put the numbers
  where a reviewer reads them, instead of behind a login.
- **`--subset 500` runs the whole pipeline locally on an M4 in under 5 minutes**, and CI runs it.
  Compute is not the binding constraint (~3-4 GPU-hr/week against a 30 GPU-hr/week quota); the
  Kaggle edit→commit→wait→read-logs loop is. No code reaches Kaggle un-smoke-tested.

---

## 12. No Spark

A Spark embedding job has one GPU; `local[*]` executors contending for one T4 is slower *and* less
defensible than a single torch `DataLoader`.

And the catalog join is 83 MB of gzipped JSON, which **DuckDB does in one query**. Standing up
Spark to join 83 MB invites exactly the follow-up question you do not want.

The Spark story lives in the sibling trading repo and is described accurately there as local-mode
`applyInPandas`, not a cluster.

---

## 13. Claims this repo does not make

- Not "147K training pairs." The `en_US` filter takes ABO to roughly 26K English-titled products.
  `catalog.parquet` (language-agnostic, ~140K) and `pairs.parquet` (English, ~26K) are separate
  tables and **both counts are published**, so neither masquerades as the other.
- Not "PySpark batch embedding over 147K products." See §12.
- Not a personalization system. The build spec claimed this closed a "personalization gap"; nothing
  here personalizes anything — there is no user model and no history.
- Not a text/logo *classifier*. `background_artifacts` reports any non-white component on the
  background without claiming to know what it is, which is a definitional check rather than a
  guess. Off-product text detection ships only once its measured precision clears the gate in
  `rules_v1.json`.

---

## 14. Milestone discipline

One milestone per session. Ask for a plan before code, then approve or correct it — cheaper than
reviewing 800 lines after the fact.
