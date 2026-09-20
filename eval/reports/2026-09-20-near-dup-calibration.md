# Near-duplicate calibration — 2026-09-20

- **Sources:** 23 real product images
- **Pairs:** 965 (184 positive, 561 hard negative, 220 easy negative)
- **Dev ASINs:** B00N2ZDXW2, B07PXGQC1Q, B08XPWDSWW
- **Held-out ASINs:** B01K8B8YA8, B07GZFM1ZM, B08RLW7918
- **Selection rule:** highest dev F1 among thresholds with dev precision ≥ 0.95

A *positive* is a mutation of the same source image. A *hard negative* is a different photo of the same product — not a duplicate, and the discriminating case, because every image in one listing is of one product. `random` and `pixel_mse` are the floors.

## Ranking discrimination (ROC-AUC)

| Method | AUC (all pairs) | AUC (held-out) | **AUC vs hard negatives only** |
|---|---|---|---|
| `phash` | 0.9938 | 0.9988 | **0.9932** |
| `dhash` | 1.0000 | 1.0000 | **1.0000** |
| `tiles` | 0.9769 | 0.9835 | **0.9754** |
| `pixel_mse` | 0.9919 | 1.0000 | **0.9894** |
| `random` | 0.5199 | 0.5709 | **0.4950** |

## At the selected threshold

| Method | Threshold | Dev P / R / F1 | Held-out P / R / F1 | Held-out P vs hard negs |
|---|---|---|---|---|
| `phash` | 10 | 1.000 / 0.927 / 0.962 | 1.000 / 0.943 / 0.971 | 1.000 |
| `dhash` | 8 | 1.000 / 1.000 / 1.000 | 1.000 / 0.989 / 0.994 | 1.000 |
| `tiles` | 4 | 1.000 / 0.927 / 0.962 | 1.000 / 0.875 / 0.933 | 1.000 |
| `pixel_mse` | 143.522 | 1.000 / 0.833 / 0.909 | 1.000 / 0.784 / 0.879 | 1.000 |
| `random` | — | — | — | no threshold reaches dev precision >= 0.95 |

## Honest reading

**The set is small.** 23 source images across 6 ASINs, with 3 ASINs held out. An AUC of 1.0000 here means *no errors on a small held-out set*, not *solved* — the interval around a zero-false-positive observation at this sample size is wide, and nothing below should be read as a ceiling claim.

**The positives are scripted, so they are systematically easier than reality.** A pasted badge with crisp edges over a flat background is a kinder test than a real one antialiased over a gradient. Treat the per-mutation recalls as an ordering of difficulty, not as field performance.

**`tiles` carries an uncalibrated sub-parameter.** Its `per_tile_max_bits` was set to 10 by hand and never measured, so its ranking is partly an artefact of that guess. Its *crop* weakness is structural rather than a tuning problem — a crop shifts content across cell boundaries, so cell *i* stops corresponding to cell *i* — which is the caveat written into its docstring before this was measured.

**`pixel_mse` is a stronger floor than it looks.** A 32x32 grayscale mean squared error comes close to the perceptual hashes on this set, which limits how much credit the hashing itself can take.

**The methods fail in complementary ways, and that is the interesting result.** `phash` is robust to cropping and weak on a pasted badge; `tiles` is the exact inverse; `dhash` handles both, which is why it wins rather than because it is more sophisticated. It compares gradient *signs* on a coarse thumbnail, so a localized paste flips few bits and a small crop barely disturbs the coarse structure.

**Consequence for the model tier.** The build plan pre-committed to shipping hashing alone if it matched a CLIP embedding, precisely to avoid hosting a model. `dhash` leaves no headroom for an embedding to demonstrate on *this* task at this set size, so a near-duplicate encoder is not justified. If the encoder earns its place later it will have to be on a different job — style similarity between *different* photos — not on this one.

## `phash` recall by modification

| Mutation | Recall | Median distance |
|---|---|---|
| `badge` | 11/23 (48%) | 12.0 |
| `brighten` | 23/23 (100%) | 0.0 |
| `crop_95` | 23/23 (100%) | 6.0 |
| `jpeg_q30` | 23/23 (100%) | 0.0 |
| `jpeg_q60` | 23/23 (100%) | 0.0 |
| `resize_50` | 23/23 (100%) | 0.0 |
| `resize_75` | 23/23 (100%) | 0.0 |
| `watermark` | 23/23 (100%) | 0.0 |

## `dhash` recall by modification

| Mutation | Recall | Median distance |
|---|---|---|
| `badge` | 23/23 (100%) | 2.0 |
| `brighten` | 23/23 (100%) | 2.0 |
| `crop_95` | 22/23 (96%) | 4.0 |
| `jpeg_q30` | 23/23 (100%) | 1.0 |
| `jpeg_q60` | 23/23 (100%) | 0.0 |
| `resize_50` | 23/23 (100%) | 0.0 |
| `resize_75` | 23/23 (100%) | 0.0 |
| `watermark` | 23/23 (100%) | 0.0 |

## `tiles` recall by modification

| Mutation | Recall | Median distance |
|---|---|---|
| `badge` | 23/23 (100%) | 1.0 |
| `brighten` | 23/23 (100%) | 0.0 |
| `crop_95` | 5/23 (22%) | 8.0 |
| `jpeg_q30` | 23/23 (100%) | 0.0 |
| `jpeg_q60` | 23/23 (100%) | 0.0 |
| `resize_50` | 23/23 (100%) | 0.0 |
| `resize_75` | 23/23 (100%) | 0.0 |
| `watermark` | 23/23 (100%) | 0.0 |
