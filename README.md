# vislens

**Two things, sharing one image encoder:**

1. **An image-audit service** — checks product images against Amazon's published main-image
   requirements, and finds near-duplicates across a set. Runs on user-supplied images. This is
   the half that [ListingLens](https://github.com/Het415/listinglens) calls over HTTP.
2. **A benchmark** — *does in-domain fine-tuning beat zero-shot CLIP at product retrieval, and at
   what data scale?* Not a service. There is no deployment for it, because there is no consumer,
   and a p99 with no workload behind it would be a number I made up.

Conventions, and the reasoning behind each: [CLAUDE.md](CLAUDE.md).

---

## Status

**Early.** What exists is honest about what it is:

| Component | State |
|---|---|
| Compliance checks (`src/vislens/rules/image_rules.py`) | **built** — 6 deterministic checks |
| Rule thresholds (`src/vislens/rules/rules_v1.json`) | **built** — committed with provenance, shipped as package data |
| Agent wire payload | **built** — 7444 → 834 chars on a real 6-image listing |
| Hardened image fetcher (`src/vislens/fetch/`) | **built** — 6 SSRF controls, 72 tests |
| Audit service (`src/vislens/service/app.py`) | **built** — verified end to end on a live listing |
| Semantic checks (text overlay, image role) | **specified, disabled** — local ONNX, gated on measured agreement |
| Near-duplicate detection | **built** — calibrated, `dhash` @ ≤8, held-out P 1.000 / R 0.989 |
| SKU group-mismatch finding | **built** — gated at the calibrated threshold |
| ABO catalog build | **built** — 145,614 products, 535,602 images, two leaks closed |
| WebDataset shard packing | **built** — 405,116 samples in 42 shards, ~30s, reproducible |
| Encoder training | **written, unrun** — two-tower + masked InfoNCE, smoke-tested on CPU against the real shards; needs a GPU |
| Retrieval eval harness | not started |
| ListingLens client + agent wiring | **built** — in the sibling repo, 21 tests |
| ABO `en_US` gate | **measured** — see below |
| Split-leakage fix | **measured** — 27,554 leaking images → 0, then 5 more the packer caught |
| Split reproducibility | **measured** — two builds, one byte-identical assignment |
| CI | **built** — ruff + 275 tests across three jobs, no secrets, no network, no GPU |

Nothing above is claimed in a UI before it is backed by code. That rule exists because the
predecessor repo shipped a landing page advertising *"CLIP model analyzes your product images for
lighting, composition, and presentation quality"* against no CLIP model, no image scoring, and no
vision path of any kind.

---

## The compliance checks

Every check carries a **tier**, and the tier determines how it may be rendered. Only the first two
can drive a headline verdict:

| Tier | Meaning | Drives the headline? |
|---|---|---|
| `rule_exact` | Arithmetic on image metadata. Cannot be wrong | yes |
| `measured` | A pixel statistic, with published precision/recall against a labeled set | yes |
| `advisory` | A signal that is explicitly **not** a verdict | no |
| `model` | A VLM judgement, shipped with its measured agreement against hand labels | no |

Six deterministic checks are built:

| Check | Tier | What it measures |
|---|---|---|
| `resolution_and_format` | `rule_exact` | Longest side against the 1000px zoom threshold and 500px floor; format; colour mode |
| `image_count` | `rule_exact` | How many images were **supplied** — never a claim about the live listing |
| `white_background` | `measured` | Border-band purity, as two statistics plus the modal RGB |
| `frame_occupancy` | `measured` | Product **bounding box** against the 85% rule, with silhouette area reported beside it |
| `background_artifacts` | `measured` | Non-product marks on the background — badges, watermarks, borders, props |
| `aspect_ratio` | `advisory` | Distance from square. Never pass/fail |

### Three decisions worth explaining

**Occupancy is measured on the bounding box, not the silhouette.** Amazon's "the product must fill
85% of the frame" is conventionally the product's bbox. A pair of earbuds fills roughly 40% of its
own bounding box, so a silhouette-area reading fails nearly every *compliant* listing — the check
would be wrong more often than right. Both numbers are reported so a reader can check the
interpretation instead of trusting it, and the silhouette reading is published as a **baseline** in
the eval rather than merely dismissed here.

**`background_artifacts` replaces text/logo detection, and is more honest than it.** The rule is
that the background must be pure white, so *any* connected non-white component on the background
that is not the product violates it — whatever it is. The check never has to classify the mark,
which is why it needs no model and why its precision is high.

**Off-product text is deliberately not shipped yet, and will be a model check rather than a
heuristic.** The violation Amazon cares about is text the seller *overlaid*; text *printed on the
product* is allowed, and the two are visually identical — the distinction is semantic and spatial.
No stroke-width or edge-density heuristic can separate them, so one shipped as a verdict would be
wrong more often than right. It ships only once a VLM's measured agreement against hand labels
clears the gate in `rules_v1.json`, and that gate lives in the rules file so it cannot drift from
the eval report.

---

## Why the checks are rules, and where a model comes in

The split is **definitional vs semantic**.

"Is the background pure white" is *defined in pixel values*, so it gets computed. A model would
approximate a definition that is exactly calculable, and asking one whether RGB equals 255 is
strictly worse than checking. Three properties follow from computing it, and they are the point:

- **Determinism.** 54 tests in under 3 seconds, no API key, no cost, in CI on every push. The
  predecessor's agent benchmark has a measured ~37% run-to-run noise floor and a budget of roughly
  one judged run per day — a model in the verdict path would import that problem into the one part
  of this feature currently free of it.
- **Citability.** "0.16 against an 0.85 rule" is actionable. "The product looks small in the frame"
  is an opinion.
- **No shared rate limit.** Nothing in the deterministic path touches an LLM quota.

Some checks are genuinely semantic, and those are **`model`-tier**: reported and labelled, shipped
with their measured agreement against hand labels, and **never able to produce or override a
computed verdict** — the same guard that keeps `aspect_ratio` from turning a compliant listing red.

**They run as local pinned ONNX, not a hosted model.** Determinism is the property that motivated
computing everything else, and a hosted model would forfeit exactly that — plus it would cost per
call and consume an LLM rate limit. (Groq is not even an option: checked 2026-09-20, its live
catalog is 13 models and none accept images. `llama-3.2-11b-vision` and Llama 4 Scout/Maverick were
removed in the same family-wide deprecation that broke the predecessor three times.)

And the useful discovery is that **neither check that matters actually needs a VLM**:

| Check | How it runs locally | Size |
|---|---|---|
| `off_product_text` | Text **detection** only (`PP-OCRv6_medium_det_onnx`, apache-2.0) → then the overlaid-vs-printed question reduces to **geometry** against the background mask this module already computes: centroid on the background = overlaid violation, centroid on the product = printed on the product and allowed | 62 MB |
| `image_role` | CLIP zero-shot against fixed role prompts, with the prompt embeddings **precomputed offline** and shipped as a small `.npy` — so the runtime needs only the vision tower already required for near-duplicate detection | 0 extra |
| `prohibited_content` | **Narrow scope only** (person present, multiple distinct objects). "Props not included" needs to know what the purchase includes, and competitor-logo detection needs a logo dataset. The full check is not claimed | — |

The known gap is honest and will be measured separately rather than papered over: text overlaid
*on top of* the product region reads as printed-on-product under the geometric rule.

---

## Scope: three rules govern the main image only

Amazon's pure-white background, 85% frame occupancy, and no-marks-on-the-background requirements
apply to the **main image**. Secondary images are explicitly permitted lifestyle backgrounds,
props, in-use scenes, text and graphics.

Applying those three across a whole set would fail a seller's perfectly compliant lifestyle photo
against a rule that does not govern it — wrong more often than right on any real listing. Measured
on a live listing (`B08XPWDSWW`): of six product images, the main image failed occupancy at 45.8%
(a real finding), while five secondary images produced identical-looking numbers that mean nothing.

So the payload separates **verdicts** (`f`) from **measurements** (`a`) structurally, rather than
with a tier field a consumer has to remember to check. The same check appears in `f` for the main
image and in `a` for a secondary one; a per-code tier field would have mislabelled one of them, and
did, until a live run caught it.

---

## Near-duplicate detection

Sellers repost the same photo resized, recompressed, or with a promo badge added, and Amazon
expects each image in a listing to show something different. Three hashes were implemented and the
*measurement* picked which to ship, rather than the conventional wisdom:

| Method | ROC-AUC vs hard negatives | `badge` recall | `crop_95` recall |
|---|---|---|---|
| **`dhash`** ← shipped | **1.0000** | 100% | 96% |
| `phash` | 0.9932 | **48%** | 100% |
| tiled `dhash` | 0.9754 | 100% | **22%** |
| `pixel_mse` (baseline) | 0.9894 | — | — |
| `random` (floor) | 0.4950 | — | — |

Full report: [`eval/reports/2026-09-20-near-dup-calibration.md`](eval/reports/2026-09-20-near-dup-calibration.md).
Measured on 23 real product images across 6 ASINs, 965 labeled pairs, split by ASIN with 3 held
out. Shipped at `dhash` ≤ 8: held-out precision 1.000, recall 0.989.

**pHash losing was the surprise**, and it is explainable: a 20%-wide promo badge is a large
high-contrast paste that moves the low-frequency DCT coefficients pHash is built on, while dHash
compares gradient *signs* on a 9×8 thumbnail, where the badge flips two bits. The tiled hash is
the mirror image — it catches badges perfectly and fails crops, exactly the caveat written into its
docstring before any of this was measured.

Two label definitions matter more than the numbers:

- A **positive** is a mutation of the *same image*. The build plan proposed same-product-different-shot
  pairs as positives; that is the wrong question. A second genuine photo of the same product is not
  a duplicate, it is an image the seller *should* have.
- That case is therefore the **hard negative**, and it is the one that counts — every image in one
  listing is of one product, so a method that only separates *different products* is useless here.
  Hard and easy negatives are reported separately for that reason.

### Two guards, both added after they fired on real data

**Featureless images are excluded, not compared.** Every hash here encodes contrast structure, so
an image with none hashes to all-zero: a flat white square and a flat grey square both come out as
`0x0000000000000000` under dHash *and* pHash, putting two visibly different images at distance 0.
Images below a minimum detail threshold are reported as skipped.

**The group-mismatch finding is gated at the calibrated duplicate threshold.** Without the gate it
reported a mismatch whenever an image's nearest neighbour sat in another group *at any distance*.
On seven real photos every pairwise distance fell between 19 and 35 bits, so the nearest-group
ordering was arbitrary and **5 of 7 images were flagged**. Gated, the same input yields 2 findings,
both genuine. The claim is now precise: *this image is a near-duplicate of an image you tagged to a
different SKU.*

### No index, deliberately

At 3-12 user-supplied images, exact all-pairs comparison is faster and less code than building one.
The recall/latency tradeoff table that justifies an approximate index belongs in the benchmark half,
where the corpus is six figures.

### What this means for the model tier

The build plan pre-committed to shipping hashing alone if it matched a CLIP embedding, precisely to
avoid hosting a model. `dhash` leaves no headroom for an embedding to demonstrate on *this* task at
this set size, so **a near-duplicate encoder is not justified**. If the encoder earns its place it
will be on a different job — style similarity between genuinely *different* photos — not this one.

---

## The agent payload took three iterations, each one measured

The hardest problem in this repo was not computing a verdict. It was handing a
verdict to a language model without it inventing a stronger one. Each attempt was
tested against a live agent run rather than assumed:

**1. One list of findings with a per-check `tier` field.** Broken on arrival: the
same check is a *verdict* on the main image and a *measurement* on a secondary one,
so a single per-code tier mislabelled one of them. A real listing's secondary-image
occupancy number was being presented as `tier: measured` — i.e. as a violation.

**2. Two lists, `f` (verdicts) and `a` (measurements) — but both carrying
`"fail"` / `"warn"` statuses.** The agent reported **every** `a` entry as a
violation anyway, despite an explicit prompt rule forbidding exactly that. It wrote
"the background isn't pure white, there are background marks" about rules that had
never been evaluated, and invented a unit ("14.44° deviation" for a 14.44:1 ratio).
The word `"fail"` in the structure beat the instruction in the prompt.

**3. `a` statuses neutralised to `"measured"`.** Better — the evidence field started
saying `measured` and the model quoted the caveat — but the prose still asserted
"the background isn't pure white" and called them "key rule failures".

**What finally worked was deleting the data.** The agent payload now carries only
findings the agent may legitimately assert. Advisory measurements are *counted*
(`n_measured_only`) and never enumerated; their values live in the detail record,
which the UI renders rather than reasons about. With that change the same query
returned `NEEDS_MORE_DATA` and said the main image could not be identified, with
zero claims about rules that were never evaluated.

The generalisable lesson: **a prompt instruction is weaker than a data structure.**
If a consumer must not draw a conclusion from a number, the fix is to not send the
number — not to ask it nicely.

---

## Amazon Berkeley Objects: the `en_US` gate, measured

The build plan mandated filtering titles to `en_US`. Run against the real 83 MB
listings archive before committing to the dataset:

| | |
|---|---|
| Total listings | **147,702** |
| With `main_image_id` | 147,127 (99.6%) |
| With `product_type` | 147,702 (100%) |
| **`en_US` `item_name`** | **26,424 (17.9%)** |
| Any `en_*` | 122,734 (83.1%) |

Top tags: `en_IN` 76,443 · `en_US` 26,424 · `de_DE` 15,097 · `es_US` 12,012 ·
`zh_CN` 11,701. The dominant country is India at 76,442 listings.

### The split-by-product rule is not sufficient on this dataset

This is the most important finding of the build, and it would have silently
inflated every retrieval number.

The build plan mandates splitting by `product_id`, never by image. Done exactly
that way, the result still leaked:

| | |
|---|---|
| Unique images | 397,240 |
| Images used by **more than one product** | 61,665 |
| **Images appearing in more than one split** | **27,554** |
| Rows touching a cross-split image | **285,692 (40.3%)** |

ABO lists the same product across marketplaces sharing byte-identical assets, so
product A in train and product B in test can hold the same pixels. A
product-level split passes its own check while the images cross freely — and the
build's original leak assertion tested exactly that insufficient invariant.

**The fix is to split connected components of the product-image graph**, so two
products sharing any image land in the same split. Un-pruned, that graph has a
giant component of 53,140 products (36.6%) driven by 23 boilerplate images —
three attached to ~33,300 products each — forcing an 89/5/5 split. Dropping
images attached to more than 50 products removes **70 images (0.018%)**,
collapses the largest component to 1.9%, and restores 80/10/10. A sweep showed
K of 25, 50 and 200 all land within 0.3 points, so this is not a knife-edge
tuning artefact.

Independently of leakage, an image belonging to 33,313 products cannot be a
retrieval target: "which product is this?" has no single answer for it.

**Result: images spanning more than one split went from 27,554 to 0**, asserted
by the build itself (which aborts) and by the test suite.

### Three consequences of the language filter

- **The `en_US`-only training set is ~26K, not 147K** — smaller than the fallback
  dataset the plan proposed. So "147K training pairs" is a claim this repo will not
  make.
- **The `en_*` ladder is not a marginal fallback, it 4.6×s the training set**
  (26,424 → 122,734). That makes the mandated data-scaling sweep genuinely
  informative across a real range, and `title_lang` is recorded per row so the
  choice can be ablated rather than assumed.
- **`product_type` is present on 100% of rows with no language tag**, so the
  classification task and the index corpus can use the full ~147K catalogue. Hence
  two tables, both counts published: a language-agnostic image corpus and an
  English pairs table.

---

## Packing: 398K loose files into 42 shards

Kaggle allows a notebook ~500 output files and the archive holds 398,212 images,
so the images go into WebDataset tar shards and training streams them.
`scripts/pack_shards.py` computes the whole plan from the catalog first, then
makes **one sequential pass** over `abo-images-small.tar` — the 3.0 GB source is
never extracted, because extracting a tar in order to repack it writes 398K
files to change nothing but how they are grouped. The full pack takes ~30s.

Bytes are copied, never re-encoded: no resize, no recompression. That is
`CLAUDE.md` §9 — preprocessing gets exactly one definition, and it is reserved
for `vislens.data.transforms`, which training, eval and the ONNX export will all
import. A packer that resized to 224 would be a second one, and the two would
disagree silently rather than fail.

Two roles, and an image is in exactly one of them:

| series | samples | shards | size | what it is |
|---|---|---|---|---|
| `pairs-train` | 97,287 | 10 | 942 MB | the contrastive training set: image, title, metadata |
| `pairs-val` | 11,747 | 2 | 113 MB | |
| `pairs-test` | 12,107 | 2 | 117 MB | |
| `index-train` | 228,619 | 22 | 2,166 MB | the rest of the retrieval corpus, no title |
| `index-val` | 26,652 | 3 | 251 MB | |
| `index-test` | 28,704 | 3 | 268 MB | |

405,116 samples over 392,324 images, 3.86 GB. Splitting `pairs` out is not
tidiness: `pairs-train` is 942 MB of that, and one undivided series would make
every training epoch read the other 2.9 GB to train on none of it. Shards are pure in split as well
as role, so a job that globs `pairs-train-*.tar` cannot physically read a val
pixel — a stronger guarantee than filtering at load time, which is one line away
from being wrong.

### The packer found a leak the catalog's own checks could not see

Asserting the no-image-in-two-splits invariant on the bytes about to be written,
rather than on the table they came from, caught **five images heading 57 catalog
rows (43 of them in `pairs`) that were in two or three splits at once**.

`catalog.main_image_path` is taken straight from the listings and never passes
through the product-image edge table, so pruning the generic images (the >50
products cap, above) left those rows pointing at a dropped image — and, because
the prune also removed the edge that would have unioned those listings into one
component, the split was assigned to each of them independently. Both existing
checks read `product_images`, where the offending edges were already gone, so
both reported zero. The build now drops listings headed by a generic image, for
the reason already written at the cap: an image on more products than the cap
cannot be a retrieval target, so a listing headed by one has no valid target at
all. A third invariant, on `catalog.main_image_id`, closes the class.

Small — 0.04% of rows — and it would have put the same pixels in train and val.

### The split was not reproducible, and no invariant could tell

Chasing the above surfaced a larger one. `assign_splits` keyed each component on
its union-find **root**, which is whichever member happened to arrive first, and
the edge list comes out of an unordered DuckDB scan. Rebuilding an unchanged
archive moved whole components between splits: train +118, val -214, test +39.
Every leak check passed throughout, because components stayed intact and only
their labels moved — which is exactly why it went unnoticed, and why
"reproducible from a clean clone" was false.

The key is now the component's smallest `product_id`, which is a function of the
graph rather than of the traversal. Two consecutive full builds now produce a
byte-identical assignment over all 145,614 rows, and a test shuffles the edge
list 25 ways and demands one answer.

### The pack is byte-reproducible; the first version was not

Two packs of one input are now byte-identical across all 42 shards. The first
attempt was not, and the way it failed is worth recording: shard count, per-shard
sample counts and per-shard **sizes** all matched exactly, and only the SHA-256s
disagreed.

`product_id` is not unique in `pairs` — the same ASIN is listed per marketplace —
so the sample key carries an occurrence suffix, assigned by `row_number()` over
(`main_image_id`, `title`). 101 groups of rows tie on that pair, 99 of them
differing only in `title_lang`, so which row got `-00` was whatever the scan
happened to return. Every `en_XX` tag is five characters, which is why the sizes
never moved. The ordering now covers every column that can distinguish two rows,
and the index sidecars' product arrays are sorted in Python rather than trusting
a grouped aggregate to keep an order it never promised.

### Sizes are the tar's, not the pixels'

`--shard-bytes` budgets the archive. Every member costs a 512-byte header and is
padded to a 512-byte boundary, which on 7.4 KB images plus two sidecars is ~31%
overhead: 2.94 GB of images becomes 3.86 GB of shards. The first version counted
payload and quietly produced 120 MB shards from a 100 MB budget; both numbers are
published per series in `manifest.json`, along with a SHA-256 per shard — tar
headers are written with fixed mtime and ownership, so two packs of one input are
byte-identical and a run record can cite a shard set by hash.

```bash
python -m scripts.pack_shards                 # 42 shards, ~30s
python -m scripts.pack_shards --roles pairs   # training set only, 0.9 GB
```

---

## Fetching user-supplied URLs

`src/vislens/fetch/image_fetch.py` is the only code in either repo that dereferences a URL a user
supplied, so it carries six controls, each with its own regression test:

1. **https only.**
2. **Exact-match host allowlist** — suffix matching passes `m.media-amazon.com.attacker.net`.
3. **Resolve DNS here, validate every returned address, then connect to the validated address with
   SNI and `Host` pinned to the original name.** Validating a hostname and then handing it to an
   HTTP client is a DNS-rebinding TOCTOU. The address gate ends in an `is_global` allowlist rather
   than a denylist of named predicates, because `100.64.0.0/10` (RFC 6598 carrier-grade NAT) is
   `is_private == False` on Python 3.11 and slips through one.
4. **Redirects are re-validated**, not followed — a 302 to `169.254.169.254` defeats 1-3.
5. **Byte cap enforced on bytes received**, never on `Content-Length`.
6. **Magic-byte sniffing**, not the extension and not `Content-Type`.

The product-page path is best-effort and **gives up rather than working around a refusal**. A
detected bot challenge returns `blocked` with "upload the files instead"; nothing is solved,
evaded, or retried, and a test asserts no circumvention tooling appears in the module. Note also
that Amazon serves non-browser clients a placeholder gallery, so that path returns `partial` and
declines to guess which image is the main one.

Before any of this is enabled, the gate is a **measured RSS delta inside the service container**
against its 512 MB cap. Weights are only part of the cost — the onnxruntime arena does not reliably
return memory to the OS, and the failure signal is a silent `exit 137`. There is no off-the-shelf
int8 vision-only CLIP export (fp32 is 352 MB, fp16 176 MB), so it gets quantized here and the
recall delta published.

---

## Run it

```bash
uv venv --python 3.11 && uv pip install -e ".[dev,data]" && .venv/bin/pytest
```

275 tests (72 fetch controls, 65 rules, 33 service, 30 near-duplicate, 26 catalog build,
14 shard packing, 26 training), no network, no API key and no GPU — which is why the whole suite
runs in CI on every push. The 26 training tests need the `train` extra and skip without it; a
dedicated CI job installs CPU torch and runs them, so the main job can go on asserting that torch
is **absent** from a serving install.
Then start the service:

```bash
.venv/bin/python -m uvicorn vislens.service.app:app --port 8100
```

```bash
curl -s localhost:8100/healthz
```

`POST /audit/upload` takes multipart image files and is the default path — no
network, no ToS surface. `POST /audit/urls` takes CDN image URLs, or an `asin`
for a best-effort product-page read. Pass `main_index` to say which image is the
main one; without it the three main-image-only rules are measured but not
claimed as verdicts.

Python is pinned to **3.11** to match Kaggle's GPU runtime — not ListingLens' 3.13. The two repos
do not share a venv.

The `data` extra is what the suite needs beyond `dev` — the catalog-build and shard-packing
tests import DuckDB. **Serving needs no extras at all**, and that is the point of how the
dependencies are split:

| Install | site-packages | What it is for |
|---|---|---|
| `pip install .` | **118 MB** | The audit service — pillow, numpy, scipy, fastapi, uvicorn, requests |
| `.[data]` | 208 MB | Catalog build and shard packing — duckdb, pandas |
| `.[bench]` | 212 MB | The retrieval benchmark — onnxruntime, faiss. Nothing imports either yet |
| `.[train]` | — | torch, on Kaggle only. Never a serving path |

Five packages used to sit in the default set, so an install for serving fetched **426 MB** to run
a request path that loads PIL, numpy and scipy and nothing else — measured by importing
`vislens.service.app` in a subprocess and reading `sys.modules`, which is how the split was
derived. Four of them moved to the extras above. The fifth, **pyarrow, was dropped outright**: at
122 MB it was the largest single package in the repo, and nothing imports it — DuckDB's parquet
`COPY` and its pandas replacement scan are both native C++, pandas treats pyarrow as optional, and
the 40 catalog and packing tests pass without it.

That is 308 MB off a serving install, against a 512 MB instance, and CI asserts both halves of the
claim: the heavy five stay out, and what remains still imports the service. (Sizes are `du` on a
fresh install before byte-compilation, macOS arm64 wheels; Linux differs in the details, not the
conclusion.)

`torch` lives only in the `train` extra and is never installed into anything that serves a
request. ListingLens bans it outright: torch plus sentence-transformers cost ~490 MiB RSS *before
reading a weight* and were OOM-killed under a 512 MB cap.

---

## CI

One job, no secrets, no network, no GPU — and that is a property of the design
rather than luck. Every check here is deterministic: the compliance rules are
arithmetic over pixels, the perceptual hashes are fixed functions of the bytes,
the request-forgery tests reject before connecting, and the catalog build runs
against a miniature ABO fixture the test writes (the real archives are 83 MB
and 3 GB).

Contrast the sibling project's judged agent eval, which costs Groq tokens, is
bounded to roughly one run a day, and carries a ~37% run-to-run noise floor.
That cannot live in CI. This can, and it means something when it passes.

Three guards beyond lint and tests, each enforcing a rule the repo would
otherwise only *state*:

- **torch must be absent from a default install.** It belongs to the `train`
  extra alone. If it appears, the ONNX serving path has silently regained
  ~490 MiB of import overhead and the reason the sibling project OOM-killed at
  512 MB is back.
- **a serving install must stay small, and must still work.** A second job
  installs the default set with no extras and asserts both directions: none of
  torch, onnxruntime, faiss, duckdb, pyarrow or pandas is present, *and*
  `vislens.service.app` imports on what is left, loading nothing heavier than
  PIL, numpy and scipy. The test job cannot make that assertion — it installs
  `.[dev,data]`, so duckdb is legitimately on its path and it has no way to
  tell a serving dependency from its own. Only a clean install can.
- **no archives, parquet or weights may be tracked.** Download scripts rebuild
  everything; a 3 GB archive in git history is not something you undo.

---

## Why this is a separate repo

The predecessor's container has ~42 MiB of headroom on a 512 MB free-tier instance. A vision
encoder does not fit, and the dependency that would bring it is the one that was deliberately
removed. So all vision code lives here, and ListingLens holds only a thin HTTP client.

## What this repo does not claim

- Not "147K training pairs." The `en_US` filter takes Amazon Berkeley Objects to roughly **26K**
  English-titled products. The language-agnostic image corpus (~140K) and the English pairs table
  (~26K) are separate tables and **both counts get published**, so neither stands in for the other.
- No PySpark claim. The catalog join is 83 MB of gzipped JSON, which is one DuckDB query.
- Not a personalization system. Nothing here personalizes anything.
- Not a text or logo classifier. See above.
