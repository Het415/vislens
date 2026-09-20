# Data card — Amazon Berkeley Objects

Built 2026-09-20 by
`scripts/build_catalog.py`. Source archives are **not committed**; the script
rebuilds everything from `abo-listings.tar` (83 MB) and
`abo-images-small.tar` (3.0 GB).

License: CC BY 4.0, bundled in both archives.

## What was built, and why it is two tables

| Table | Rows | What it is |
|---|---|---|
| `catalog.parquet` | 145,614 | One row per listing with a resolvable main image. **Language-agnostic.** The index corpus, the classification training set, and the image-to-image eval corpus |  # noqa: E501
| `product_images.parquet` | 535,602 | One row per (product, image). ~3.73 images per product |  # noqa: E501
| `pairs.parquet` | 121,307 | The subset with a usable English title, carrying `title_lang`. The contrastive training set and the text-to-image eval corpus |  # noqa: E501

The build plan mandated filtering titles to `en_US`. Measured here, that leaves
**25,747** listings — against 147,702 total. One table would have
forced a choice between a text tower poisoned by mixed languages and a corpus
four-fifths discarded. `product_type` carries no `language_tag`, so the
classification task and the index legitimately use the full catalogue.

**Both counts are published, and neither stands in for the other.** This repo
does not claim "147K training pairs".

## Title language

| `title_lang` | Rows |
|---|---|
| `en_IN` | 76,211 |
| `en_US` | 25,747 |
| `en_GB` | 7,905 |
| `en_CA` | 6,478 |
| `en_AU` | 2,102 |
| `en_AE` | 1,537 |
| `en_SG` | 1,327 |

`en_US` alone is 25,747; the full `en_*` ladder reaches
121,307, a 4.7x
difference. That makes the mandated data-scaling sweep informative across a real
range rather than a narrow one, and `title_lang` is recorded per row so the
choice is ablatable instead of baked in.

## Splits

test: 14,575 · train: 116,967 · val: 14,072

Assigned by `sha1(component_root) % 10` — order-independent, reproducible from
a clean clone with no split file to lose, and stable when rows are added.

**Split by product, never by image — and that is not sufficient here.** A
product's photos must all land in one split or every retrieval number is
inflated, but on this dataset the same photo appears under several marketplace
listings, so two *different* products can hold byte-identical pixels. Splitting
by `product_id` alone left **27,554 images spanning two splits, touching 40.3%
of rows**, while passing its own check.

So the split key is a **connected component** of the product-image graph: two
products sharing any image land together. 105,854 components, largest
2,738 (1.9%).

The build **aborts** if any product *or any image* spans more than one split,
and `tests/test_catalog_build.py` asserts both invariants against a miniature
ABO fixture.

## Quirks worth knowing

- **`images.csv.gz` `height`/`width` describe the ORIGINAL image, not the 256px
  file that ships in the archive.** Verified: a row declaring 1920x1080 is
  256x144 on disk. Columns are therefore named `orig_width` / `orig_height`;
  the spec's `image_width` would be quietly wrong, and anything computing
  geometry from it would be working on numbers that do not describe its pixels.
- **256px is the cap**, which matches a resize-256 / crop-224 pipeline exactly —
  and means training above 224 is not possible from this archive.
- **5,999 images are more elongated than 4.0:1** and are
  flagged `plausible_product_photo = false`. These are banners and divider
  strips (rows declaring 2560x71 exist, arriving as 256x7), not product photos.
- **0 listings** named a `main_image_id` absent from the image
  metadata and were dropped after the join.
- **575 listings** had no `main_image_id` at all.
- **0 listings** carried more than one `product_type`; the first is
  taken, and the count is logged rather than assumed away.
- Only 2 of 535,602 image files are PNG; the rest are JPEG.
