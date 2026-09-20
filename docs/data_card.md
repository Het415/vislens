# Data card — Amazon Berkeley Objects

Built 2026-09-20 by
`scripts/build_catalog.py`. Source archives are **not committed**; the script
rebuilds everything from `abo-listings.tar` (83 MB) and
`abo-images-small.tar` (3.0 GB).

License: CC BY 4.0, bundled in both archives.

## What was built, and why it is two tables

| Table | Rows | What it is |
|---|---|---|
| `catalog.parquet` | 145,671 | One row per listing with a resolvable main image. **Language-agnostic.** The index corpus, the classification training set, and the image-to-image eval corpus |  # noqa: E501
| `product_images.parquet` | 535,734 | One row per (product, image). ~3.73 images per product |  # noqa: E501
| `pairs.parquet` | 121,350 | The subset with a usable English title, carrying `title_lang`. The contrastive training set and the text-to-image eval corpus |  # noqa: E501

The build plan mandated filtering titles to `en_US`. Measured here, that leaves
**25,775** listings — against 147,702 total. One table would have
forced a choice between a text tower poisoned by mixed languages and a corpus
four-fifths discarded. `product_type` carries no `language_tag`, so the
classification task and the index legitimately use the full catalogue.

**Both counts are published, and neither stands in for the other.** This repo
does not claim "147K training pairs".

## Title language

| `title_lang` | Rows |
|---|---|
| `en_IN` | 76,212 |
| `en_US` | 25,775 |
| `en_GB` | 7,907 |
| `en_CA` | 6,481 |
| `en_AU` | 2,103 |
| `en_AE` | 1,545 |
| `en_SG` | 1,327 |

`en_US` alone is 25,775; the full `en_*` ladder reaches
121,350, a 4.7x
difference. That makes the mandated data-scaling sweep informative across a real
range rather than a narrow one, and `title_lang` is recorded per row so the
choice is ablatable instead of baked in.

## Splits

test: 14,283 · train: 114,587 · val: 16,801

Assigned by `sha1(product_id) % 10` — order-independent, reproducible from a
clean clone with no split file to lose, and stable when rows are added. **Split
by product, never by image**: a product's five photos all land in one split, or
every retrieval number is inflated. The build aborts if any product spans more
than one split, and `tests/test_split_leakage.py` asserts the same property.

## Quirks worth knowing

- **`images.csv.gz` `height`/`width` describe the ORIGINAL image, not the 256px
  file that ships in the archive.** Verified: a row declaring 1920x1080 is
  256x144 on disk. Columns are therefore named `orig_width` / `orig_height`;
  the spec's `image_width` would be quietly wrong, and anything computing
  geometry from it would be working on numbers that do not describe its pixels.
- **256px is the cap**, which matches a resize-256 / crop-224 pipeline exactly —
  and means training above 224 is not possible from this archive.
- **6,007 images are more elongated than 4.0:1** and are
  flagged `plausible_product_photo = false`. These are banners and divider
  strips (rows declaring 2560x71 exist, arriving as 256x7), not product photos.
- **0 listings** named a `main_image_id` absent from the image
  metadata and were dropped after the join.
- **575 listings** had no `main_image_id` at all.
- **0 listings** carried more than one `product_type`; the first is
  taken, and the count is logged rather than assumed away.
- Only 2 of 535,734 image files are PNG; the rest are JPEG.
