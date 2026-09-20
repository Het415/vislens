"""Build the ABO catalog tables.

Produces **two** tables, and that split is the whole point:

*   `catalog.parquet` — every listing with a resolvable main image,
    **language-agnostic**. This is the index corpus, the classification training
    set, and the image-to-image eval corpus. `product_type` carries no
    `language_tag` (verified: it is `[{"value": "SHOES"}]`), so it is sound to
    keep the full catalogue here.
*   `pairs.parquet` — the subset with a usable English title, carrying a
    `title_lang` column recording which tag supplied it. This is the contrastive
    training set and the text-to-image eval corpus.

**Why two.** The build plan mandated filtering titles to `en_US`, which measured
against the real archive leaves **26,424 of 147,702 listings (17.9%)** — smaller
than the fallback dataset the plan proposed as a backup. One table would have
forced a choice between a poisoned text tower and a corpus four-fifths thrown
away. Two tables keep an honest ~147K index-scale claim *and* clean text, and
both counts get published so neither can stand in for the other.

**No Spark.** This is 83 MB of gzipped JSON and a 398K-row CSV join. The parse
is Python because the language-preference ladder over nested multilingual arrays
is far clearer imperatively than as a lateral unnest; the join, aggregation and
parquet write are DuckDB, which is already a dependency of the sibling project.
Standing up Spark to join 83 MB invites the follow-up question nobody wants.

Usage:
    python -m scripts.build_catalog                # full build
    python -m scripts.build_catalog --subset 500   # local smoke, <5 min
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
import pathlib
import sys

import duckdb
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Overridable so the test suite can point the whole pipeline at synthetic
# fixtures. CI has no ABO archives — the listings tar is 83 MB and the image
# metadata lives inside a 3 GB tar — so the end-to-end build test constructs a
# miniature ABO layout instead. That is the right thing to test anyway: the
# logic worth guarding is the join, the language ladder and the split, none of
# which need 147K real rows to exercise.
ABO = pathlib.Path(os.getenv("VISLENS_ABO_DIR", ROOT / "data" / "abo"))
OUT = pathlib.Path(os.getenv("VISLENS_CATALOG_DIR", ROOT / "data" / "catalog"))
DOCS = pathlib.Path(os.getenv("VISLENS_DOCS_DIR", ROOT / "docs"))

# Ordered preference. `en_US` first, then the rest of the English locales by
# rough corpus size. `title_lang` records which one supplied each row, so the
# choice is ablatable rather than baked in — and so a mixed-tag column can never
# masquerade as a clean one, which is the subtler form of the multilingual trap.
EN_LADDER = ("en_US", "en_GB", "en_IN", "en_CA", "en_AU", "en_AE", "en_SG")

SPLIT_PATTERN = ("train",) * 8 + ("val", "test")

# An image attached to more than this many products is a generic asset — a brand
# banner, a badge, a size chart — not a photograph of one product.
#
# This is a leak control AND a data-quality filter, and it was added after
# measuring, not by intuition. Splitting by `product_id` exactly as the build
# plan specifies still leaked badly: **27,554 images appeared in more than one
# split, touching 285,692 rows (40.3%)**, because ABO lists the same product
# across marketplaces sharing byte-identical assets. So a product-level split
# passes its own check while the pixels cross freely.
#
# The fix is to split connected components of the product-image graph, but
# un-pruned that graph has a giant component of 53,140 products (36.6%) — driven
# by 23 images, three of them attached to ~33,300 products each — which forces a
# 89/5/5 split. Dropping images above this cap removes 70 images (0.018% of the
# corpus) and collapses the largest component to 1.9%, restoring 80/10/10.
#
# Measured sweep: K of 25, 50 and 200 all land within 0.3 points of each other
# on largest-component share, so the result is not a knife-edge tuning artefact.
# Independently of leakage, an image belonging to 33,313 products cannot serve as
# a retrieval target — "which product is this?" has no single answer for it.
GENERIC_IMAGE_MAX_PRODUCTS = 50

# Long-tail cap: keep the product types covering this share of the catalogue and
# bucket the rest as `other`.
PRODUCT_TYPE_COVERAGE = 0.90

# An image this elongated is a banner or a divider strip, not a product photo.
# Measured in the archive: rows declaring 2560x71 and 2560x52 exist, which
# become 256x7 and 256x5 on disk.
MAX_ASPECT = 4.0


def split_for(key: str) -> str:
    """Deterministic split from a hash of a key.

    Hash-based rather than a stored shuffle: order-independent, reproducible
    from a clean clone with no split file to lose, and stable when rows are
    added later.

    The key is a **connected-component root**, not a product id — see
    `assign_splits` and `GENERIC_IMAGE_MAX_PRODUCTS` for why a product-level key
    is not sufficient on this dataset.
    """
    digest = int(hashlib.sha1(key.encode()).hexdigest(), 16)
    return SPLIT_PATTERN[digest % 10]


def assign_splits(pairs: list[tuple[str, str]]) -> tuple[dict[str, str], dict]:
    """Split products so that **no image ever spans two splits**.

    Products are unioned whenever they share a (non-generic) image, and the
    split is assigned per component. This is strictly stronger than the build
    plan's split-by-product rule, which this dataset defeats: the same photo
    appears under several marketplace listings, so two products in different
    splits can hold identical pixels.

    `pairs` must already have generic images removed, or the components merge
    into one blob.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_image: dict[str, list[str]] = collections.defaultdict(list)
    for product_id, image_id in pairs:
        find(product_id)
        by_image[image_id].append(product_id)

    for members in by_image.values():
        root = find(members[0])
        for other in members[1:]:
            other_root = find(other)
            if other_root != root:
                parent[other_root] = root

    assignment = {product: split_for(find(product)) for product in parent}
    sizes = collections.Counter(find(p) for p in parent)
    ordered = sorted(sizes.values(), reverse=True)
    stats = {
        "components": len(sizes),
        "largest_component": ordered[0] if ordered else 0,
        "largest_component_share": round(ordered[0] / max(len(parent), 1), 4) if ordered else 0.0,
        "singleton_components": sum(1 for s in ordered if s == 1),
    }
    return assignment, stats


def _pick_localised(entries: list | None, ladder: tuple[str, ...]) -> tuple[str | None, str | None]:
    """First value whose language_tag appears in `ladder`, with its tag."""
    if not entries:
        return None, None
    by_tag = {}
    for entry in entries:
        tag = entry.get("language_tag")
        if tag and tag not in by_tag and entry.get("value"):
            by_tag[tag] = entry["value"]
    for tag in ladder:
        if tag in by_tag:
            return by_tag[tag], tag
    return None, None


def parse_listings(subset: int | None = None) -> tuple[pd.DataFrame, dict]:
    shards = sorted((ABO / "listings" / "metadata").glob("listings_*.json.gz"))
    if not shards:
        sys.exit(f"no listing shards under {ABO / 'listings' / 'metadata'}")

    rows: list[dict] = []
    stats = collections.Counter()
    multi_product_type: list[str] = []

    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                stats["listings_total"] += 1
                row = json.loads(line)
                product_id = row.get("item_id")
                if not product_id:
                    stats["dropped_no_item_id"] += 1
                    continue

                types = row.get("product_type") or []
                if len(types) > 1:
                    # Logged rather than silently taking [0]: the spec asserts
                    # exactly one, and an unlogged assumption is how a schema
                    # surprise becomes a quiet mislabel.
                    if len(multi_product_type) < 20:
                        multi_product_type.append(product_id)
                    stats["multi_product_type"] += 1
                product_type = (types[0].get("value") if types else None) or "UNKNOWN"

                if not row.get("main_image_id"):
                    stats["dropped_no_main_image"] += 1
                    continue

                title, title_lang = _pick_localised(row.get("item_name"), EN_LADDER)
                brand, _ = _pick_localised(row.get("brand"), EN_LADDER)
                colour, _ = _pick_localised(row.get("color"), EN_LADDER)

                rows.append(
                    {
                        "product_id": product_id,
                        "product_type_raw": product_type,
                        "country": row.get("country"),
                        "marketplace": row.get("marketplace"),
                        "main_image_id": row["main_image_id"],
                        "other_image_ids": row.get("other_image_id") or [],
                        "title": title,
                        "title_lang": title_lang,
                        "brand": brand,
                        "color": colour,
                    }
                )
                if subset and len(rows) >= subset:
                    break
        if subset and len(rows) >= subset:
            break

    stats["multi_product_type_examples"] = multi_product_type[:5]  # type: ignore[assignment]
    return pd.DataFrame(rows), dict(stats)


def load_images() -> pd.DataFrame:
    """The image metadata CSV.

    `height`/`width` describe the **original** image, not the 256px file that
    ships in `abo-images-small.tar`. Verified: a row declaring 1920x1080 is
    256x144 on disk. Hence `orig_*` naming — the spec's `image_width` would be
    quietly wrong, and anything downstream computing geometry from it would be
    working on numbers that do not describe the pixels it is holding.
    """
    path = ABO / "images" / "metadata" / "images.csv.gz"
    with gzip.open(path, "rt") as f:
        frame = pd.DataFrame(list(csv.DictReader(f)))
    frame["orig_height"] = frame["height"].astype(int)
    frame["orig_width"] = frame["width"].astype(int)
    return frame[["image_id", "orig_width", "orig_height", "path"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="cap listings, for the local smoke path that runs in under 5 minutes",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"parsing listings{f' (subset {args.subset})' if args.subset else ''} ...")
    listings, stats = parse_listings(args.subset)
    print(f"  {len(listings):,} listings with a main image")

    print("loading image metadata ...")
    images = load_images()
    print(f"  {len(images):,} image rows")

    con = duckdb.connect()
    con.register("listings", listings)
    con.register("images", images)

    # Long-tail cap on product_type, computed from the data rather than guessed.
    type_counts = listings["product_type_raw"].value_counts()
    cumulative = type_counts.cumsum() / len(listings)
    kept_types = set(type_counts[cumulative <= PRODUCT_TYPE_COVERAGE].index)
    if not kept_types:  # tiny subsets
        kept_types = set(type_counts.index[:1])
    con.register("kept_types", pd.DataFrame({"product_type_raw": sorted(kept_types)}))

    # Step 1: the raw product-image edges, before any split exists. The split
    # depends on this graph, so it cannot be assigned during parsing.
    con.execute(
        """
        CREATE TABLE edges_raw AS
        SELECT l.product_id, l.main_image_id AS image_id, TRUE AS is_main
        FROM listings l
        UNION ALL
        SELECT l.product_id, u.image_id, FALSE AS is_main
        FROM listings l, UNNEST(l.other_image_ids) AS u(image_id)
        """
    )
    con.execute(
        """
        CREATE TABLE edges AS
        SELECT e.product_id, e.image_id, e.is_main,
               i.path, i.orig_width, i.orig_height
        FROM edges_raw e JOIN images i ON e.image_id = i.image_id
        """
    )

    # Step 2: identify generic assets and exclude them. See
    # GENERIC_IMAGE_MAX_PRODUCTS — this is what stops the component graph
    # collapsing into one blob, and it removes images that could not be
    # retrieval targets anyway.
    con.execute(
        f"""
        CREATE TABLE generic_images AS
        SELECT image_id, count(DISTINCT product_id) AS n_products
        FROM edges GROUP BY image_id
        HAVING count(DISTINCT product_id) > {GENERIC_IMAGE_MAX_PRODUCTS}
        """
    )
    generic_n = con.execute("SELECT count(*) FROM generic_images").fetchone()[0]
    dropped_edges = con.execute(
        "SELECT count(*) FROM edges WHERE image_id IN (SELECT image_id FROM generic_images)"
    ).fetchone()[0]
    con.execute(
        "DELETE FROM edges WHERE image_id IN (SELECT image_id FROM generic_images)"
    )
    print(f"  dropped {generic_n:,} generic images ({dropped_edges:,} edges)")

    # Step 3: split by connected component of the pruned product-image graph.
    edge_pairs = con.execute("SELECT product_id, image_id FROM edges").fetchall()
    assignment, comp_stats = assign_splits(edge_pairs)
    print(
        f"  {comp_stats['components']:,} components, "
        f"largest {comp_stats['largest_component']:,} "
        f"({comp_stats['largest_component_share']:.1%})"
    )
    con.register(
        "splits",
        pd.DataFrame(
            {"product_id": list(assignment), "split": [assignment[p] for p in assignment]}
        ),
    )

    con.execute(
        """
        CREATE TABLE catalog AS
        SELECT
            l.product_id,
            CASE WHEN k.product_type_raw IS NULL THEN 'other' ELSE l.product_type_raw END
                AS product_type,
            l.product_type_raw,
            l.country,
            l.marketplace,
            l.title,
            l.title_lang,
            l.brand,
            l.color,
            s.split,
            l.main_image_id,
            i.path         AS main_image_path,
            i.orig_width   AS main_orig_width,
            i.orig_height  AS main_orig_height
        FROM listings l
        JOIN splits s ON l.product_id = s.product_id
        LEFT JOIN images i ON l.main_image_id = i.image_id
        LEFT JOIN kept_types k ON l.product_type_raw = k.product_type_raw
        """
    )

    join_misses = con.execute(
        "SELECT count(*) FROM catalog WHERE main_image_path IS NULL"
    ).fetchone()[0]
    con.execute("DELETE FROM catalog WHERE main_image_path IS NULL")

    con.execute(
        f"""
        CREATE TABLE product_images AS
        SELECT e.product_id, s.split, e.image_id, e.path,
               e.orig_width, e.orig_height, e.is_main,
               (e.orig_width::DOUBLE / e.orig_height)
                   BETWEEN {1 / MAX_ASPECT} AND {MAX_ASPECT} AS plausible_product_photo
        FROM edges e
        JOIN splits s ON e.product_id = s.product_id
        WHERE e.product_id IN (SELECT product_id FROM catalog)
        """
    )

    # The English pairs table.
    con.execute(
        """
        CREATE TABLE pairs AS
        SELECT product_id, title, title_lang, product_type, split,
               main_image_id, main_image_path
        FROM catalog
        WHERE title IS NOT NULL
        """
    )

    for name in ("catalog", "product_images", "pairs"):
        con.execute(f"COPY {name} TO '{OUT / f'{name}.parquet'}' (FORMAT PARQUET)")

    counts = {
        name: con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        for name in ("catalog", "product_images", "pairs")
    }
    lang = con.execute(
        "SELECT title_lang, count(*) c FROM pairs GROUP BY 1 ORDER BY c DESC"
    ).fetchall()
    splits = con.execute(
        "SELECT split, count(*) c FROM catalog GROUP BY 1 ORDER BY 1"
    ).fetchall()
    en_us_only = con.execute(
        "SELECT count(*) FROM pairs WHERE title_lang = 'en_US'"
    ).fetchone()[0]
    implausible = con.execute(
        "SELECT count(*) FROM product_images WHERE NOT plausible_product_photo"
    ).fetchone()[0]
    per_product = con.execute(
        "SELECT avg(n) FROM (SELECT count(*) n FROM product_images GROUP BY product_id)"
    ).fetchone()[0]

    # Leakage checks, run here as well as in the test suite: a data-transform
    # script that can silently produce a leaking split is worse than one that
    # refuses to finish.
    #
    # BOTH invariants are asserted. The product-level one is what the build plan
    # specifies and it is not sufficient — it passed while 27,554 images spanned
    # splits, because the same photo appears under several marketplace listings.
    product_overlap = con.execute(
        """
        SELECT count(*) FROM (
            SELECT product_id FROM product_images GROUP BY product_id
            HAVING count(DISTINCT split) > 1
        )
        """
    ).fetchone()[0]
    if product_overlap:
        sys.exit(f"ABORT: {product_overlap} products span more than one split")

    image_overlap = con.execute(
        """
        SELECT count(*) FROM (
            SELECT image_id FROM product_images GROUP BY image_id
            HAVING count(DISTINCT split) > 1
        )
        """
    ).fetchone()[0]
    if image_overlap:
        sys.exit(f"ABORT: {image_overlap} images span more than one split")

    report = {
        "built": dt.date.today().isoformat(),
        "subset": args.subset,
        "listings_total": stats.get("listings_total"),
        "dropped_no_main_image": stats.get("dropped_no_main_image", 0),
        "main_image_join_misses": join_misses,
        "multi_product_type": stats.get("multi_product_type", 0),
        "counts": counts,
        "en_us_titles": en_us_only,
        "title_langs": lang,
        "splits": splits,
        "product_types_kept": len(kept_types),
        "generic_images_dropped": generic_n,
        "generic_image_edges_dropped": dropped_edges,
        "generic_image_max_products": GENERIC_IMAGE_MAX_PRODUCTS,
        **{f"component_{k}": v for k, v in comp_stats.items()},
        "implausible_aspect_images": implausible,
        "images_per_product": round(float(per_product or 0), 2),
    }
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    DOCS.mkdir(exist_ok=True)
    (DOCS / "data_card.md").write_text(render_data_card(report))

    print()
    summary = {k: v for k, v in report.items() if k != "title_langs"}
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {OUT}/*.parquet and {DOCS / 'data_card.md'}")


def render_data_card(r: dict) -> str:
    langs = "\n".join(f"| `{tag}` | {count:,} |" for tag, count in r["title_langs"][:12])
    splits = " · ".join(f"{s}: {c:,}" for s, c in r["splits"])
    components = r.get("component_components", 0)
    largest = r.get("component_largest_component", 0)
    largest_share = r.get("component_largest_component_share", 0.0)
    return f"""# Data card — Amazon Berkeley Objects

Built {r['built']}{" (SUBSET: " + str(r['subset']) + " listings)" if r['subset'] else ""} by
`scripts/build_catalog.py`. Source archives are **not committed**; the script
rebuilds everything from `abo-listings.tar` (83 MB) and
`abo-images-small.tar` (3.0 GB).

License: CC BY 4.0, bundled in both archives.

## What was built, and why it is two tables

| Table | Rows | What it is |
|---|---|---|
| `catalog.parquet` | {r['counts']['catalog']:,} | One row per listing with a resolvable main image. **Language-agnostic.** The index corpus, the classification training set, and the image-to-image eval corpus |  # noqa: E501
| `product_images.parquet` | {r['counts']['product_images']:,} | One row per (product, image). ~{r['images_per_product']} images per product |  # noqa: E501
| `pairs.parquet` | {r['counts']['pairs']:,} | The subset with a usable English title, carrying `title_lang`. The contrastive training set and the text-to-image eval corpus |  # noqa: E501

The build plan mandated filtering titles to `en_US`. Measured here, that leaves
**{r['en_us_titles']:,}** listings — against {r['listings_total']:,} total. One table would have
forced a choice between a text tower poisoned by mixed languages and a corpus
four-fifths discarded. `product_type` carries no `language_tag`, so the
classification task and the index legitimately use the full catalogue.

**Both counts are published, and neither stands in for the other.** This repo
does not claim "147K training pairs".

## Title language

| `title_lang` | Rows |
|---|---|
{langs}

`en_US` alone is {r['en_us_titles']:,}; the full `en_*` ladder reaches
{r['counts']['pairs']:,}, a {r['counts']['pairs'] / max(r['en_us_titles'], 1):.1f}x
difference. That makes the mandated data-scaling sweep informative across a real
range rather than a narrow one, and `title_lang` is recorded per row so the
choice is ablatable instead of baked in.

## Splits

{splits}

Assigned by `sha1(component_root) % 10` — order-independent, reproducible from
a clean clone with no split file to lose, and stable when rows are added.

**Split by product, never by image — and that is not sufficient here.** A
product's photos must all land in one split or every retrieval number is
inflated, but on this dataset the same photo appears under several marketplace
listings, so two *different* products can hold byte-identical pixels. Splitting
by `product_id` alone left **27,554 images spanning two splits, touching 40.3%
of rows**, while passing its own check.

So the split key is a **connected component** of the product-image graph: two
products sharing any image land together. {components:,} components, largest
{largest:,} ({largest_share:.1%}).

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
- **{r['implausible_aspect_images']:,} images are more elongated than {MAX_ASPECT}:1** and are
  flagged `plausible_product_photo = false`. These are banners and divider
  strips (rows declaring 2560x71 exist, arriving as 256x7), not product photos.
- **{r['main_image_join_misses']:,} listings** named a `main_image_id` absent from the image
  metadata and were dropped after the join.
- **{r['dropped_no_main_image']:,} listings** had no `main_image_id` at all.
- **{r['multi_product_type']:,} listings** carried more than one `product_type`; the first is
  taken, and the count is logged rather than assumed away.
- Only 2 of {r['counts']['product_images']:,} image files are PNG; the rest are JPEG.
"""


if __name__ == "__main__":
    main()
