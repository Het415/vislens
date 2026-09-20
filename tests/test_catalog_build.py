"""Catalog build: the language ladder, the join, and the split.

`CLAUDE.md` requires a test with every data-transform commit, and the split
logic in particular earned one the hard way — the first build shipped an
assertion for the *wrong* invariant and passed while 27,554 images spanned two
splits.

CI has no ABO archives (the listings tar is 83 MB and the image metadata lives
inside a 3 GB tar), so the end-to-end test constructs a miniature ABO layout.
That is the right thing to test regardless: the logic worth guarding is the
join, the ladder and the split, none of which need 147K real rows.
"""

from __future__ import annotations

import collections
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_catalog import (  # noqa: E402
    EN_LADDER,
    GENERIC_IMAGE_MAX_PRODUCTS,
    SPLIT_PATTERN,
    _pick_localised,
    assign_splits,
    split_for,
)

# ── The language ladder ───────────────────────────────────────────────────────


def test_en_us_wins_when_present():
    entries = [
        {"language_tag": "de_DE", "value": "Deutscher Titel"},
        {"language_tag": "en_US", "value": "US title"},
        {"language_tag": "en_GB", "value": "GB title"},
    ]
    assert _pick_localised(entries, EN_LADDER) == ("US title", "en_US")


def test_ladder_falls_back_in_order_and_records_the_tag():
    """The tag is recorded so the choice is ablatable rather than baked in.

    This matters more than it looks: `en_US` alone is 26,424 rows of ABO while
    the full ladder reaches 122,734, so which tag supplied a title is a
    variable the scaling sweep needs to control for.
    """
    entries = [
        {"language_tag": "ja_JP", "value": "japanese"},
        {"language_tag": "en_IN", "value": "indian english"},
        {"language_tag": "en_GB", "value": "british english"},
    ]
    value, tag = _pick_localised(entries, EN_LADDER)
    assert (value, tag) == ("british english", "en_GB")  # en_GB precedes en_IN
    assert EN_LADDER.index("en_GB") < EN_LADDER.index("en_IN")


def test_no_english_title_yields_none_rather_than_a_foreign_one():
    """Silently substituting a German title is the multilingual trap. A row
    with no English title belongs in the catalog and out of the pairs table."""
    entries = [
        {"language_tag": "de_DE", "value": "Deutscher Titel"},
        {"language_tag": "zh_CN", "value": "chinese"},
    ]
    assert _pick_localised(entries, EN_LADDER) == (None, None)


def test_untagged_and_empty_entries_are_ignored():
    assert _pick_localised([{"value": "no tag"}], EN_LADDER) == (None, None)
    assert _pick_localised([{"language_tag": "en_US", "value": ""}], EN_LADDER) == (None, None)
    assert _pick_localised(None, EN_LADDER) == (None, None)
    assert _pick_localised([], EN_LADDER) == (None, None)


# ── The split key ─────────────────────────────────────────────────────────────


def test_split_is_deterministic_and_order_independent():
    """Hash-based rather than a stored shuffle, so a clean clone reproduces it
    with no split file to lose and adding rows does not reshuffle."""
    keys = [f"COMP{i}" for i in range(200)]
    first = [split_for(k) for k in keys]
    assert [split_for(k) for k in reversed(keys)] == list(reversed(first))
    assert first == [split_for(k) for k in keys]


def test_split_pattern_is_eighty_ten_ten():
    assert collections.Counter(SPLIT_PATTERN) == {"train": 8, "val": 1, "test": 1}


def test_split_distribution_is_roughly_the_pattern():
    counts = collections.Counter(split_for(f"component-{i}") for i in range(4000))
    assert 0.75 < counts["train"] / 4000 < 0.85
    assert 0.05 < counts["val"] / 4000 < 0.15
    assert 0.05 < counts["test"] / 4000 < 0.15


# ── The split itself: the invariant the plan's rule misses ────────────────────


def test_products_sharing_an_image_land_in_the_same_split():
    """The whole reason the split key is a component root and not a product id.

    ABO lists the same product across marketplaces with byte-identical assets,
    so two products can hold the same pixels. Splitting by `product_id` exactly
    as the build plan specifies left 27,554 images spanning two splits and
    touched 40.3% of rows — while passing its own check.
    """
    pairs = [
        ("P1", "imgA"),
        ("P2", "imgA"),  # shares imgA with P1
        ("P3", "imgB"),
        ("P4", "imgB"),  # shares imgB with P3
        ("P5", "imgC"),  # alone
    ]
    assignment, _ = assign_splits(pairs)

    assert assignment["P1"] == assignment["P2"]
    assert assignment["P3"] == assignment["P4"]


def test_no_image_can_span_two_splits():
    """Stated as the invariant rather than as a property of one example."""
    pairs = [(f"P{p}", f"img{p // 3}") for p in range(60)]  # chains of 3
    assignment, _ = assign_splits(pairs)

    by_image: dict[str, set[str]] = collections.defaultdict(set)
    for product, image in pairs:
        by_image[image].add(assignment[product])
    assert all(len(splits) == 1 for splits in by_image.values())


def test_transitive_sharing_merges_a_whole_chain():
    """A shares with B, B shares with C: all three must move together, even
    though A and C share nothing directly."""
    pairs = [("A", "i1"), ("B", "i1"), ("B", "i2"), ("C", "i2")]
    assignment, stats = assign_splits(pairs)

    assert assignment["A"] == assignment["B"] == assignment["C"]
    assert stats["components"] == 1


def test_component_stats_are_reported():
    pairs = [("A", "i1"), ("B", "i1"), ("C", "i2"), ("D", "i3")]
    _, stats = assign_splits(pairs)

    assert stats["components"] == 3          # {A,B}, {C}, {D}
    assert stats["largest_component"] == 2
    assert stats["singleton_components"] == 2
    assert stats["largest_component_share"] == pytest.approx(0.5)


def test_a_generic_image_would_merge_everything_if_not_pruned():
    """Why `GENERIC_IMAGE_MAX_PRODUCTS` exists.

    Measured on the real archive: un-pruned, 23 boilerplate images — three
    attached to ~33,300 products each — produced a giant component of 36.6% of
    the catalogue and forced an 89/5/5 split. This is that effect in miniature:
    passing a shared asset into `assign_splits` collapses the graph, which is
    why pruning happens upstream of it.
    """
    products = [f"P{i}" for i in range(30)]
    own = [(p, f"img-{p}") for p in products]

    pruned, pruned_stats = assign_splits(own)
    assert pruned_stats["components"] == 30
    assert len(set(pruned.values())) > 1  # a real split exists

    with_boilerplate = own + [(p, "BOILERPLATE") for p in products]
    _, collapsed = assign_splits(with_boilerplate)
    assert collapsed["components"] == 1
    assert collapsed["largest_component"] == 30


def test_generic_cap_is_a_real_threshold():
    assert 10 <= GENERIC_IMAGE_MAX_PRODUCTS <= 200


# ── End to end, on a miniature ABO layout ────────────────────────────────────


def _write_fixture_abo(base: Path) -> None:
    """A miniature ABO: two listing shards and an image metadata CSV.

    Deliberately includes the cases that broke the real build — a shared image
    across two products, a boilerplate image on many products, a listing with
    no English title, one with no main image, and dimensions that differ from
    the on-disk file.

    60 products, not 24: `BOILER` has to sit on more than
    `GENERIC_IMAGE_MAX_PRODUCTS` products for the pruning path to be exercised
    at all, and a fixture that silently skips the code it is meant to test is
    worse than no fixture.
    """
    meta = base / "listings" / "metadata"
    meta.mkdir(parents=True)

    rows = []
    for i in range(60):
        pid = f"PROD{i:03d}"
        row = {
            "item_id": pid,
            "product_type": [{"value": "SPEAKER" if i % 2 else "SHOES"}],
            "country": "US",
            "marketplace": "Amazon",
            "main_image_id": f"m{i:03d}",
            "other_image_id": [f"o{i:03d}a", "BOILER"],
            "item_name": [{"language_tag": "en_US", "value": f"Product {i}"}],
            "brand": [{"language_tag": "en_US", "value": "Acme"}],
        }
        if i == 5:
            # No English title: belongs in catalog, out of pairs.
            row["item_name"] = [{"language_tag": "de_DE", "value": "Deutsch"}]
        if i == 6:
            # No main image at all: dropped before the join.
            del row["main_image_id"]
        if i == 7:
            # Shares its main image with PROD008 — the leak case.
            row["main_image_id"] = "m008"
        rows.append(row)

    half = len(rows) // 2
    for name, chunk in (("listings_0", rows[:half]), ("listings_1", rows[half:])):
        with gzip.open(meta / f"{name}.json.gz", "wt", encoding="utf-8") as f:
            for row in chunk:
                f.write(json.dumps(row) + "\n")

    images_dir = base / "images" / "metadata"
    images_dir.mkdir(parents=True)
    lines = ["image_id,height,width,path"]
    for i in range(60):
        # 1080x1920 declared: the archive's CSV carries ORIGINAL dimensions,
        # not those of the 256px file on disk.
        lines.append(f"m{i:03d},1080,1920,ab/m{i:03d}.jpg")
        lines.append(f"o{i:03d}a,800,800,cd/o{i:03d}a.jpg")
    lines.append("BOILER,50,2560,ef/boiler.jpg")  # elongated boilerplate banner
    with gzip.open(images_dir / "images.csv.gz", "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    """Run the real build script against the fixture layout, once."""
    root = tmp_path_factory.mktemp("abo")
    abo, out, docs = root / "abo", root / "catalog", root / "docs"
    _write_fixture_abo(abo)

    env = {
        **os.environ,
        "VISLENS_ABO_DIR": str(abo),
        "VISLENS_CATALOG_DIR": str(out),
        "VISLENS_DOCS_DIR": str(docs),
    }
    result = subprocess.run(
        [sys.executable, "-m", "scripts.build_catalog"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"build failed:\n{result.stdout}\n{result.stderr}"
    return {
        "out": out,
        "docs": docs,
        "report": json.loads((out / "build_report.json").read_text()),
    }


def test_build_produces_all_three_tables(built):
    for name in ("catalog", "product_images", "pairs"):
        assert (built["out"] / f"{name}.parquet").exists(), name


def test_listing_without_a_main_image_is_dropped(built):
    con = duckdb.connect()
    ids = {
        r[0]
        for r in con.execute(
            f"SELECT product_id FROM '{built['out'] / 'catalog.parquet'}'"
        ).fetchall()
    }
    assert "PROD006" not in ids  # had no main_image_id
    assert built["report"]["dropped_no_main_image"] == 1


def test_non_english_listing_is_in_the_catalog_but_not_the_pairs(built):
    """The two-table split, in one assertion. A German-titled listing is a
    perfectly good index and classification row and a useless training pair."""
    con = duckdb.connect()
    catalog = {
        r[0] for r in con.execute(
            f"SELECT product_id FROM '{built['out'] / 'catalog.parquet'}'"
        ).fetchall()
    }
    pairs = {
        r[0] for r in con.execute(
            f"SELECT product_id FROM '{built['out'] / 'pairs.parquet'}'"
        ).fetchall()
    }
    assert "PROD005" in catalog
    assert "PROD005" not in pairs


def test_dimensions_are_named_orig_because_they_describe_the_original(built):
    """Verified against the real archive: a row declaring 1920x1080 is 256x144
    on disk. The spec's `image_width` naming would be quietly wrong, and
    anything computing geometry from it would be using numbers that do not
    describe its pixels."""
    con = duckdb.connect()
    cols = {
        r[0] for r in con.execute(
            f"DESCRIBE SELECT * FROM '{built['out'] / 'product_images.parquet'}'"
        ).fetchall()
    }
    assert {"orig_width", "orig_height"} <= cols
    assert not {"image_width", "image_height"} & cols

    width, height = con.execute(
        f"SELECT orig_width, orig_height FROM '{built['out'] / 'product_images.parquet'}' "
        "WHERE image_id = 'm000'"
    ).fetchone()
    assert (width, height) == (1920, 1080)


def test_the_boilerplate_image_is_pruned(built):
    """`BOILER` sits on every product, so it is a generic asset: it cannot be a
    retrieval target, and left in it would collapse the split graph."""
    con = duckdb.connect()
    remaining = con.execute(
        f"SELECT count(*) FROM '{built['out'] / 'product_images.parquet'}' "
        "WHERE image_id = 'BOILER'"
    ).fetchone()[0]
    assert remaining == 0
    assert built["report"]["generic_images_dropped"] >= 1


def test_elongated_images_are_flagged_not_silently_kept(built):
    con = duckdb.connect()
    flagged = con.execute(
        f"SELECT count(*) FROM '{built['out'] / 'product_images.parquet'}' "
        "WHERE NOT plausible_product_photo"
    ).fetchone()[0]
    # Every m-series image is 1920x1080 (1.78:1, plausible); the only
    # implausible one was BOILER at 2560x50, and it was pruned.
    assert flagged == 0


def test_no_product_and_no_image_spans_two_splits(built):
    """BOTH invariants. The product-level one is what the build plan specifies
    and it is not sufficient — it passed while 27,554 images crossed."""
    con = duckdb.connect()
    path = built["out"] / "product_images.parquet"

    product_overlap = con.execute(
        f"SELECT count(*) FROM (SELECT product_id FROM '{path}' "
        "GROUP BY product_id HAVING count(DISTINCT split) > 1)"
    ).fetchone()[0]
    image_overlap = con.execute(
        f"SELECT count(*) FROM (SELECT image_id FROM '{path}' "
        "GROUP BY image_id HAVING count(DISTINCT split) > 1)"
    ).fetchone()[0]

    assert product_overlap == 0
    assert image_overlap == 0


def test_products_sharing_a_main_image_share_a_split(built):
    """PROD007 was given PROD008's main image, which is exactly the real
    dataset's marketplace-duplicate shape."""
    con = duckdb.connect()
    splits = dict(
        con.execute(
            f"SELECT product_id, split FROM '{built['out'] / 'catalog.parquet'}' "
            "WHERE product_id IN ('PROD007', 'PROD008')"
        ).fetchall()
    )
    assert splits["PROD007"] == splits["PROD008"]


def test_the_build_writes_a_data_card_with_measured_counts(built):
    card = (built["docs"] / "data_card.md").read_text()
    assert "Data card" in card
    assert "orig_width" in card  # the quirk is documented, not just handled
    assert "Split by product, never by image" in card
    assert str(built["report"]["counts"]["catalog"]) in card.replace(",", "")


def test_report_records_the_generic_cap_it_used(built):
    """A threshold that shaped the output has to be recoverable from the
    output, or a future reader cannot reproduce the split."""
    assert built["report"]["generic_image_max_products"] == GENERIC_IMAGE_MAX_PRODUCTS
    assert "component_largest_component_share" in built["report"]
