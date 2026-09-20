"""Shard packing: the plan, the invariants, and the rolling writer.

CI has no ABO archives, so this builds a miniature one — the same 60-product
layout the catalog test uses, plus a tar of real (tiny) JPEGs — and runs the
real scripts end to end. What is worth guarding here is not image handling,
which is a byte copy, but everything around it: that a sample's members stay
together, that a shard rolls between samples and never inside one, that the
same image cannot reach two splits, and that two packs of one input produce the
same bytes.

The fixture writer is imported rather than copied. Two definitions of "what ABO
looks like" would drift, and the first symptom would be a test passing against
a shape the real archive no longer has.
"""

from __future__ import annotations

import collections
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import duckdb
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_catalog_build import _write_fixture_abo  # noqa: E402


def _jpeg(seed: int) -> bytes:
    """A small but real JPEG. Seeded, so a determinism check means something."""
    rng = __import__("random").Random(seed)
    img = Image.new("RGB", (64, 64))
    img.putdata(
        [(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(64 * 64)]
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _write_fixture_tar(abo: Path) -> None:
    """Pack every path the image CSV declares into an archive shaped like ABO's."""
    import csv
    import gzip

    with gzip.open(abo / "images" / "metadata" / "images.csv.gz", "rt") as f:
        paths = [row["path"] for row in csv.DictReader(f)]

    with tarfile.open(abo / "abo-images-small.tar", "w") as tf:
        info = tarfile.TarInfo("LICENSE-CC-BY-4.0.txt")  # the archive's non-image member
        payload = b"cc-by-4.0"
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
        for i, path in enumerate(paths):
            payload = _jpeg(i)
            info = tarfile.TarInfo(f"images/small/{path}")
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))


def _run(script: str, env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", script, *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def _env(root: Path) -> dict:
    return {
        **os.environ,
        "VISLENS_ABO_DIR": str(root / "abo"),
        "VISLENS_CATALOG_DIR": str(root / "catalog"),
        "VISLENS_DOCS_DIR": str(root / "docs"),
        "VISLENS_SHARDS_DIR": str(root / "shards"),
    }


def _read_shards(shards: Path) -> dict[str, list[dict]]:
    """Every shard, decoded into samples in the order they appear in the tar."""
    out: dict[str, list[dict]] = {}
    for tar_path in sorted(shards.glob("*.tar")):
        samples: list[dict] = []
        with tarfile.open(tar_path) as tf:
            for member in tf:
                key, _, suffix = member.name.rpartition(".")
                if not samples or samples[-1]["key"] != key:
                    samples.append({"key": key, "suffixes": [], "payload": {}})
                samples[-1]["suffixes"].append(suffix)
                samples[-1]["payload"][suffix] = tf.extractfile(member).read()
        out[tar_path.name] = samples
    return out


@pytest.fixture(scope="module")
def packed(tmp_path_factory) -> dict:
    """Build the catalog and pack it, once, with a shard size that forces rolls."""
    root = tmp_path_factory.mktemp("pack")
    _write_fixture_abo(root / "abo")
    _write_fixture_tar(root / "abo")
    env = _env(root)

    build = _run("scripts.build_catalog", env)
    assert build.returncode == 0, f"{build.stdout}\n{build.stderr}"

    # 8 KB: the fixture's JPEGs are ~3 KB, so shards hold two or three samples
    # and the rolling path is exercised. The default 100 MB would produce one
    # shard per series and test nothing about it.
    result = _run("scripts.pack_shards", env, "--shard-bytes", "8000")
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    shards = root / "shards"
    return {
        "root": root,
        "env": env,
        "shards": shards,
        "manifest": json.loads((shards / "manifest.json").read_text()),
        "by_shard": _read_shards(shards),
        "catalog": root / "catalog",
    }


def test_shards_are_named_by_role_and_split(packed):
    names = sorted(packed["by_shard"])
    assert names, "no shards written"
    for name in names:
        role, split, index = name[: -len(".tar")].rsplit("-", 2)
        assert role in {"pairs", "index"}, name
        assert split in {"train", "val", "test"}, name
        assert index.isdigit() and len(index) == 6, name


def test_a_sample_is_never_split_across_two_shards(packed):
    """The reason the writer rolls before a sample rather than between members.

    WebDataset builds a sample from consecutive members sharing a key; half a
    sample at the end of one shard is not recoverable from the next one, it is
    simply two broken samples.
    """
    seen: collections.Counter = collections.Counter()
    for samples in packed["by_shard"].values():
        for sample in samples:
            seen[sample["key"]] += 1
    assert seen, "no samples"
    assert [k for k, n in seen.items() if n > 1] == []


def test_rolling_actually_happened(packed):
    """Otherwise the assertion above is vacuous."""
    per_series: collections.Counter = collections.Counter()
    for name in packed["by_shard"]:
        per_series[name.rsplit("-", 1)[0]] += 1
    assert max(per_series.values()) > 1, per_series


def test_shard_size_is_the_size_of_the_shard(packed):
    """`--shard-bytes` has to mean the file, not the pixels inside it.

    The first version counted payload and ignored tar's 512-byte headers and
    padding, so a 100 MB budget produced 120 MB shards on the real archive.
    The rule checked here is exact: a shard rolls once the archive reaches the
    budget, so dropping its LAST sample must leave it under.
    """
    budget = 8000
    for name, samples in packed["by_shard"].items():
        on_disk = []
        for sample in samples:
            cost = sum(
                512 + (len(payload) + 511) // 512 * 512 for payload in sample["payload"].values()
            )
            on_disk.append(cost)
        assert sum(on_disk[:-1]) < budget, f"{name} kept writing past the budget"
        assert (packed["shards"] / name).stat().st_size >= sum(on_disk)


def test_pairs_carry_a_title_and_index_samples_do_not(packed):
    """`.txt` is what a CLIP dataloader reads. An index sample has no caption —
    it is a retrieval target, not a training pair — and inventing one would put
    a fabricated caption into the training signal."""
    for name, samples in packed["by_shard"].items():
        role = name.split("-")[0]
        for sample in samples:
            assert "json" in sample["suffixes"], sample["key"]
            if role == "pairs":
                assert "txt" in sample["suffixes"], sample["key"]
                assert (
                    sample["payload"]["txt"].decode()
                    == json.loads(sample["payload"]["json"])["title"]
                )
            else:
                assert "txt" not in sample["suffixes"], sample["key"]


def test_one_asin_with_two_listings_gets_two_keys(packed):
    """`product_id` is not unique in `pairs`: the same ASIN is listed per
    marketplace. The occurrence suffix separates them without inventing an id,
    and the ordering that assigns it has to cover every column that can differ
    — `title_lang` here — or the assignment is whatever the scan returned.
    """
    found = {}
    for name, samples in packed["by_shard"].items():
        if not name.startswith("pairs-"):
            continue
        for sample in samples:
            meta = json.loads(sample["payload"]["json"])
            if meta["product_id"] == "PROD011":
                found[sample["key"]] = meta

    assert sorted(found) == ["pairs/PROD011-00", "pairs/PROD011-01"]
    assert {m["title_lang"] for m in found.values()} == {"en_US", "en_GB"}
    assert found["pairs/PROD011-00"]["title_lang"] == "en_GB"  # en_GB sorts first
    assert {m["path"] for m in found.values()} == {"ab/m011.jpg"}


def test_no_image_reaches_two_splits(packed):
    """The invariant the shards exist to preserve, asserted on the bytes.

    Checked here and not only in the catalog build because the two look at
    different things: the build checks a table, this checks what was written.
    On the real archive the first run of the packer found five images the
    build's own checks could not see.
    """
    splits_by_path: dict[str, set[str]] = collections.defaultdict(set)
    for samples in packed["by_shard"].values():
        for sample in samples:
            meta = json.loads(sample["payload"]["json"])
            splits_by_path[meta["path"]].add(meta["split"])
    assert splits_by_path
    assert {p: s for p, s in splits_by_path.items() if len(s) > 1} == {}


def test_an_image_is_in_one_role_not_both(packed):
    """Roles partition the corpus; overlapping them would pay for the same
    bytes twice and silently weight those images higher in any pass over both."""
    roles_by_path: dict[str, set[str]] = collections.defaultdict(set)
    for name, samples in packed["by_shard"].items():
        role = name.split("-")[0]
        for sample in samples:
            roles_by_path[json.loads(sample["payload"]["json"])["path"]].add(role)
    assert {p: r for p, r in roles_by_path.items() if len(r) > 1} == {}


def test_bytes_are_copied_not_re_encoded(packed):
    """CLAUDE.md §9: preprocessing has exactly one definition. A packer that
    resized would be a second one, and the disagreement would be silent."""
    source = tarfile.open(packed["root"] / "abo" / "abo-images-small.tar")
    original = {
        m.name[len("images/small/") :]: source.extractfile(m).read()
        for m in source.getmembers()
        if m.name.startswith("images/small/")
    }
    checked = 0
    for samples in packed["by_shard"].values():
        for sample in samples:
            meta = json.loads(sample["payload"]["json"])
            assert sample["payload"]["jpg"] == original[meta["path"]], meta["path"]
            checked += 1
    assert checked > 0


def test_manifest_counts_match_the_shards_on_disk(packed):
    """A manifest is a claim about an artifact; this is the claim being checked
    against the artifact rather than against the code that wrote both."""
    manifest = packed["manifest"]
    actual_samples = sum(len(s) for s in packed["by_shard"].values())
    assert manifest["planned_samples"] == actual_samples
    assert manifest["shard_count"] == len(packed["by_shard"])
    for series, info in manifest["series"].items():
        on_disk = [n for n in packed["by_shard"] if n.startswith(f"{series}-")]
        assert len(info["shards"]) == len(on_disk), series
        assert info["samples"] == sum(len(packed["by_shard"][n]) for n in on_disk), series


def test_the_pack_is_reproducible(packed):
    """Fixed tar headers, or the checksums only prove the file was not touched.

    Reproducibility is what lets a run record cite a shard set by hash instead
    of by "the shards that were on the disk that day".
    """
    first = {
        s["name"]: s["sha256"]
        for series in packed["manifest"]["series"].values()
        for s in series["shards"]
    }
    again = _run("scripts.pack_shards", packed["env"], "--shard-bytes", "8000")
    assert again.returncode == 0, f"{again.stdout}\n{again.stderr}"
    manifest = json.loads((packed["shards"] / "manifest.json").read_text())
    second = {
        s["name"]: s["sha256"] for series in manifest["series"].values() for s in series["shards"]
    }
    assert first == second


def test_roles_can_be_packed_alone(packed):
    """The training set is 0.9 GB of a 3.0 GB corpus; a run that only trains
    should not have to move the other 2.1 GB."""
    out = packed["root"] / "shards_pairs_only"
    env = {**packed["env"], "VISLENS_SHARDS_DIR": str(out)}
    result = _run("scripts.pack_shards", env, "--roles", "pairs", "--shard-bytes", "8000")
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert {n.split("-")[0] for n in _read_shards(out)} == {"pairs"}


def test_a_straddling_image_aborts_the_pack(packed):
    """The check earning its keep, against a table doctored to leak.

    This is not hypothetical: the real `pairs.parquet` had five such images
    when the packer was first pointed at it.
    """
    out = packed["root"] / "leaky"
    out.mkdir()
    catalog = packed["catalog"]
    con = duckdb.connect()
    for name in ("catalog", "product_images", "pairs"):
        con.execute(
            f"CREATE TABLE {name} AS SELECT * FROM '{(catalog / f'{name}.parquet').as_posix()}'"
        )
    # PROD007 and PROD008 share a main image, which is what makes this a leak
    # rather than a relabel: move PROD007 alone and one path now heads rows in
    # two splits. Picking an arbitrary product would not do it — most fixture
    # images belong to exactly one product, so moving one moves the whole path
    # with it and nothing straddles.
    current = con.execute("SELECT split FROM pairs WHERE product_id = 'PROD007'").fetchone()[0]
    other = "val" if current != "val" else "test"
    for name in ("catalog", "product_images", "pairs"):
        con.execute(f"UPDATE {name} SET split = '{other}' WHERE product_id = 'PROD007'")
    for name in ("catalog", "product_images", "pairs"):
        con.execute(f"COPY {name} TO '{(out / f'{name}.parquet').as_posix()}' (FORMAT PARQUET)")
    (out / "build_report.json").write_text((catalog / "build_report.json").read_text())

    env = {
        **packed["env"],
        "VISLENS_CATALOG_DIR": str(out),
        "VISLENS_SHARDS_DIR": str(packed["root"] / "shards_leaky"),
    }
    result = _run("scripts.pack_shards", env, "--shard-bytes", "8000")
    assert result.returncode != 0
    assert "span more than one split" in result.stderr + result.stdout


def test_an_image_missing_from_the_archive_aborts_the_pack(packed):
    """The catalog and the archive disagreeing is a build error, not something
    to pack around: a silently smaller training set is a number nobody can
    reproduce later."""
    root = packed["root"] / "truncated"
    root.mkdir()
    abo = root / "abo"
    abo.mkdir()
    (abo / "images").mkdir()
    source = packed["root"] / "abo" / "abo-images-small.tar"
    with tarfile.open(source) as src, tarfile.open(abo / "abo-images-small.tar", "w") as dst:
        members = [m for m in src.getmembers()]
        for member in members[: len(members) // 2]:
            dst.addfile(member, src.extractfile(member))

    env = {
        **packed["env"],
        "VISLENS_ABO_DIR": str(abo),
        "VISLENS_SHARDS_DIR": str(root / "shards"),
    }
    result = _run("scripts.pack_shards", env, "--shard-bytes", "8000")
    assert result.returncode != 0
    assert "absent from" in result.stderr + result.stdout
