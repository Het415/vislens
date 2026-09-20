"""Pack the ABO images into WebDataset shards.

Kaggle allows a notebook ~500 output files, and the archive holds 398,212
images. Loose files are not an option, so the images go into tar shards and the
training loop streams them — which is what WebDataset is for, and it is also
simply faster: one sequential read per shard instead of 398K opens on a disk
shared with the rest of the runtime.

**The source tar is never extracted.** `abo-images-small.tar` is 3.0 GB and
already a tar; extracting it to repack it would write 398K files to do nothing
but change how they are grouped. Instead the plan — which image goes to which
shard, under which key — is computed from the catalog parquet first, and then
one streaming pass copies bytes across. That pass is I/O-bound and takes about
as long as reading the archive once.

**Bytes are copied, never re-encoded.** No resize, no re-compression, no colour
conversion. Partly because re-encoding 398K JPEGs is slow and lossy for no
gain, but mostly because of `CLAUDE.md` §9: preprocessing has exactly one
definition, in `vislens.data.transforms`. A packer that resized to 224 would be
a second one, and the failure mode is silent — training and ONNX export would
disagree about what a pixel is, and the export parity test would be comparing
two things that were both already wrong.

Two roles, and an image appears in exactly one of them per split:

*   `pairs` — one sample per `pairs.parquet` row: image, title, metadata. This
    is the contrastive training set, and it is a shard series of its own so a
    training run reads only the bytes it uses. Folding it into one undivided
    series would make every epoch read 3.0 GB to train on 0.9 GB of it.
*   `index` — every remaining packable image, no title. These complete the
    retrieval corpus for image-to-image eval (`CLAUDE.md` §6: the index is the
    full catalog, the queries are held out). A consumer that wants the whole
    corpus reads both series; nothing is in both.

Shards are pure in split as well as role, so a training job that globs
`pairs-train-*.tar` cannot physically read a val pixel. That is a stronger
guarantee than filtering at load time, which is a line of code away from being
wrong.

Usage:
    python -m scripts.pack_shards                    # full pack, ~3.0 GB
    python -m scripts.pack_shards --roles pairs      # training set only
    python -m scripts.pack_shards --subset 500       # local smoke
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import io
import json
import os
import pathlib
import sys
import tarfile
import time
from dataclasses import dataclass, field

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Overridable so the test suite can point the packer at a miniature archive.
# CI has no ABO data, and the end-to-end test builds a synthetic catalog and a
# synthetic tar rather than skipping — the logic worth guarding is the plan,
# the invariants and the rolling writer, none of which need 398K real images.
ABO = pathlib.Path(os.getenv("VISLENS_ABO_DIR", ROOT / "data" / "abo"))
CATALOG = pathlib.Path(os.getenv("VISLENS_CATALOG_DIR", ROOT / "data" / "catalog"))
OUT = pathlib.Path(os.getenv("VISLENS_SHARDS_DIR", ROOT / "data" / "shards"))

# Members below this prefix are the images; the archive also carries a LICENSE
# at the root, which is not one.
MEMBER_PREFIX = "images/small/"

# 100 MB per shard. The usual WebDataset advice is 100 MB–1 GB, tuned for object
# storage where per-request latency dominates; on Kaggle the shards sit on local
# disk, so the trade-off is different and the smaller end is better. Measured at
# 100 MB: the full pack is 42 shards and `pairs-train` is 10, which leaves the
# shard-level shuffle something to work with while staying far under the output
# cap. Shards are not split mid-sample, so a shard overshoots by at most one
# sample.
#
# The budget is the size of the tar, not of the pixels in it. Every member costs
# a 512-byte header and is padded to a 512-byte boundary, which on images
# averaging 7.4 KB plus two sidecars of a few hundred bytes is ~31% overhead:
# 2.94 GB of images becomes 3.86 GB of shards. Measuring the payload instead, as
# the first version did, quietly turned a 100 MB budget into 120 MB shards.
SHARD_BYTES = 100_000_000

# Kaggle's limit is ~500 files per notebook output. 450 leaves room for
# checkpoints and metrics alongside the shards, and is a tripwire rather than a
# target: at the default shard size the real pack lands near 34.
MAX_SHARDS = 450


@dataclass
class Sample:
    """One WebDataset sample: a key, the bytes to copy, and its sidecar."""

    role: str
    split: str
    key: str
    meta: dict
    title: str | None = None


@dataclass
class ShardStats:
    shards: list[dict] = field(default_factory=list)
    samples: int = 0
    bytes: int = 0
    payload_bytes: int = 0


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ShardWriter:
    """Rolls a new tar every `shard_bytes`, and keeps samples intact.

    Tar headers are written with fixed mtime, ownership and mode. Without that,
    two packs of identical inputs produce shards that differ in every header, so
    the manifest's checksums would only ever prove that a file had not been
    touched since it was written — not that the pack is reproducible.
    """

    def __init__(self, out: pathlib.Path, role: str, split: str, shard_bytes: int):
        self.out = out
        self.role = role
        self.split = split
        self.shard_bytes = shard_bytes
        self.stats = ShardStats()
        self._tar: tarfile.TarFile | None = None
        self._path: pathlib.Path | None = None
        self._n = 0
        self._samples = 0

    def _open(self) -> None:
        self._path = self.out / f"{self.role}-{self.split}-{self._n:06d}.tar"
        # USTAR rather than the default PAX. Both write 512-byte headers for
        # names and sizes this small, but PAX emits extended headers on rules
        # that have shifted between Python versions, and a shard set whose
        # checksums depend on the interpreter that packed it is not reproducible
        # in any sense a run record can rely on.
        self._tar = tarfile.open(self._path, "w", format=tarfile.USTAR_FORMAT)
        self._samples = 0

    def _close(self) -> None:
        if self._tar is None or self._path is None:
            return
        self._tar.close()
        self.stats.shards.append(
            {
                "name": self._path.name,
                "samples": self._samples,
                "bytes": self._path.stat().st_size,
                "sha256": sha256_file(self._path),
            }
        )
        self.stats.bytes += self._path.stat().st_size
        self._tar = None
        self._n += 1

    def _add(self, name: str, payload: bytes) -> None:
        assert self._tar is not None
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        info.mtime = 0
        info.mode = 0o644
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        self._tar.addfile(info, io.BytesIO(payload))

    def write(self, sample: Sample, suffix: str, payload: bytes) -> None:
        # Roll BEFORE the sample, never inside it: WebDataset groups a sample
        # from consecutive members sharing a key, so a split sample is two
        # broken ones rather than one large shard.
        if self._tar is None or self._tar.offset >= self.shard_bytes:
            self._close()
            self._open()
        stem = f"{sample.role}/{sample.key}"
        sidecar = json.dumps(sample.meta, sort_keys=True).encode()
        self._add(f"{stem}{suffix}", payload)
        self._add(f"{stem}.json", sidecar)
        self.stats.payload_bytes += len(payload) + len(sidecar)
        if sample.title is not None:
            title = sample.title.encode()
            self._add(f"{stem}.txt", title)
            self.stats.payload_bytes += len(title)
        self._samples += 1
        self.stats.samples += 1

    def close(self) -> ShardStats:
        self._close()
        return self.stats


def build_plan(
    con: duckdb.DuckDBPyConnection, roles: tuple[str, ...], subset: int | None
) -> tuple[dict[str, list[Sample]], dict]:
    """Decide every sample before a single byte is read.

    The plan is keyed by archive path because that is what the streaming pass
    can look up, and because one path can serve several products — up to the
    generic-image cap of 50 — so a path maps to a *list* of samples, not one.
    """
    pairs_path = (CATALOG / "pairs.parquet").as_posix()
    images_path = (CATALOG / "product_images.parquet").as_posix()
    catalog_path = (CATALOG / "catalog.parquet").as_posix()
    stats: dict = {}

    # `plausible_product_photo` is the aspect-ratio filter from the catalog
    # build: 4:1 or wider is a banner or a divider strip. The catalog keeps
    # those rows and flags them, which is right for a published table; a shard
    # feeding a training loop should not carry them at all.
    con.execute(
        f"""
        CREATE TABLE packable AS
        SELECT DISTINCT path, split, image_id, orig_width, orig_height
        FROM '{images_path}' WHERE plausible_product_photo
        """
    )

    limit = f"LIMIT {subset}" if subset else ""
    con.execute(
        f"""
        CREATE TABLE pairs_plan AS
        SELECT p.product_id, p.title, p.title_lang, p.product_type, p.split,
               p.main_image_id AS image_id, p.main_image_path AS path,
               k.orig_width, k.orig_height,
               row_number() OVER (
                   PARTITION BY p.product_id
                   ORDER BY p.main_image_id, p.main_image_path, p.title,
                            p.title_lang, p.product_type
               ) - 1 AS occurrence
        FROM '{pairs_path}' p
        JOIN packable k ON k.path = p.main_image_path
        ORDER BY p.product_id, p.main_image_id, p.main_image_path, p.title,
                 p.title_lang, p.product_type
        {limit}
        """
    )
    total_pairs = con.execute(f"SELECT count(*) FROM '{pairs_path}'").fetchone()[0]
    kept_pairs = con.execute("SELECT count(*) FROM pairs_plan").fetchone()[0]
    stats["pairs_rows"] = total_pairs
    stats["pairs_packed"] = kept_pairs
    stats["pairs_dropped_unpackable"] = total_pairs - kept_pairs if not subset else None

    # An image serving several products is stored once. The sidecar carries the
    # full product list, so a consumer can still answer "what is in this
    # photograph" without a join back to the parquet.
    con.execute(
        f"""
        CREATE TABLE index_plan AS
        SELECT k.path, k.split, k.image_id, k.orig_width, k.orig_height,
               json_group_array(json_object(
                   'product_id', i.product_id,
                   'product_type', c.product_type,
                   'is_main', i.is_main
               )) AS products
        FROM packable k
        JOIN '{images_path}' i ON i.path = k.path
        JOIN '{catalog_path}' c ON c.product_id = i.product_id
        WHERE k.path NOT IN (SELECT path FROM pairs_plan)
        GROUP BY k.path, k.split, k.image_id, k.orig_width, k.orig_height
        ORDER BY k.path
        {limit}
        """
    )
    stats["index_images"] = con.execute("SELECT count(*) FROM index_plan").fetchone()[0]

    # The ordering above lists every column that can distinguish two rows, and
    # that is load-bearing rather than belt-and-braces. Ordering on
    # (main_image_id, title) alone left 101 groups of rows tied — 99 of them
    # differing only in `title_lang` — so which row got occurrence 00 was
    # whatever the scan returned, and two packs of one input disagreed. It
    # survived the reproducibility check for a while because every `en_XX` tag
    # is five characters: the shards came out byte-for-byte the same LENGTH,
    # identical sample counts, identical sizes, different contents. With the
    # ordering total, a remaining tie means the rows are identical and so are
    # the samples they produce.
    plan: dict[str, list[Sample]] = collections.defaultdict(list)
    if "pairs" in roles:
        for row in con.execute(
            "SELECT product_id, title, title_lang, product_type, split, image_id, path,"
            " orig_width, orig_height, occurrence FROM pairs_plan"
            " ORDER BY path, product_id, occurrence"
        ).fetchall():
            (pid, title, lang, ptype, split, image_id, path, w, h, occ) = row
            # `product_id` is NOT unique in this table — the same ASIN is listed
            # in several marketplaces with different titles and, usually,
            # different main images (724 such rows in the full build). The
            # occurrence index makes the key unique without inventing an id.
            plan[path].append(
                Sample(
                    role="pairs",
                    split=split,
                    key=f"{pid}-{occ:02d}",
                    title=title,
                    meta={
                        "product_id": pid,
                        "image_id": image_id,
                        "path": path,
                        "split": split,
                        "title": title,
                        "title_lang": lang,
                        "product_type": ptype,
                        "orig_width": w,
                        "orig_height": h,
                        "is_main": True,
                    },
                )
            )
    if "index" in roles:
        for path, split, image_id, w, h, products in con.execute(
            "SELECT path, split, image_id, orig_width, orig_height, products FROM index_plan"
            " ORDER BY path"
        ).fetchall():
            plan[path].append(
                Sample(
                    role="index",
                    split=split,
                    key=image_id,
                    meta={
                        "image_id": image_id,
                        "path": path,
                        "split": split,
                        "orig_width": w,
                        "orig_height": h,
                        # Sorted in Python, not left to `json_group_array`:
                        # DuckDB does not promise an order inside a grouped
                        # aggregate, and an array that reshuffles between runs
                        # is the same reproducibility bug as the one above.
                        "products": sorted(
                            json.loads(products),
                            key=lambda row: (row["product_id"], row["is_main"]),
                        ),
                    },
                )
            )
    return plan, stats


def check_invariants(plan: dict[str, list[Sample]]) -> dict:
    """Assert on the bytes about to be written, not on the table they came from.

    The split invariant is checked in `build_catalog` too, and that is not
    redundant: the checks there read `product_images`, and the first run of this
    function found five images heading rows in two or three splits at once
    because `catalog.main_image_path` never passes through that table. A leak
    that reaches the shards is one that reaches training, so the last place to
    catch it is here.
    """
    by_path_split = {path: {s.split for s in samples} for path, samples in plan.items()}
    straddling = {p: sorted(v) for p, v in by_path_split.items() if len(v) > 1}
    if straddling:
        head = list(straddling.items())[:5]
        detail = "; ".join(f"{p} in {v}" for p, v in head)
        sys.exit(
            f"ABORT: {len(straddling)} image paths span more than one split ({detail})"
            " — the shards would put the same pixels in train and val"
        )

    keys = collections.Counter((s.role, s.key) for samples in plan.values() for s in samples)
    duplicates = [k for k, n in keys.items() if n > 1]
    if duplicates:
        sys.exit(
            f"ABORT: {len(duplicates)} duplicate sample keys, e.g. {duplicates[:5]}"
            " — WebDataset would merge them into one sample"
        )
    return {"planned_paths": len(plan), "planned_samples": sum(keys.values())}


def pack(plan: dict[str, list[Sample]], source: pathlib.Path, shard_bytes: int) -> dict:
    """One sequential pass over the source archive.

    Stream mode (`r|`) rather than random access: the plan is already known, so
    there is nothing to seek for, and streaming holds one member in memory at a
    time instead of an index of 398K.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    writers: dict[tuple[str, str], ShardWriter] = {}
    seen: set[str] = set()
    started = time.time()
    members = 0

    with tarfile.open(source, "r|") as tf:
        for member in tf:
            members += 1
            if not member.isfile() or not member.name.startswith(MEMBER_PREFIX):
                continue
            path = member.name[len(MEMBER_PREFIX) :]
            samples = plan.get(path)
            if not samples:
                continue
            handle = tf.extractfile(member)
            if handle is None:  # pragma: no cover - defensive
                continue
            payload = handle.read()
            seen.add(path)
            suffix = pathlib.PurePosixPath(path).suffix
            for sample in samples:
                key = (sample.role, sample.split)
                if key not in writers:
                    writers[key] = ShardWriter(OUT, sample.role, sample.split, shard_bytes)
                writers[key].write(sample, suffix, payload)
            if len(seen) % 50_000 == 0:
                rate = len(seen) / max(time.time() - started, 1e-9)
                print(f"  {len(seen):,} images packed ({rate:,.0f}/s)")

    missing = sorted(set(plan) - seen)
    if missing:
        sys.exit(
            f"ABORT: {len(missing)} planned images are absent from {source.name}, "
            f"e.g. {missing[:5]} — the catalog and the archive disagree"
        )

    series = {}
    for (role, split), writer in sorted(writers.items()):
        stats = writer.close()
        series[f"{role}-{split}"] = {
            "samples": stats.samples,
            "bytes": stats.bytes,
            # Published so the gap between these two reads as a documented
            # property of tar rather than a surprise when the upload turns out
            # larger than the archive it came from.
            "payload_bytes": stats.payload_bytes,
            "shards": stats.shards,
        }

    shard_total = sum(len(s["shards"]) for s in series.values())
    if shard_total > MAX_SHARDS:
        sys.exit(
            f"ABORT: {shard_total} shards exceeds the {MAX_SHARDS} tripwire "
            f"(Kaggle allows ~500 notebook output files) — raise --shard-bytes"
        )
    return {
        "series": series,
        "members_scanned": members,
        "images_packed": len(seen),
        "shard_count": shard_total,
        "seconds": round(time.time() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roles",
        default="pairs,index",
        help="which shard series to write: pairs (training), index (retrieval corpus)",
    )
    parser.add_argument("--shard-bytes", type=int, default=SHARD_BYTES)
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="cap samples per role, for the smoke path that runs in seconds",
    )
    args = parser.parse_args()

    roles = tuple(r.strip() for r in args.roles.split(",") if r.strip())
    unknown = set(roles) - {"pairs", "index"}
    if unknown:
        sys.exit(f"unknown role(s): {sorted(unknown)}")

    source = ABO / "abo-images-small.tar"
    if not source.exists():
        sys.exit(f"no image archive at {source}")

    con = duckdb.connect()
    print(f"planning {', '.join(roles)} ...")
    plan, plan_stats = build_plan(con, roles, args.subset)
    invariants = check_invariants(plan)
    print(f"  {invariants['planned_samples']:,} samples over {len(plan):,} images")

    print(f"packing from {source.name} ({source.stat().st_size / 1e9:.1f} GB) ...")
    result = pack(plan, source, args.shard_bytes)

    catalog_report = json.loads((CATALOG / "build_report.json").read_text())
    manifest = {
        "built": dt.date.today().isoformat(),
        "roles": list(roles),
        "subset": args.subset,
        "shard_bytes": args.shard_bytes,
        "source": {
            "name": source.name,
            "bytes": source.stat().st_size,
            "members": result.pop("members_scanned"),
        },
        "catalog": {
            "built": catalog_report.get("built"),
            "counts": catalog_report.get("counts"),
        },
        **plan_stats,
        **invariants,
        **result,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print()
    summary = {k: v for k, v in manifest.items() if k != "series"}
    summary["series"] = {
        name: f"{s['samples']:,} samples, {len(s['shards'])} shards, {s['bytes'] / 1e6:.0f} MB"
        for name, s in manifest["series"].items()
    }
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {manifest['shard_count']} shards and a manifest to {OUT}")


if __name__ == "__main__":
    main()
