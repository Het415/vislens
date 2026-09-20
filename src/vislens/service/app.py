"""The audit service.

A deliberately small FastAPI app: it decodes images, runs the deterministic
compliance checks, and returns the compact payload ListingLens' agent consumes.
It holds no LLM key, no review corpus, and no database — which is the point.
The fetcher in `vislens.fetch.image_fetch` is the only code in either repo that
dereferences a user-supplied URL, and keeping it in a process with nothing worth
stealing is a meaningful reduction in blast radius, independent of the memory
argument for the split.

Three design constraints worth stating, because each shaped the code:

*   **No persistent disk.** The target host offers none, so the audit store is
    an in-memory bounded map. It is a cache, not a record — `GET /audit/{id}`
    can legitimately 404 after a restart or an eviction, and the client has to
    tolerate that rather than treat it as an error.
*   **No stored pixels.** Only hashes, dimensions and derived measurements. With
    no auth and no tenancy, anything retained is effectively public, and hosting
    other people's product photography is a bill and a liability rather than a
    feature.
*   **Memory is the binding constraint**, so decoding is sequential and each
    working array is released before the next image is read. Fetching is
    concurrent because it is network-bound, but bounded — three 8 MB bodies in
    flight is fine, twelve is not.
"""

from __future__ import annotations

import os
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from vislens.fetch.image_fetch import (
    IMAGE_HOSTS,
    MAX_IMAGE_BYTES,
    FetchResult,
    fetch_image,
    fetch_listing_image_urls,
)
from vislens.rules.image_dedup import (
    find_group_mismatches,
    find_near_duplicates,
    load_thresholds,
)
from vislens.rules.image_hash import compute_hashes
from vislens.rules.image_rules import (
    audit_image,
    build_audit_payload,
    decode_for_audit,
    load_rules,
)

MAX_IMAGES_PER_AUDIT = 12
FETCH_CONCURRENCY = 3
AUDIT_STORE_MAX = 64

app = FastAPI(
    title="vislens audit",
    version="0.1.0",
    description="Deterministic Amazon main-image compliance checks over user-supplied images.",
)

# The browser posts uploads straight here rather than through ListingLens,
# deliberately: proxying image bytes through that process would put it back on
# the image path it was split off from, and a backend image proxy is both an
# SSRF amplifier and a bandwidth bill.
#
# `allow_credentials` is False and stays False. This service has no auth and no
# cookies, so there is nothing for a credentialed cross-origin request to carry
# — and leaving it off means a permissive origin list cannot be escalated into
# reading an authenticated response.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "VISLENS_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Bounded LRU. Unbounded would be a slow memory leak on a 512 MB instance, which
# is the failure the parent project already hit once with an unbounded chain
# cache.
_AUDITS: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _remember(audit_id: str, record: dict[str, Any]) -> None:
    _AUDITS[audit_id] = record
    _AUDITS.move_to_end(audit_id)
    while len(_AUDITS) > AUDIT_STORE_MAX:
        _AUDITS.popitem(last=False)


class UrlAuditRequest(BaseModel):
    image_urls: list[str] | None = Field(
        default=None,
        description=f"Image URLs on an allowlisted Amazon CDN host: {sorted(IMAGE_HOSTS)}",
    )
    asin: str | None = Field(
        default=None,
        description=(
            "An ASIN or product URL. Best-effort: Amazon usually refuses "
            "non-browser clients, in which case the response says so and asks "
            "for uploads instead."
        ),
    )
    main_index: int | None = Field(
        default=None,
        description=(
            "Which image is the MAIN image. Three of the checks are "
            "main-image-only rules, so without this they are measured and "
            "reported but not claimed as verdicts."
        ),
    )


def _audit_one(
    data: bytes, *, is_main: bool | None, rules: dict[str, Any], source: str
) -> tuple[list, Any, dict[str, Any]]:
    """Decode, check, hash, and build the detail record for one image.

    Returns (results, hashes, detail). The decoded array goes out of scope on
    return, which is deliberate: holding nine decoded images at once is what
    would put this over the memory cap. The hashes are 64-bit ints, so keeping
    those costs nothing.
    """
    img = decode_for_audit(data, rules)
    results = audit_image(img, rules, is_main=is_main)
    hashes = compute_hashes(img.rgb)
    detail = {
        "source": source,
        "is_main": is_main,
        "sha256": img.sha256,
        "orig_width": img.orig_width,
        "orig_height": img.orig_height,
        "format": img.image_format,
        "n_bytes": img.n_bytes,
        "downscale": round(img.downscale, 4),
        # Full per-check detail, including artifact boxes and modal RGB. This is
        # what the UI renders; the agent gets the compact form instead.
        "checks": [asdict(r) for r in results],
        "hashes": hashes.as_dict(),
    }
    return results, hashes, detail


def _resolve_main(index: int, main_index: int | None) -> bool | None:
    """`None` means unknown, and unknown is not the same as False.

    Unknown downgrades the main-image-only checks to advisory; False asserts the
    image is secondary. Collapsing the two would make the scraped-page path
    silently claim every image is a secondary image.
    """
    if main_index is None:
        return None
    return index == main_index


def _assemble(
    per_image: list[list],
    hashes: list[Any],
    details: list[dict[str, Any]],
    n_supplied: int,
    rules: dict[str, Any],
    notes: list[str],
    sku_groups: list[str | None] | None = None,
) -> dict[str, Any]:
    audit_id = uuid.uuid4().hex[:12]
    payload = build_audit_payload(per_image, n_supplied, rules, audit_id=audit_id)

    thresholds = load_thresholds()
    dedup = find_near_duplicates(hashes, thresholds)
    mismatches = (
        find_group_mismatches(hashes, sku_groups, thresholds) if sku_groups else []
    )

    # Compact for the agent: the clusters are the finding, the pair distances
    # are detail. `method` and `threshold` travel with it so the claim is
    # attributable to a specific calibration rather than floating free.
    compact: dict[str, Any] = {
        "method": dedup.method,
        "threshold": dedup.threshold,
        "clusters": dedup.clusters,
    }
    if dedup.skipped_featureless:
        compact["skipped_no_contrast"] = dedup.skipped_featureless
    if not dedup.calibrated:
        # Never present an uncalibrated detector's output as a finding.
        compact["unavailable"] = dedup.reason or "detector not calibrated"
    if mismatches:
        compact["group_mismatches"] = [
            {"i": m.index, "tagged": m.own_group, "nearest": m.nearest_group}
            for m in mismatches
        ]
    payload["duplicates"] = compact

    # Duplicate images across a listing are a real quality problem, so this
    # reaches the headline — but as a `warn`, not a `fail`. The detector's
    # measured precision is 1.000 on a deliberately small set, and a held-out
    # zero-false-positive observation at that size does not support a
    # rejection-grade claim.
    if dedup.clusters and payload["headline"] == "pass":
        payload["headline"] = "warn"

    if notes:
        payload["notes"] = notes

    _remember(
        audit_id,
        {
            "payload": payload,
            "images": details,
            "duplicates": dedup.as_dict(),
            "group_mismatches": [m.as_dict() for m in mismatches],
        },
    )
    return payload


def _thresholds() -> dict[str, Any]:
    return load_thresholds()


@app.get("/")
def root() -> dict[str, Any]:
    """What this service is and what it exposes.

    Present because the preview and any human poking at the port both land
    here, and a bare 404 tells them nothing.
    """
    return {
        "service": "vislens audit",
        "what": (
            "Deterministic Amazon main-image compliance checks over "
            "user-supplied images. No LLM, no model, no stored pixels."
        ),
        "endpoints": {
            "GET /healthz": "rules version and which checks are actually enabled",
            "POST /audit/upload": (
                "multipart image files — the default path, with no network "
                "and no ToS surface"
            ),
            "POST /audit/urls": (
                "image_urls on an allowlisted Amazon CDN host, or a "
                "best-effort asin"
            ),
            "GET /audit/{audit_id}": (
                "full per-check detail plus hashes and duplicate pairs "
                "(in-memory cache, may 404)"
            ),
            "GET /docs": "OpenAPI UI",
        },
    }


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    rules = load_rules()
    return {
        "status": "ok",
        "rules_version": rules["rules_version"],
        "rules_verified_on": rules["verified_on"],
        # Enabled checks only. A check that is specified but disabled is not a
        # capability, and advertising it as one is how the predecessor ended up
        # with a landing page describing a CLIP model it did not have.
        "checks_enabled": sorted(
            name
            for name, cfg in rules["checks"].items()
            if cfg.get("enabled", True) and cfg.get("tier") != "deferred"
        ),
        "checks_specified_not_enabled": sorted(
            name for name, cfg in rules["checks"].items() if cfg.get("enabled") is False
        ),
        "main_image_only": rules["main_image_only"]["checks"],
        "near_duplicate": {
            "calibrated": bool(_thresholds().get("calibrated")),
            "method": _thresholds().get("selected_method"),
            "calibrated_on": _thresholds().get("calibrated_on"),
            "report": _thresholds().get("calibration_report"),
            "measured": _thresholds().get("measured"),
        },
        "audits_cached": len(_AUDITS),
    }


@app.post("/audit/upload")
async def audit_upload(
    files: list[UploadFile] = File(...),
    main_index: int | None = None,
    sku_groups: str | None = None,
) -> dict[str, Any]:
    """Audit uploaded image files. The path with no network and no ToS surface.

    `sku_groups` is an optional comma-separated tag per file, in order, which
    turns on the group-mismatch finding: an image whose nearest neighbour
    belongs to a different SKU than its own tag.
    """
    if not files:
        raise HTTPException(status_code=422, detail="no files supplied")
    if len(files) > MAX_IMAGES_PER_AUDIT:
        raise HTTPException(
            status_code=422,
            detail=f"{len(files)} images supplied; the cap is {MAX_IMAGES_PER_AUDIT}",
        )

    rules = load_rules()
    per_image, all_hashes, details, notes = [], [], [], []
    tags = [t.strip() or None for t in sku_groups.split(",")] if sku_groups else None
    kept_tags: list[str | None] = []

    for index, upload in enumerate(files):
        data = await upload.read()
        if len(data) > MAX_IMAGE_BYTES:
            notes.append(f"{upload.filename}: larger than {MAX_IMAGE_BYTES} bytes, skipped")
            continue
        try:
            results, hashes, detail = _audit_one(
                data,
                is_main=_resolve_main(index, main_index),
                rules=rules,
                source=upload.filename or f"upload-{index}",
            )
        except Exception as exc:  # noqa: BLE001 - a bad file must not lose the set
            notes.append(f"{upload.filename}: could not be decoded ({type(exc).__name__})")
            continue
        per_image.append(results)
        all_hashes.append(hashes)
        details.append(detail)
        if tags is not None:
            kept_tags.append(tags[index] if index < len(tags) else None)

    if not per_image:
        raise HTTPException(status_code=422, detail={"reason": "no image decoded", "notes": notes})
    return _assemble(
        per_image,
        all_hashes,
        details,
        len(per_image),
        rules,
        notes,
        kept_tags if tags is not None else None,
    )


@app.post("/audit/urls")
def audit_urls(request: UrlAuditRequest) -> dict[str, Any]:
    """Audit images by URL, or by ASIN via a best-effort product-page read."""
    rules = load_rules()
    notes: list[str] = []
    main_index = request.main_index
    urls = list(request.image_urls or [])

    if request.asin and not urls:
        page = fetch_listing_image_urls(request.asin)
        if page.status == "blocked":
            # Not an error on our side, and not something to retry or work
            # around. Say so and point at the upload path.
            return {
                "status": "blocked",
                "reason": page.reason,
                "remedy": "upload the image files to /audit/upload instead",
            }
        if page.status not in ("ok", "partial"):
            raise HTTPException(status_code=422, detail={"reason": page.reason})

        urls = page.image_urls[:MAX_IMAGES_PER_AUDIT]
        if page.status == "ok":
            # The page's own gallery lists the MAIN variant first.
            main_index = 0 if main_index is None else main_index
        else:
            notes.append(page.reason)
            # Deliberately leave main_index unknown: these candidates are not
            # known to be product images, let alone which one is the main one.

    if not urls:
        raise HTTPException(status_code=422, detail="supply image_urls or an asin")
    if len(urls) > MAX_IMAGES_PER_AUDIT:
        raise HTTPException(
            status_code=422,
            detail=f"{len(urls)} URLs supplied; the cap is {MAX_IMAGES_PER_AUDIT}",
        )

    # Fetching is network-bound, so it is worth overlapping — but bounded, since
    # each in-flight body can be 8 MB.
    with ThreadPoolExecutor(max_workers=FETCH_CONCURRENCY) as pool:
        fetched: list[FetchResult] = list(pool.map(fetch_image, urls))

    per_image, all_hashes, details = [], [], []
    for index, result in enumerate(fetched):
        if not result.ok:
            notes.append(f"{result.url}: {result.status} — {result.reason}")
            continue
        assert result.data is not None
        try:
            results, hashes, detail = _audit_one(
                result.data,
                is_main=_resolve_main(index, main_index),
                rules=rules,
                source=result.url,
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{result.url}: could not be decoded ({type(exc).__name__})")
            continue
        per_image.append(results)
        all_hashes.append(hashes)
        details.append(detail)

    if not per_image:
        raise HTTPException(
            status_code=422, detail={"reason": "no image could be fetched", "notes": notes}
        )
    return _assemble(per_image, all_hashes, details, len(per_image), rules, notes)


@app.get("/audit/{audit_id}")
def get_audit(audit_id: str) -> dict[str, Any]:
    """Full per-check detail for the UI.

    A 404 here is ordinary, not exceptional: the store is a bounded in-memory
    cache on a host with no persistent disk, so a restart or an eviction loses
    it. Clients render the compact payload they already hold rather than
    treating this as a failure.
    """
    record = _AUDITS.get(audit_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail="audit not found — the store is an in-memory cache, not a record",
        )
    _AUDITS.move_to_end(audit_id)
    return record
