"""Audit service behaviour.

No real network: the URL paths patch the fetcher seam. The upload path needs
nothing patched, which is itself the point — it is the route with no network and
no ToS surface, and it is the one the UI defaults to.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vislens.fetch.image_fetch import FetchResult
from vislens.rules.image_rules import load_rules
from vislens.service.app import AUDIT_STORE_MAX, MAX_IMAGES_PER_AUDIT, app

SIZE = 1000


@pytest.fixture
def client() -> TestClient:
    import vislens.service.app as service_module

    service_module._AUDITS.clear()
    return TestClient(app)


def png_bytes(side: int = SIZE, product: int = 950, bg=(255, 255, 255), seed: int = 0) -> bytes:
    """A compliant synthetic product image.

    With `seed`, the product is a coarse random texture rather than a solid
    block. That matters and the first attempt got it wrong: varying only a
    fine interior detail leaves images *identical at dHash's working
    resolution* (a 9x8 thumbnail), so six "different" fixtures were correctly
    clustered as one duplicate set. A stand-in for six different photographs
    has to differ in coarse layout, not in texture detail.

    `seed=0` keeps a solid block, because the occupancy tests depend on an
    exactly computable bounding box.
    """
    a = np.full((side, side, 3), bg, dtype=np.uint8)
    off = (side - product) // 2
    if seed:
        rng = np.random.default_rng(seed)
        block = rng.integers(0, 200, size=(8, 8, 3), dtype=np.uint8)
        tex = Image.fromarray(block).resize((product, product), Image.NEAREST)
        a[off : off + product, off : off + product] = np.asarray(tex)
    else:
        a[off : off + product, off : off + product] = 0
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    return buf.getvalue()


def compliant_set(n: int = 6) -> list[bytes]:
    """A set that passes every check, count included — and is genuinely varied.

    Six is not arbitrary: `image_count` recommends 6+ and is a `rule_exact`
    check, so it legitimately drives the headline. A single compliant image
    still warns — correctly — which is why a "clean pass" test needs a set.
    """
    return [png_bytes(seed=i + 1) for i in range(n)]


def upload(client: TestClient, blobs: list[bytes], **params):
    files = [("files", (f"img{i}.png", b, "image/png")) for i, b in enumerate(blobs)]
    return client.post("/audit/upload", files=files, params=params)


# ── healthz ───────────────────────────────────────────────────────────────────


def test_healthz_reports_the_rules_version_it_is_serving(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["rules_version"] == load_rules()["rules_version"]
    assert body["rules_verified_on"]


def test_healthz_does_not_advertise_disabled_checks_as_capabilities(client):
    """A check that is specified but disabled is not a capability. Advertising
    one is how the predecessor ended up with a landing page describing a CLIP
    model that did not exist."""
    body = client.get("/healthz").json()
    enabled = set(body["checks_enabled"])
    pending = set(body["checks_specified_not_enabled"])

    assert not (enabled & pending)
    assert "white_background" in enabled
    assert {"off_product_text", "image_role", "prohibited_content"} <= pending


def test_healthz_publishes_which_rules_are_main_image_only(client):
    body = client.get("/healthz").json()
    assert set(body["main_image_only"]) == {
        "white_background",
        "frame_occupancy",
        "background_artifacts",
    }


# ── upload path ───────────────────────────────────────────────────────────────


def test_upload_audits_a_compliant_set(client):
    body = upload(client, compliant_set(), main_index=0).json()
    assert body["headline"] == "pass"
    assert body["duplicates"]["clusters"] == []
    assert body["n_images"] == 6
    assert body["audit_id"]
    assert body["legend"] == {}  # nothing breached, so nothing in the legend


def test_a_single_compliant_image_still_warns_on_the_count(client):
    """The set-level count check is a real finding and does reach the headline:
    one image is compliant in itself and an incomplete listing."""
    body = upload(client, [png_bytes()], main_index=0).json()
    assert body["headline"] == "warn"
    assert body["set_checks"][0][0] == "cnt"
    # ...and the image itself is clean, which the group must show.
    assert body["groups"][0]["f"] == []


def test_upload_flags_a_non_compliant_main_image(client):
    # 400px product on a 1000px canvas -> 16% bbox occupancy.
    body = upload(client, [png_bytes(product=400)], main_index=0).json()
    assert body["headline"] == "fail"
    codes = {f[0] for g in body["groups"] for f in g["f"]}
    assert "occ" in codes


def test_main_image_rules_do_not_condemn_a_secondary_image(client):
    """The correctness bug this guards: a lifestyle photo with a coloured
    background and a small product is compliant as a secondary image, and
    failing it would be a verdict against a rule that does not govern it."""
    images = [png_bytes(seed=1)] + [
        png_bytes(product=300, bg=(180, 140, 90), seed=i + 10) for i in range(5)
    ]
    body = upload(client, images, main_index=0).json()

    assert body["headline"] == "pass"
    # The lifestyle measurements are still reported — just as measurements,
    # in `a`, never as verdicts in `f`.
    verdicts = {f[0] for g in body["groups"] for f in g["f"]}
    assert not (verdicts & {"wbg", "occ", "art"})
    # The secondary-image measurements are counted, never enumerated as
    # findings the agent could mistake for violations.
    assert sum(g.get("n_measured_only", 0) for g in body["groups"]) > 0
    assert all("a" not in g for g in body["groups"])
    assert body["main_index"] == 0


def test_unknown_main_index_does_not_claim_a_verdict(client):
    """Without a declared main image the main-only rules are measured, not
    claimed — so a set that would fail as a main image does not report `fail`."""
    body = upload(client, [png_bytes(product=400)]).json()
    assert body["headline"] != "fail"


def test_image_count_is_a_set_level_finding(client):
    body = upload(client, [png_bytes()], main_index=0).json()
    assert body["set_checks"], "one image should trip the count check"
    assert body["set_checks"][0][0] == "cnt"


def test_a_single_undecodable_file_does_not_lose_the_set(client):
    """A caller auditing nine images must not lose eight to one bad file."""
    good = png_bytes()
    response = upload(client, [good, b"this is not an image at all", good], main_index=0)
    body = response.json()

    assert response.status_code == 200
    assert body["n_images"] == 2
    assert any("could not be decoded" in note for note in body["notes"])


def test_all_files_undecodable_is_a_422_with_the_reasons(client):
    response = upload(client, [b"nope", b"also nope"])
    assert response.status_code == 422
    assert response.json()["detail"]["notes"]


def test_too_many_images_is_rejected(client):
    response = upload(client, [png_bytes(side=520, product=500)] * (MAX_IMAGES_PER_AUDIT + 1))
    assert response.status_code == 422
    assert str(MAX_IMAGES_PER_AUDIT) in response.json()["detail"]


def test_no_files_is_rejected(client):
    assert client.post("/audit/upload", files=[]).status_code == 422


# ── URL path ──────────────────────────────────────────────────────────────────


def test_url_audit_uses_the_hardened_fetcher(client, monkeypatch):
    import vislens.service.app as service_module

    seen: list[str] = []

    def fake_fetch(url: str) -> FetchResult:
        seen.append(url)
        # Distinct per URL: identical images would be flagged as duplicates,
        # which is correct behaviour but not what this test is about.
        blob = png_bytes(seed=len(seen))
        return FetchResult(url=url, status="ok", data=blob, content_type="image/png")

    monkeypatch.setattr(service_module, "fetch_image", fake_fetch)
    urls = [f"https://m.media-amazon.com/images/I/{i}.jpg" for i in range(6)]
    body = client.post("/audit/urls", json={"image_urls": urls, "main_index": 0}).json()

    assert seen == urls
    assert body["headline"] == "pass"
    assert body["duplicates"]["clusters"] == []


def test_a_rejected_url_is_reported_not_swallowed(client, monkeypatch):
    import vislens.service.app as service_module

    def fake_fetch(url: str) -> FetchResult:
        if "bad" in url:
            return FetchResult(url=url, status="rejected", reason="host not in the allowlist")
        return FetchResult(url=url, status="ok", data=png_bytes(), content_type="image/png")

    monkeypatch.setattr(service_module, "fetch_image", fake_fetch)
    body = client.post(
        "/audit/urls",
        json={
            "image_urls": [
                "https://m.media-amazon.com/images/I/good.jpg",
                "https://evil.example.com/bad.jpg",
            ],
            "main_index": 0,
        },
    ).json()

    assert body["n_images"] == 1
    assert any("allowlist" in note for note in body["notes"])


def test_a_blocked_product_page_asks_for_uploads_instead(client, monkeypatch):
    """Amazon refusing a datacenter client is an expected outcome, not an
    error, and the answer is never to retry or work around it."""
    import vislens.service.app as service_module
    from vislens.fetch.image_fetch import PageResult

    monkeypatch.setattr(
        service_module,
        "fetch_listing_image_urls",
        lambda asin: PageResult(source=asin, status="blocked", reason="bot challenge served"),
    )
    body = client.post("/audit/urls", json={"asin": "B08XPWDSWW"}).json()

    assert body["status"] == "blocked"
    assert "upload" in body["remedy"]


def test_partial_page_results_do_not_assert_a_main_image(client, monkeypatch):
    """When the page withholds its gallery, the candidates are not known to be
    product images — so no main image is assumed and no verdict is claimed."""
    import vislens.service.app as service_module
    from vislens.fetch.image_fetch import PageResult

    monkeypatch.setattr(
        service_module,
        "fetch_listing_image_urls",
        lambda asin: PageResult(
            source=asin,
            status="partial",
            image_urls=["https://m.media-amazon.com/images/I/a.jpg"],
            reason="the page did not include its product-image gallery",
        ),
    )
    monkeypatch.setattr(
        service_module,
        "fetch_image",
        lambda url: FetchResult(
            url=url, status="ok", data=png_bytes(product=400), content_type="image/png"
        ),
    )
    body = client.post("/audit/urls", json={"asin": "B08XPWDSWW"}).json()

    assert body["headline"] != "fail"
    assert any("gallery" in note for note in body["notes"])


def test_urls_and_asin_both_absent_is_rejected(client):
    assert client.post("/audit/urls", json={}).status_code == 422


# ── detail store ──────────────────────────────────────────────────────────────


def test_detail_endpoint_returns_the_full_per_check_record(client):
    audit_id = upload(client, [png_bytes(product=400)], main_index=0).json()["audit_id"]
    record = client.get(f"/audit/{audit_id}").json()

    assert len(record["images"]) == 1
    image = record["images"][0]
    assert image["orig_width"] == SIZE
    assert image["sha256"]
    occ = next(c for c in image["checks"] if c["check_id"] == "frame_occupancy")
    # Detail the compact agent payload deliberately omits.
    assert "silhouette_occupancy" in occ["detail"]
    assert "bbox" in occ["detail"]


def test_no_image_bytes_are_ever_stored(client):
    """With no auth and no tenancy, anything retained is effectively public.
    Hashes and measurements are kept; pixels are not."""
    blob = png_bytes()
    audit_id = upload(client, [blob], main_index=0).json()["audit_id"]
    record = client.get(f"/audit/{audit_id}").json()

    serialised = repr(record).encode()
    assert blob[:64] not in serialised
    assert b"data" not in serialised or b"iVBOR" not in serialised
    assert "sha256" in record["images"][0]


def test_missing_audit_is_an_ordinary_404(client):
    response = client.get("/audit/deadbeef1234")
    assert response.status_code == 404
    assert "cache" in response.json()["detail"]


def test_audit_store_is_bounded(client):
    import vislens.service.app as service_module

    blob = png_bytes(side=520, product=500)
    first = upload(client, [blob], main_index=0).json()["audit_id"]
    for _ in range(AUDIT_STORE_MAX):
        upload(client, [blob], main_index=0)

    assert len(service_module._AUDITS) <= AUDIT_STORE_MAX
    # The oldest entry was evicted rather than the map growing without bound.
    assert client.get(f"/audit/{first}").status_code == 404


# ── What the service does not hold ────────────────────────────────────────────


def test_service_imports_no_llm_client_and_no_torch():
    """The isolation argument for the split: this process dereferences
    user-supplied URLs, so it must contain nothing worth reaching.

    Checked in a SUBPROCESS, and that is the whole point of the test rather
    than an implementation detail. The first version inspected `sys.modules`
    in-process, which is polluted by every other test module in the session —
    it broke the moment `test_catalog_build.py` imported duckdb, and until then
    it had been passing for the wrong reason. A global-state assertion about
    an import graph has to run in a process that imported only the thing under
    test, or it measures the test runner instead of the service.
    """
    import subprocess
    import sys

    forbidden = ("torch", "groq", "anthropic", "openai", "langchain", "duckdb", "pandas")
    probe = (
        "import sys, vislens.service.app;"
        f"bad=[m for m in {forbidden!r} if m in sys.modules];"
        "print(','.join(bad))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    leaked = result.stdout.strip()
    assert not leaked, f"the service's import graph pulls in: {leaked}"


# ── Near-duplicate detection ──────────────────────────────────────────────────


def test_duplicate_images_are_found_and_warn(client):
    """Two copies of the same photo in one listing is a real quality problem,
    so it reaches the headline — as a `warn`, because the detector's measured
    precision of 1.000 comes from a deliberately small held-out set and does
    not support a rejection-grade claim."""
    dup = png_bytes(seed=1)
    images = [dup, dup] + [png_bytes(seed=i + 2) for i in range(4)]
    body = upload(client, images, main_index=0).json()

    assert body["duplicates"]["clusters"] == [[0, 1]]
    assert body["headline"] == "warn"


def test_duplicate_finding_names_its_calibration(client):
    """A claim has to be attributable to a specific measurement, not float
    free of one."""
    body = upload(client, compliant_set(), main_index=0).json()
    dup = body["duplicates"]
    assert dup["method"] == "dhash"
    assert dup["threshold"] == 8
    assert "unavailable" not in dup


def test_recompressed_copy_is_caught_not_just_byte_identical(client):
    """The point of a perceptual hash: a re-encoded repost is still a
    duplicate even though its bytes differ entirely."""
    original = png_bytes(seed=7)
    with Image.open(io.BytesIO(original)) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=45)
    recompressed = buf.getvalue()

    assert recompressed != original
    body = upload(client, [original, recompressed] + compliant_set(4), main_index=0).json()
    assert [0, 1] in body["duplicates"]["clusters"]


def test_sku_group_mismatch_is_reported(client):
    """An image tagged to one SKU whose nearest neighbour is another SKU's."""
    a = png_bytes(seed=11)
    with Image.open(io.BytesIO(a)) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=60)
    a_copy = buf.getvalue()

    body = upload(
        client,
        [a, png_bytes(seed=12), a_copy],
        main_index=0,
        sku_groups="SKU-A,SKU-B,SKU-B",
    ).json()

    mismatches = body["duplicates"].get("group_mismatches", [])
    assert any(
        m["i"] == 2 and m["tagged"] == "SKU-B" and m["nearest"] == "SKU-A"
        for m in mismatches
    )


def test_no_sku_tags_means_no_mismatch_claim(client):
    body = upload(client, compliant_set(), main_index=0).json()
    assert "group_mismatches" not in body["duplicates"]


def test_healthz_publishes_the_dedup_calibration(client):
    """The detector's provenance is part of its health, not a footnote."""
    nd = client.get("/healthz").json()["near_duplicate"]
    assert nd["calibrated"] is True
    assert nd["method"] == "dhash"
    assert nd["report"].startswith("eval/reports/")
    assert nd["measured"]["held_out_precision"] == 1.0


def test_detail_endpoint_carries_hashes_and_pair_distances(client):
    dup = png_bytes(seed=21)
    audit_id = upload(client, [dup, dup, png_bytes(seed=22)], main_index=0).json()["audit_id"]
    record = client.get(f"/audit/{audit_id}").json()

    assert record["images"][0]["hashes"]["dhash"]
    assert record["images"][0]["hashes"]["resample"] == "LANCZOS"
    pairs = record["duplicates"]["pairs"]
    assert pairs and pairs[0]["distance"] == 0
    assert pairs[0]["method"] == "dhash"


def test_cors_allows_the_frontend_origin(client):
    """The browser posts uploads straight here rather than through ListingLens:
    proxying image bytes through that process would put it back on the image
    path it was split off from."""
    response = client.options(
        "/audit/upload",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code in (200, 204)
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_does_not_allow_credentials(client):
    """No auth and no cookies here, so there is nothing for a credentialed
    cross-origin request to carry — and keeping it off means a permissive
    origin list cannot be escalated into reading an authenticated response."""
    response = client.options(
        "/audit/upload",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-credentials" not in {
        k.lower() for k in response.headers
    }


def test_cors_rejects_an_unlisted_origin(client):
    response = client.options(
        "/audit/upload",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers.get("access-control-allow-origin") != "https://evil.example.com"
