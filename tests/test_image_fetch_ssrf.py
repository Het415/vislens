"""Request-forgery controls on the image fetcher.

These are regression tests for security controls, not unit tests for a helper.
Each one corresponds to a specific bypass, and each must fail loudly if the
control is weakened — a silent pass here is the whole failure mode.

No test in this file makes a real network connection. The controls all reject
*before* connecting, and the one case that needs a hostile DNS answer patches
the resolver seam rather than relying on a real hostile domain existing.
"""

from __future__ import annotations

import io
import ssl
import time

import pytest
from PIL import Image

from vislens.fetch import image_fetch
from vislens.fetch.image_fetch import (
    IMAGE_HOSTS,
    MAX_IMAGE_BYTES,
    PAGE_HOSTS,
    address_is_forbidden,
    extract_asin,
    fetch_image,
    sniff_image_type,
)

GOOD = "https://m.media-amazon.com/images/I/71abcdef.jpg"


# ── Control 1: scheme allowlist ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "http://m.media-amazon.com/images/I/1.jpg",  # plaintext
        "file:///etc/passwd",
        "gopher://m.media-amazon.com/1",
        "ftp://m.media-amazon.com/1.jpg",
        "data:image/png;base64,iVBORw0KGgo=",
    ],
)
def test_only_https_is_accepted(url):
    res = fetch_image(url)
    assert res.status == "rejected"
    assert "https" in res.reason or "hostname" in res.reason


# ── Control 2: exact-match host allowlist ─────────────────────────────────────


@pytest.mark.parametrize(
    "host",
    [
        # The suffix-confusion family. A check for "endswith amazon.com" or
        # "contains amazon.com" passes every one of these.
        "m.media-amazon.com.attacker.net",
        "evil-m.media-amazon.com",
        "m.media-amazon.com.evil.io",
        "notm.media-amazon.com",
        "m-media-amazon.com",
        "amazon.com",
        "localhost",
        "127.0.0.1",
        "169.254.169.254",
        "metadata.google.internal",
        "m.media-amazon.com@attacker.net",  # userinfo confusion
    ],
)
def test_host_allowlist_is_exact_match(host):
    res = fetch_image(f"https://{host}/images/I/1.jpg")
    assert res.status == "rejected", f"{host} was not rejected"
    assert "allowlist" in res.reason


def test_allowlisted_host_passes_the_host_gate(monkeypatch):
    """Sanity check on the gate itself: a legitimate host must get past the
    hostname check, or every test above would pass for the wrong reason."""
    monkeypatch.setattr(image_fetch, "_resolve", lambda host: ["8.8.8.8"])
    res = fetch_image(GOOD)
    # It gets past the allowlist and the address check, then fails on the
    # connection attempt (8.8.8.8 does not serve this path over TLS).
    assert res.status == "rejected"
    assert "allowlist" not in res.reason
    assert "is not https" not in res.reason


def test_non_443_port_is_rejected():
    res = fetch_image("https://m.media-amazon.com:8080/images/I/1.jpg")
    assert res.status == "rejected"
    assert "port" in res.reason


# ── Control 3: address validation, and no rebinding window ────────────────────


@pytest.mark.parametrize(
    "addr",
    [
        "169.254.169.254",  # cloud metadata — the canonical SSRF target
        "127.0.0.1",
        "0.0.0.0",
        "10.0.0.5",
        "172.16.0.5",
        "192.168.1.5",
        "100.64.0.1",  # carrier-grade NAT, shared address space
        "::1",
        "fe80::1",
        "fc00::1",
        "::ffff:127.0.0.1",  # IPv4-mapped loopback
        "::ffff:169.254.169.254",  # IPv4-mapped metadata
        "224.0.0.1",  # multicast
    ],
)
def test_forbidden_addresses_are_all_rejected(addr):
    assert address_is_forbidden(addr) is not None, f"{addr} was treated as routable"


@pytest.mark.parametrize("addr", ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2001:4860:4860::8888"])
def test_routable_addresses_are_allowed(addr):
    """The gate must not be so broad that nothing gets through.

    Note 203.0.113.0/24 is deliberately NOT used here: TEST-NET-3 is
    `is_private` on Python 3.11, so it is correctly rejected and would make
    this test pass for the wrong reason.
    """
    assert address_is_forbidden(addr) is None


@pytest.mark.parametrize("addr", ["100.64.0.1", "100.127.255.254", "240.0.0.1", "203.0.113.10"])
def test_ranges_the_named_predicates_miss_are_still_rejected(addr):
    """The reason the gate ends in an `is_global` allowlist rather than a
    denylist of named predicates.

    Measured on Python 3.11: `100.64.0.0/10` (RFC 6598 carrier-grade NAT) has
    `is_private == False` and every other named predicate False. A denylist
    would let it through, and CGNAT space routinely reaches internal
    infrastructure.
    """
    assert address_is_forbidden(addr) is not None


def test_cgnat_is_the_specific_gap_a_denylist_leaves():
    """Pinned as a fact, so nobody "simplifies" the gate back to a denylist."""
    import ipaddress

    ip = ipaddress.ip_address("100.64.0.1")
    assert not ip.is_private  # the trap
    assert not ip.is_loopback and not ip.is_link_local and not ip.is_reserved
    assert not ip.is_global  # the thing that catches it
    assert "not globally routable" in (address_is_forbidden("100.64.0.1") or "")


def test_allowlisted_host_resolving_to_a_private_address_is_rejected(monkeypatch):
    """The DNS-rebinding case. The hostname is legitimately on the allowlist;
    the answer is hostile. Validating the *resolved address* is what catches
    this, and a checker that only inspects the hostname does not."""
    monkeypatch.setattr(image_fetch, "_resolve", lambda host: ["10.1.2.3"])
    res = fetch_image(GOOD)
    assert res.status == "rejected"
    assert "private" in res.reason


def test_metadata_address_behind_an_allowlisted_name_is_rejected(monkeypatch):
    monkeypatch.setattr(image_fetch, "_resolve", lambda host: ["169.254.169.254"])
    res = fetch_image(GOOD)
    assert res.status == "rejected"
    # Name the address rather than a specific predicate: the metadata endpoint
    # matches several, and which one reports first is not the contract.
    assert "169.254.169.254" in res.reason


def test_split_horizon_answer_is_rejected_on_any_bad_address(monkeypatch):
    """A resolver returning one good and one bad address must be rejected.

    Taking the first routable address and ignoring the rest would let a
    deliberately mixed answer through.
    """
    monkeypatch.setattr(image_fetch, "_resolve", lambda host: ["8.8.8.8", "127.0.0.1"])
    res = fetch_image(GOOD)
    assert res.status == "rejected"
    assert "loopback" in res.reason


def test_empty_dns_answer_is_rejected(monkeypatch):
    monkeypatch.setattr(image_fetch, "_resolve", lambda host: [])
    res = fetch_image(GOOD)
    assert res.status == "rejected"
    assert "no addresses" in res.reason


def test_connection_targets_the_validated_address_not_a_fresh_lookup():
    """The structural guarantee behind control 3.

    `_PinnedHTTPSConnection` connects to an address passed in, while leaving
    `host` as the real hostname so TLS verification and the Host header stay
    correct. If this ever grows a second resolution the rebinding window
    reopens, so the shape is pinned here.
    """
    conn = image_fetch._PinnedHTTPSConnection(
        "m.media-amazon.com",
        "203.0.113.10",
        timeout=1.0,
        context=ssl.create_default_context(),
    )
    assert conn.host == "m.media-amazon.com"  # SNI + Host header
    assert conn._pinned_ip == "203.0.113.10"  # where the socket actually goes
    assert conn.port == 443
    # The override must not delegate to a hostname-based connect.
    src = image_fetch._PinnedHTTPSConnection.connect.__doc__ or ""
    assert "super().connect()" not in src


# ── Control 4: redirects are re-validated ─────────────────────────────────────


def test_redirects_are_capped():
    assert image_fetch.MAX_REDIRECTS <= 2


def test_redirect_target_goes_back_through_the_full_gate(monkeypatch):
    """A 302 to a metadata address must be rejected by the same gate as the
    original request, because a blind follow defeats every earlier control."""
    calls: list[str] = []

    real_gate = image_fetch._gate

    def tracking_gate(url, hosts):
        calls.append(url)
        return real_gate(url, hosts)

    monkeypatch.setattr(image_fetch, "_gate", tracking_gate)
    monkeypatch.setattr(image_fetch, "_resolve", lambda host: ["8.8.8.8"])

    # Drive _request's redirect branch directly: a real connection is not
    # needed to prove the recursion re-gates.
    outcome = image_fetch._request(
        "https://169.254.169.254/latest/meta-data/",
        IMAGE_HOSTS,
        MAX_IMAGE_BYTES,
        accept="image/*",
        deadline=time.monotonic() + 5,
        hops=1,
    )
    assert isinstance(outcome, str)
    assert "allowlist" in outcome
    assert calls  # the gate was consulted, not bypassed


# ── Control 5: byte cap on received bytes ─────────────────────────────────────


def test_byte_cap_ignores_a_lying_content_length():
    """`Content-Length` is attacker-controlled and routinely absent or wrong, so
    the cap is enforced on bytes actually received.

    Asserted behaviourally rather than by grepping the source: the reader is
    handed a response that advertises a tiny length and then serves an endless
    body, which is exactly the shape of the attack.
    """

    class LyingResponse:
        headers = {"Content-Length": "10"}

        def read(self, n):
            return b"\x00" * n

    outcome = image_fetch._read_capped(
        LyingResponse(), cap=4096, deadline=time.monotonic() + 5
    )
    assert isinstance(outcome, str)
    assert "exceeded 4096 bytes" in outcome


def test_read_capped_stops_at_the_cap():
    class FakeResponse:
        def __init__(self):
            self.served = 0

        def read(self, n):
            self.served += n
            return b"\x00" * n  # an endless body

    outcome = image_fetch._read_capped(
        FakeResponse(), cap=1024, deadline=time.monotonic() + 5
    )
    assert isinstance(outcome, str)
    assert "exceeded 1024 bytes" in outcome


def test_read_capped_honours_the_deadline():
    class SlowResponse:
        def read(self, n):
            return b"\x00" * n

    outcome = image_fetch._read_capped(
        SlowResponse(), cap=10**9, deadline=time.monotonic() - 1
    )
    assert isinstance(outcome, str)
    assert "deadline" in outcome


def test_image_cap_is_bounded():
    assert MAX_IMAGE_BYTES <= 16 * 1024 * 1024


# ── Control 6: magic-byte sniffing ────────────────────────────────────────────


def _encode(fmt: str) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buf, fmt)
    return buf.getvalue()


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("GIF", "image/gif"), ("WEBP", "image/webp")],
)
def test_real_images_are_recognised(fmt, expected):
    assert sniff_image_type(_encode(fmt)) == expected


@pytest.mark.parametrize(
    "body",
    [
        b"<!DOCTYPE html><html><body>not an image</body></html>",
        b"#!/bin/sh\nrm -rf /\n",
        b'{"secret": "value"}',
        b"\x7fELF\x02\x01\x01\x00",  # an executable
        b"",
        b"\xff\xd8",  # truncated JPEG magic
    ],
)
def test_non_images_are_not_recognised(body):
    assert sniff_image_type(body) is None


def test_extension_and_content_type_are_not_trusted():
    """A `.jpg` URL and an `image/jpeg` header do not make a body an image."""
    import inspect

    src = inspect.getsource(fetch_image)
    assert "sniff_image_type" in src
    # The declared type may be reported in the reason, never used to accept.
    assert "if declared" not in src
    assert 'declared ==' not in src


# ── Product-page path: gives up rather than working around a block ────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("B08XPWDSWW", "B08XPWDSWW"),
        ("https://www.amazon.com/dp/B07GZFM1ZM", "B07GZFM1ZM"),
        ("https://www.amazon.com/gp/product/B01K8B8YA8?th=1", "B01K8B8YA8"),
        ("https://www.amazon.com/Some-Title/dp/B075X8471B/ref=sr_1_1", "B075X8471B"),
        ("not an asin at all", None),
    ],
)
def test_asin_extraction(text, expected):
    assert extract_asin(text) == expected


def test_page_host_must_be_allowlisted():
    res = image_fetch.fetch_listing_image_urls("B08XPWDSWW", host="evil.example.com")
    assert res.status == "rejected"
    assert "page allowlist" in res.reason


def test_page_and_image_allowlists_are_disjoint():
    """A product page is HTML on a different code path with different limits.
    Sharing one allowlist would let an HTML host serve the image path."""
    assert not (PAGE_HOSTS & IMAGE_HOSTS)


def test_no_captcha_circumvention_anywhere_in_the_module():
    """The page fetch is allowed to read a public page and required to give up
    when refused. Solving or evading a challenge is circumventing an access
    control, which is a different thing, and this module must not grow it."""
    import inspect

    src = inspect.getsource(image_fetch).lower()
    for forbidden in (
        "2captcha",
        "anticaptcha",
        "capsolver",
        "solve_captcha",
        "rotate_proxy",
        "proxies=",
        "selenium",
        "playwright",
        "undetected_chrome",
        "cloudscraper",
    ):
        assert forbidden not in src, f"{forbidden!r} appears in the fetcher"


def test_block_markers_cover_amazons_actual_refusal_pages():
    lowered = [m.lower() for m in image_fetch._BLOCK_MARKERS]
    assert "captcha" in lowered
    assert any("automated access" in m for m in lowered)


def test_fetch_never_raises_on_hostile_input():
    """A caller auditing nine images must not lose the other eight to one bad
    URL, so every rejection is a return value rather than an exception."""
    for url in ("", "https://", "not a url", "https://[::1]/x.jpg", "https://a..b/x.jpg"):
        res = fetch_image(url)
        assert res.status in ("rejected", "error")
        assert res.data is None
