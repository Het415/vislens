"""Fetching user-supplied images, hardened against request forgery.

This module is the only code in the project that dereferences a URL a user
handed us, which makes it the only code that can be turned into a request
forgery primitive against the host's internal network and cloud metadata
endpoints. Everything here is a security control, not input validation, and the
distinction matters: input validation rejects malformed input, whereas these
controls have to hold against input that is *well-formed and hostile*.

The policy constants below are deliberately module-level rather than loaded from
`rules_v1.json`. Image thresholds are tunable configuration that a reviewer
should be able to adjust in a dated diff; an allowlist is not, and putting it in
a data file invites someone to widen it without review. `tests/
test_image_fetch_ssrf.py` asserts each control independently.

Six controls, each closing a specific bypass:

1.  **Scheme allowlist** — `https` only. `file://`, `gopher://` and friends are
    the classic exfiltration schemes.
2.  **Exact-match host allowlist.** Suffix matching is the bug: a check for
    "ends with amazon.com" passes `evil.amazon.com.attacker.net`, and a check
    for "contains" passes almost anything.
3.  **Resolve DNS ourselves, validate every returned address, then connect to
    the validated address with SNI and Host pinned to the original name.**
    Resolving a name, checking the address, and *then* handing the name to an
    HTTP library is a DNS-rebinding TOCTOU: the second lookup can return a
    different address. This is the control most often implemented incorrectly.
4.  **Redirects are re-validated, not followed blindly.** A 302 to
    `http://169.254.169.254/` defeats every check above, so each hop goes back
    through the full gate.
5.  **Byte cap enforced while streaming**, never from `Content-Length`, which is
    attacker-controlled and routinely lies.
6.  **Magic-byte sniffing**, not the file extension and not `Content-Type`.

Amazon product-page fetching is separate, best-effort, and explicitly gives up
rather than working around a block — see `fetch_listing_image_urls`.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import re
import socket
import ssl
import time
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlparse

Status = Literal["ok", "partial", "blocked", "rejected", "error"]

# ── Policy ────────────────────────────────────────────────────────────────────

# Exact hostnames only. Amazon's image CDN is a small, well-known set, which is
# what makes an allowlist viable here rather than merely aspirational.
IMAGE_HOSTS: frozenset[str] = frozenset(
    {
        "m.media-amazon.com",
        "images-na.ssl-images-amazon.com",
        "images-eu.ssl-images-amazon.com",
        "images-fe.ssl-images-amazon.com",
        "images-cn.ssl-images-amazon.com",
    }
)

# Separate, and deliberately narrower: a product page is HTML, so it goes down a
# different code path with different limits and no magic-byte check.
PAGE_HOSTS: frozenset[str] = frozenset(
    {"www.amazon.com", "www.amazon.co.uk", "www.amazon.de", "www.amazon.ca"}
)

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_PAGE_BYTES = 3 * 1024 * 1024
CONNECT_TIMEOUT_S = 3.0
READ_TIMEOUT_S = 10.0
TOTAL_DEADLINE_S = 20.0
MAX_REDIRECTS = 2
READ_CHUNK = 64 * 1024

# Magic bytes, checked against the decoded prefix. Extension and Content-Type
# are both attacker-controlled; the first bytes of the body are not.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)

_ASIN_RE = re.compile(r"\b([A-Z0-9]{10})\b")

# Amazon encodes a requested render size as a token between the image id and the
# extension: `.../I/51eGpMA9lML._AC_SS115_.jpg` is a 115px thumbnail of
# `51eGpMA9lML`. Stripping the token yields the ORIGINAL uploaded asset.
#
# Normalising to the bare URL is not a nicety, it is a correctness requirement.
# A product page hands out 115/230/345px variants; auditing one of those and
# reporting "fails the 1000px zoom requirement" would be a false verdict about
# an image that may well be 1500px in the seller's actual listing. Measured on
# B08XPWDSWW: every `data-a-dynamic-image` entry was <=345px.
_VARIANT_RE = re.compile(
    r"^(https://[^/]+/images/I/[A-Za-z0-9+%-]+)(\._[A-Za-z0-9_,]+_)?(\.(?:jpg|jpeg|png))$",
    re.IGNORECASE,
)

# The authoritative product-image list, when the page includes it.
_COLOR_IMAGES_RE = re.compile(
    r"'colorImages'\s*:\s*\{\s*'initial'\s*:\s*A\.\$\.parseJSON\('(\[.*?\])'\)",
    re.DOTALL,
)
_CDN_IMAGE_RE = re.compile(
    r"https://(?:m\.media-amazon\.com|images-[a-z]{2}\.ssl-images-amazon\.com)"
    r"/images/I/[A-Za-z0-9._%+-]+\.(?:jpg|jpeg|png)"
)
# Markers that mean "Amazon is refusing us", not "the page has no images".
_BLOCK_MARKERS = (
    "captcha",
    "api-services-support@amazon.com",
    "to discuss automated access",
    "type the characters you see in this image",
    "enter the characters you see below",
    "robot check",
)


@dataclass(frozen=True)
class FetchResult:
    url: str
    status: Status
    data: bytes | None = None
    reason: str = ""
    content_type: str | None = None
    n_bytes: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class PageResult:
    source: str
    status: Status
    image_urls: list[str] = field(default_factory=list)
    reason: str = ""


# ── Address validation ────────────────────────────────────────────────────────


def _resolve(host: str) -> list[str]:
    """Resolve a hostname to addresses.

    A module-level seam on purpose: the SSRF tests need to simulate a hostile
    DNS answer (an allowlisted name resolving to a private address), and the
    only honest way to test that is to control resolution.
    """
    infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    seen: list[str] = []
    for info in infos:
        addr = info[4][0]
        if addr not in seen:
            seen.append(addr)
    return seen


def address_is_forbidden(addr: str) -> str | None:
    """Return a rejection reason for an address, or None when it is routable.

    Covers the cloud metadata endpoint (169.254.169.254), and unwraps
    IPv4-mapped IPv6 (`::ffff:127.0.0.1`) before testing — without that unwrap,
    a mapped loopback address passes every IPv6 predicate.

    The decisive check is `is_global`, treated as an allowlist of routable
    space. A denylist of named predicates is not sufficient: see the comment
    below for the carrier-grade-NAT range that slips through one.
    """
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return f"unparseable address {addr!r}"

    if isinstance(ip, ipaddress.IPv6Address):
        mapped = ip.ipv4_mapped
        if mapped is not None:
            ip = mapped

    # Named predicates first, purely so the rejection reason is specific.
    for attr in (
        "is_loopback",
        "is_link_local",
        "is_multicast",
        "is_unspecified",
        "is_reserved",
        "is_private",
    ):
        if getattr(ip, attr, False):
            return f"address {ip} is {attr.removeprefix('is_')}"

    # Then the control that actually closes the set: only globally routable
    # space is allowed. This is an allowlist, not a denylist, which is why it
    # catches ranges the named predicates miss. Measured on Python 3.11:
    # 100.64.0.0/10 (RFC 6598 carrier-grade NAT) has is_private == False and
    # every other predicate False, so a denylist of named predicates lets it
    # through — and CGNAT space routinely reaches internal infrastructure.
    # is_global is False for it, and for every other special-purpose range.
    if not getattr(ip, "is_global", False):
        return f"address {ip} is not globally routable"
    return None


def _gate(url: str, allowed_hosts: frozenset[str]) -> tuple[str, str, str] | str:
    """Validate a URL and resolve it. Returns (host, ip, path) or a reason string."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        return f"scheme {parsed.scheme!r} is not https"

    host = (parsed.hostname or "").lower()
    if not host:
        return "no hostname"
    # Exact match. Never endswith, never `in` — see the module docstring.
    if host not in allowed_hosts:
        return f"host {host!r} is not in the allowlist"
    if parsed.port not in (None, 443):
        return f"port {parsed.port} is not 443"

    try:
        addrs = _resolve(host)
    except OSError as exc:
        return f"DNS resolution failed for {host}: {exc}"
    if not addrs:
        return f"DNS returned no addresses for {host}"

    # Every returned address must be routable. Accepting the first good one and
    # ignoring a bad one would let a split-horizon answer through.
    for addr in addrs:
        bad = address_is_forbidden(addr)
        if bad is not None:
            return bad

    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return host, addrs[0], path


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS to a pre-validated address, with SNI and Host on the real name.

    This is the piece that closes the DNS-rebinding window. `self.host` stays
    the hostname, so certificate verification and the `Host` header are both
    correct, while the TCP connection goes to the address we already validated —
    so there is no second lookup for an attacker to race.
    """

    def __init__(self, host: str, ip: str, *, timeout: float, context: ssl.SSLContext):
        super().__init__(host, port=443, timeout=timeout, context=context)
        self._pinned_ip = ip

    def connect(self) -> None:  # pragma: no cover - exercised via fetch_image
        self.sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)
        if self._tunnel_host:
            self._tunnel()
        self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host)


def _read_capped(response: http.client.HTTPResponse, cap: int, deadline: float) -> bytes | str:
    """Stream the body, enforcing the cap as we go. Returns bytes or a reason."""
    chunks: list[bytes] = []
    total = 0
    while True:
        if time.monotonic() > deadline:
            return "total deadline exceeded while reading"
        chunk = response.read(READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > cap:
            # Enforced on bytes actually received, not on Content-Length.
            return f"body exceeded {cap} bytes"
        chunks.append(chunk)
    return b"".join(chunks)


def _request(
    url: str,
    allowed_hosts: frozenset[str],
    cap: int,
    *,
    accept: str,
    deadline: float,
    hops: int = 0,
) -> tuple[bytes, str | None] | str:
    """One validated GET, re-validating any redirect. Returns (body, ctype) or a reason."""
    gated = _gate(url, allowed_hosts)
    if isinstance(gated, str):
        return gated
    host, ip, path = gated

    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED

    conn = _PinnedHTTPSConnection(host, ip, timeout=CONNECT_TIMEOUT_S, context=ctx)
    try:
        conn.request(
            "GET",
            path,
            headers={
                "Accept": accept,
                "Accept-Encoding": "identity",
                "User-Agent": "vislens-image-audit/0.1 (+listinglens)",
                "Connection": "close",
            },
        )
        conn.sock.settimeout(READ_TIMEOUT_S)
        response = conn.getresponse()

        if response.status in (301, 302, 303, 307, 308):
            location = response.headers.get("Location", "")
            if hops >= MAX_REDIRECTS:
                return f"too many redirects (>{MAX_REDIRECTS})"
            if not location:
                return f"redirect {response.status} with no Location"
            # Full re-validation of the new target. A 302 to a metadata address
            # is the whole reason this cannot be a blind follow.
            return _request(
                location,
                allowed_hosts,
                cap,
                accept=accept,
                deadline=deadline,
                hops=hops + 1,
            )

        if response.status != 200:
            return f"HTTP {response.status}"

        body = _read_capped(response, cap, deadline)
        if isinstance(body, str):
            return body
        return body, response.headers.get("Content-Type")
    except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
        return f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()


# ── Public API ────────────────────────────────────────────────────────────────


def sniff_image_type(data: bytes) -> str | None:
    """The image type implied by the leading bytes, or None."""
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def fetch_image(url: str) -> FetchResult:
    """Fetch one image from an allowlisted Amazon CDN host.

    Never raises for a hostile or malformed URL: a rejection is a result, so a
    caller auditing nine images does not lose the other eight to one bad input.
    """
    deadline = time.monotonic() + TOTAL_DEADLINE_S
    outcome = _request(
        url, IMAGE_HOSTS, MAX_IMAGE_BYTES, accept="image/*", deadline=deadline
    )
    if isinstance(outcome, str):
        return FetchResult(url=url, status="rejected", reason=outcome)

    body, declared = outcome
    sniffed = sniff_image_type(body)
    if sniffed is None:
        # The body is not an image whatever the server called it. Reporting the
        # declared type is useful; trusting it is not.
        return FetchResult(
            url=url,
            status="rejected",
            reason=f"body is not a recognised image (server declared {declared!r})",
            n_bytes=len(body),
        )

    return FetchResult(
        url=url, status="ok", data=body, content_type=sniffed, n_bytes=len(body)
    )


def full_resolution_url(url: str) -> str:
    """Strip Amazon's size-variant token, yielding the original uploaded asset.

    See `_VARIANT_RE`: auditing a 115px thumbnail and reporting a resolution
    failure would be a false verdict about the seller's real image.
    """
    matched = _VARIANT_RE.match(url)
    return f"{matched.group(1)}{matched.group(3)}" if matched else url


def _gallery_from_color_images(html: str) -> list[str]:
    """The authoritative product-image set, if the page actually carries it.

    Returns [] when the blob is absent or is the placeholder Amazon serves to
    non-browser clients. Measured on B08XPWDSWW fetched without a browser user
    agent: `colorImages.initial` contained exactly one entry, a 40x60 GIF
    (`01RmK+J4pJL`) with `hiRes: null` — the "image loading" placeholder. So the
    absence of a real gallery has to be detected rather than assumed away.
    """
    found = _COLOR_IMAGES_RE.search(html)
    if not found:
        return []
    try:
        entries = json.loads(found.group(1).replace("\\'", "'"))
    except (ValueError, TypeError):
        return []

    urls: list[str] = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        for key in ("hiRes", "large"):
            candidate = entry.get(key)
            if not isinstance(candidate, str):
                continue
            # The placeholder is a GIF; real product assets are jpg/png.
            if not candidate.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            normalised = full_resolution_url(candidate)
            if normalised not in urls:
                urls.append(normalised)
            break
    return urls


def extract_asin(url_or_asin: str) -> str | None:
    """Pull a 10-character ASIN out of a bare id or an Amazon URL."""
    text = url_or_asin.strip()
    if re.fullmatch(r"[A-Z0-9]{10}", text):
        return text
    for pattern in (r"/dp/([A-Z0-9]{10})", r"/gp/product/([A-Z0-9]{10})", r"/ASIN/([A-Z0-9]{10})"):
        found = re.search(pattern, text)
        if found:
            return found.group(1)
    found = _ASIN_RE.search(text)
    return found.group(1) if found else None


def fetch_listing_image_urls(url_or_asin: str, host: str = "www.amazon.com") -> PageResult:
    """Best-effort: read a product page and pull its CDN image URLs out.

    Explicitly best-effort, and explicitly *not* worked around when it fails.
    Amazon serves a bot challenge to datacenter addresses most of the time, so
    from a hosted environment this path is expected to return `blocked` more
    often than not. That is reported as a result, and the caller's answer is to
    ask the user to upload the files instead.

    **No challenge is solved, evaded, or retried.** A detected block returns
    immediately with `status="blocked"`. Bypassing a bot challenge is
    circumventing an access control, which is a different thing from reading a
    public page, and this function does not do it.
    """
    asin = extract_asin(url_or_asin)
    if asin is None:
        return PageResult(source=url_or_asin, status="rejected", reason="no ASIN found in input")
    if host not in PAGE_HOSTS:
        return PageResult(
            source=asin, status="rejected", reason=f"host {host!r} is not in the page allowlist"
        )

    deadline = time.monotonic() + TOTAL_DEADLINE_S
    outcome = _request(
        f"https://{host}/dp/{asin}",
        PAGE_HOSTS,
        MAX_PAGE_BYTES,
        accept="text/html",
        deadline=deadline,
    )
    if isinstance(outcome, str):
        # A 503 from this path is overwhelmingly a block, not an outage.
        blocked = "HTTP 503" in outcome or "HTTP 429" in outcome
        return PageResult(
            source=asin,
            status="blocked" if blocked else "error",
            reason=outcome,
        )

    body, _ = outcome
    html = body.decode("utf-8", errors="replace")
    lowered = html.lower()
    if any(marker in lowered for marker in _BLOCK_MARKERS):
        return PageResult(
            source=asin,
            status="blocked",
            reason="Amazon served a bot challenge; upload the images instead",
        )

    # Authoritative path: the page's own gallery blob.
    gallery = _gallery_from_color_images(html)
    if gallery:
        return PageResult(source=asin, status="ok", image_urls=gallery)

    # Fallback: every CDN image on the page, normalised to full resolution and
    # deduplicated by image id. This is genuinely ambiguous output — the page
    # mixes product photos with site chrome, and one candidate on B08XPWDSWW
    # was a 650x45 banner — so it is reported as `partial`, never as `ok`.
    # The caller's correct response is to tell the user to upload instead.
    urls: list[str] = []
    for match in _CDN_IMAGE_RE.finditer(html):
        normalised = full_resolution_url(match.group(0))
        if normalised not in urls:
            urls.append(normalised)

    if not urls:
        return PageResult(
            source=asin, status="error", reason="no CDN image URLs found in the page"
        )
    return PageResult(
        source=asin,
        status="partial",
        image_urls=urls,
        reason=(
            "the page did not include its product-image gallery, so these are "
            "candidate CDN images and may include site graphics — upload the "
            "files for a reliable audit"
        ),
    )
