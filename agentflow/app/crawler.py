"""Safe, small web crawler used by the agent's web tool."""
from __future__ import annotations

import ipaddress
import re
import socket
from collections import deque
from dataclasses import dataclass
from urllib.parse import parse_qs, urldefrag, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

USER_AGENT = "AgentFlowBot/1.0 (research agent; +https://example.invalid)"
MAX_BYTES = 1_500_000
MAX_CHARS = 5000
URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")


class UnsafeURL(ValueError):
    pass


@dataclass
class Page:
    url: str
    title: str
    text: str


def find_urls(text: str) -> list[str]:
    return [u.rstrip(".,;:!?") for u in URL_RE.findall(text)]


def check_url(url: str, allow_private: bool = False) -> None:
    """Reject non-http(s) URLs and hosts that resolve to private/loopback/link-local addresses (SSRF guard)."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UnsafeURL(f"Only http(s) URLs are allowed: {url}")
    if allow_private:
        return
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 80, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeURL(f"Cannot resolve host '{parsed.hostname}'") from exc
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise UnsafeURL(f"Blocked non-public address for host '{parsed.hostname}'")


def fetch(url: str, client: httpx.Client, allow_private: bool = False, max_redirects: int = 3) -> tuple[str, str]:
    """GET a page, validating every redirect hop. Returns (final_url, html)."""
    for _ in range(max_redirects + 1):
        check_url(url, allow_private)
        with client.stream("GET", url, follow_redirects=False) as r:
            if r.is_redirect and r.headers.get("location"):
                url = urljoin(url, r.headers["location"])
                continue
            r.raise_for_status()
            ctype = r.headers.get("content-type", "")
            if "html" not in ctype and "text" not in ctype:
                raise ValueError(f"Unsupported content type '{ctype}'")
            body = b""
            for chunk in r.iter_bytes():
                body += chunk
                if len(body) > MAX_BYTES:
                    break
            return url, body.decode(r.encoding or "utf-8", errors="replace")
    raise ValueError("Too many redirects")


def extract(html: str, url: str) -> tuple[Page, list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href, _ = urldefrag(urljoin(url, a["href"]))
        if href.startswith(("http://", "https://")):
            links.append(href)
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]):
        tag.decompose()
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else url
    text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
    return Page(url=url, title=title, text=text[:MAX_CHARS]), links


def crawl(seeds: list[str], max_pages: int = 3, max_depth: int = 0, allow_private: bool = False,
          timeout: float = 8.0, client: httpx.Client | None = None) -> list[Page]:
    """Breadth-first crawl. Only follows links on the same host as the seed that discovered them."""
    own = client is None
    client = client or httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT})
    pages: list[Page] = []
    seen: set[str] = set()
    queue: deque[tuple[str, int, str]] = deque((u, 0, urlparse(u).netloc) for u in seeds)
    try:
        while queue and len(pages) < max_pages:
            url, depth, host = queue.popleft()
            if url in seen:
                continue
            seen.add(url)
            try:
                final_url, html = fetch(url, client, allow_private)
            except (UnsafeURL, httpx.HTTPError, ValueError):
                continue
            page, links = extract(html, final_url)
            if page.text:
                pages.append(page)
            if depth < max_depth:
                queue.extend((l, depth + 1, host) for l in links if urlparse(l).netloc == host and l not in seen)
    finally:
        if own:
            client.close()
    return pages


def search_web(query: str, n: int = 3, timeout: float = 8.0) -> list[str]:
    """Return top result URLs from DuckDuckGo's HTML endpoint. Returns [] on any failure."""
    try:
        r = httpx.post("https://html.duckduckgo.com/html/", data={"q": query},
                       headers={"User-Agent": USER_AGENT}, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
    except httpx.HTTPError:
        return []
    urls: list[str] = []
    for a in BeautifulSoup(r.text, "html.parser").select("a.result__a"):
        href = a.get("href", "")
        if "uddg=" in href:
            href = parse_qs(urlparse(href).query).get("uddg", [""])[0]
        if href.startswith("http") and href not in urls:
            urls.append(href)
        if len(urls) >= n:
            break
    return urls


def research_web(question: str, max_pages: int = 3, allow_private: bool = False) -> list[Page]:
    """Crawl URLs mentioned in the question (following same-site links one level), else search the web."""
    urls = find_urls(question)
    if urls:
        return crawl(urls[:max_pages], max_pages=max_pages, max_depth=1, allow_private=allow_private)
    return crawl(search_web(question, n=max_pages), max_pages=max_pages, max_depth=0, allow_private=allow_private)
