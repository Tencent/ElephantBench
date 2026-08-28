"""Bounded public-web search and fetch tools with a persistent local cache."""

from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import re
import socket
import tempfile
import threading
import urllib.parse
import urllib.request
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

DEFAULT_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
SEARCH_CACHE_VERSION = "brave-search-v1"
USER_AGENT = "Mozilla/5.0 (compatible; ElephantBench/0.1; public-evidence-verifier)"


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _public_http_url(url: str, *, allow_private: bool = False) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("WebFetch accepts only absolute HTTP(S) URLs")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed")
    if allow_private:
        return url
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError(f"non-public destination is not allowed: {parsed.hostname}")
    return url


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, allow_private: bool) -> None:
        super().__init__()
        self.allow_private = allow_private

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        _public_http_url(newurl, allow_private=self.allow_private)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.title_depth = 0
        self.title_parts: list[str] = []
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg", "template"}:
            self.skip_depth += 1
        elif tag == "title":
            self.title_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg", "template"} and self.skip_depth:
            self.skip_depth -= 1
        elif tag == "title" and self.title_depth:
            self.title_depth -= 1
        elif tag in {"p", "div", "li", "br", "h1", "h2", "h3", "tr", "section"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        if self.title_depth:
            self.title_parts.append(data)
        self.parts.append(data)

    @property
    def title(self) -> str:
        return _normalize_space(" ".join(self.title_parts))

    @property
    def text(self) -> str:
        lines = [_normalize_space(line) for line in " ".join(self.parts).splitlines()]
        return "\n".join(line for line in lines if line)


def parse_brave_results(payload: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    """Normalize Brave Web Search results for the verification agent."""
    web = payload.get("web")
    raw_results = web.get("results") if isinstance(web, dict) else []
    if not isinstance(raw_results, list):
        return []
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or url in seen:
            continue
        seen.add(url)
        results.append(
            {
                "rank": len(results) + 1,
                "title": _normalize_space(html.unescape(str(item.get("title") or ""))),
                "url": url,
                "snippet": _normalize_space(html.unescape(str(item.get("description") or ""))),
            }
        )
        if len(results) >= limit:
            break
    return results


class WebCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def get_or_create(
        self, kind: str, key: str, producer: Callable[[], dict[str, Any]]
    ) -> dict[str, Any]:
        path = self.directory / f"{kind}_{_hash(key)}.json"
        with self.lock:
            if path.exists():
                value = json.loads(path.read_text(encoding="utf-8"))
                value["cache_hit"] = True
                return value
        value = producer()
        value["cache_hit"] = False
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.directory,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                temporary = Path(handle.name)
            with self.lock:
                temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return value


class WebTools:
    def __init__(
        self,
        cache_dir: Path,
        *,
        search_endpoint: str = DEFAULT_SEARCH_ENDPOINT,
        search_api_key: str = "",
        timeout: float = 30.0,
        search_results: int = 5,
        search_concurrency: int = 2,
        fetch_concurrency: int = 8,
        allow_private_network: bool = False,
    ) -> None:
        parsed_endpoint = urllib.parse.urlsplit(search_endpoint)
        if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.hostname:
            raise ValueError("search endpoint must be an absolute HTTP(S) URL")
        self.cache = WebCache(cache_dir)
        self.search_endpoint = search_endpoint
        self.search_api_key = search_api_key
        self.timeout = timeout
        self.search_results = search_results
        self.search_semaphore = threading.Semaphore(max(1, search_concurrency))
        self.fetch_semaphore = threading.Semaphore(max(1, fetch_concurrency))
        self.allow_private_network = allow_private_network
        self.opener = urllib.request.build_opener(_SafeRedirectHandler(allow_private_network))

    def _request(
        self, url: str, *, extra_headers: dict[str, str] | None = None
    ) -> tuple[str, bytes, str]:
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": (
                "text/html,application/xhtml+xml,text/plain,application/json;q=0.8,*/*;q=0.2"
            ),
        }
        headers.update(extra_headers or {})
        request = urllib.request.Request(
            url,
            headers=headers,
        )
        with self.opener.open(request, timeout=self.timeout) as response:
            body = response.read()
            content_type = str(response.headers.get("Content-Type") or "")
            return str(response.geturl()), body, content_type

    def search(self, query: str) -> dict[str, Any]:
        query = _normalize_space(query)
        if not query:
            raise ValueError("WebSearch requires a non-empty query")
        if len(query) > 400 or len(query.split()) > 50:
            raise ValueError("Brave Web Search queries are limited to 400 characters and 50 words")
        if not self.search_api_key:
            raise ValueError("WebSearch requires BRAVE_SEARCH_API_KEY")

        def produce() -> dict[str, Any]:
            parameters = urllib.parse.urlencode(
                {
                    "q": query,
                    "count": max(1, min(self.search_results, 20)),
                    "safesearch": "moderate",
                    "text_decorations": "false",
                }
            )
            separator = "&" if "?" in self.search_endpoint else "?"
            url = self.search_endpoint + separator + parameters
            with self.search_semaphore:
                final_url, body, content_type = self._request(
                    url,
                    extra_headers={
                        "Accept": "application/json",
                        "X-Subscription-Token": self.search_api_key,
                    },
                )
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Brave Search response is not a JSON object")
            return {
                "query": query,
                "search_url": final_url,
                "content_type": content_type,
                "provider": "brave",
                "results": parse_brave_results(payload, self.search_results),
            }

        cache_key = f"{SEARCH_CACHE_VERSION}|{self.search_endpoint}|{query.casefold()}"
        return self.cache.get_or_create("search", cache_key, produce)

    def fetch(self, url: str) -> dict[str, Any]:
        url = _public_http_url(url.strip(), allow_private=self.allow_private_network)

        def produce() -> dict[str, Any]:
            with self.fetch_semaphore:
                final_url, body, content_type = self._request(url)
            _public_http_url(final_url, allow_private=self.allow_private_network)
            charset_match = re.search(r"charset=([^;\s]+)", content_type, flags=re.I)
            charset = charset_match.group(1).strip("\"'") if charset_match else "utf-8"
            try:
                decoded = body.decode(charset, errors="replace")
            except LookupError:
                decoded = body.decode("utf-8", errors="replace")
            if "html" in content_type.casefold() or "<html" in decoded[:1000].casefold():
                parser = _TextExtractor()
                parser.feed(decoded)
                title, text = parser.title, parser.text
            else:
                title, text = final_url, _normalize_space(decoded)
            return {
                "requested_url": url,
                "url": final_url,
                "title": title or final_url,
                "content_type": content_type,
                "text": text,
                "content_sha256": hashlib.sha256(body).hexdigest(),
            }

        return self.cache.get_or_create("fetch", url, produce)
