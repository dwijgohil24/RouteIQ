"""
Adapter Pattern — HTTP client.

WeatherEngine, GeoEngine, and OSRMStrategy all previously held raw httpx.get()
calls with duplicated timeout/retry/status-check logic. They now depend on
HttpAdapter — a single seam. Swapping to aiohttp, requests, or a mock test
adapter requires changing only this file and the module-level default.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx


# ── Target interface ──────────────────────────────────────────────────────────

class HttpAdapter(ABC):
    @abstractmethod
    def get(self, url: str | list[str], **kwargs) -> dict | None:
        """
        GET one or more URLs (tried in order). Returns parsed JSON or None.
        Implementations handle timeout, retries, and status-code validation.
        """


# ── Concrete Adapter ──────────────────────────────────────────────────────────

class HttpxAdapter(HttpAdapter):
    def __init__(self, timeout: float = 8.0) -> None:
        self._timeout = timeout

    def get(self, url: str | list[str], **kwargs) -> dict | None:
        urls: list[str] = [url] if isinstance(url, str) else url
        kwargs.setdefault("follow_redirects", True)

        for u in urls:
            try:
                resp = httpx.get(u, timeout=self._timeout, **kwargs)
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                continue
        return None


# ── Module-level singleton ────────────────────────────────────────────────────

_adapter: HttpAdapter = HttpxAdapter(timeout=8.0)


def get_http_adapter() -> HttpAdapter:
    return _adapter


def set_http_adapter(adapter: HttpAdapter) -> None:
    global _adapter
    _adapter = adapter
