"""HTTP fetch with timeouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
DEFAULT_HEADERS = {
    "User-Agent": "FinanceIngestBot/1.0 (+https://example.local; research)",
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class FetchResult:
    url: str
    status_code: Optional[int]
    body: Optional[str]
    error: Optional[str]


def fetch_url(
    url: str,
    *,
    timeout: httpx.Timeout | float | None = None,
    follow_redirects: bool = True,
    max_redirects: int = 10,
) -> FetchResult:
    t = timeout if timeout is not None else DEFAULT_TIMEOUT
    if isinstance(t, (int, float)):
        t = httpx.Timeout(float(t), connect=min(10.0, float(t)))
    try:
        with httpx.Client(timeout=t, follow_redirects=follow_redirects, max_redirects=max_redirects) as client:
            r = client.get(url, headers=DEFAULT_HEADERS)
            text = r.text if r.content else ""
            err: Optional[str] = None
            if r.status_code >= 400:
                err = f"HTTP {r.status_code}"
            return FetchResult(url=str(r.url), status_code=r.status_code, body=text, error=err)
    except httpx.RequestError as e:
        return FetchResult(url=url, status_code=None, body=None, error=str(e))
