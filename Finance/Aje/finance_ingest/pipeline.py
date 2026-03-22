"""Orchestrate fetch → extract → DB upsert."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import sqlite3

from finance_ingest.db import init_db, upsert_page
from finance_ingest.extract import extract_title_and_text
from finance_ingest.fetch import fetch_url


def ingest_urls(
    db_path: str | Path,
    urls: Iterable[str],
    *,
    timeout_seconds: float = 30.0,
) -> list[tuple[str, int]]:
    """Fetch and store each URL; returns (url, page_id) for each non-empty URL processed."""
    path = Path(db_path)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        init_db(conn)
        results: list[tuple[str, int]] = []
        for url in urls:
            u = url.strip()
            if not u:
                continue
            fr = fetch_url(u, timeout=timeout_seconds)
            title = None
            extracted = None
            if fr.body:
                try:
                    title, extracted = extract_title_and_text(fr.body)
                except Exception as e:  # noqa: BLE001 — keep ingest resilient
                    pid = upsert_page(conn, u, None, None, fr.status_code, f"extract: {e}")
                    results.append((u, pid))
                    continue
            if fr.error and not extracted:
                pid = upsert_page(conn, u, title, extracted, fr.status_code, fr.error)
            else:
                pid = upsert_page(conn, u, title, extracted, fr.status_code, fr.error)
            results.append((u, pid))
        conn.commit()
        return results
    finally:
        conn.close()
