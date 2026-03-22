"""Tests for finance_ingest.pipeline."""

from __future__ import annotations

from unittest.mock import patch

from finance_ingest.fetch import FetchResult
from finance_ingest.pipeline import ingest_urls


@patch("finance_ingest.pipeline.fetch_url")
def test_ingest_urls_stores_page(mock_fetch, tmp_path) -> None:
    html = "<html><head><title>Hello</title></head><body><p>World</p></body></html>"
    mock_fetch.return_value = FetchResult(
        url="https://example.com/",
        status_code=200,
        body=html,
        error=None,
    )
    db = tmp_path / "p.db"
    pairs = ingest_urls(db, ["https://example.com/"], timeout_seconds=5.0)
    assert len(pairs) == 1
    url, pid = pairs[0]
    assert url == "https://example.com/"
    assert pid >= 1

    from finance_ingest import db as dbmod

    with dbmod.db_session(db) as conn:
        row = dbmod.get_page_by_id(conn, pid)
    assert row is not None
    assert row["title"] == "Hello"
    assert "World" in (row["extracted_text"] or "")


@patch("finance_ingest.pipeline.fetch_url")
def test_ingest_skips_empty_url(mock_fetch, tmp_path) -> None:
    mock_fetch.return_value = FetchResult(url="x", status_code=200, body="<html><body>x</body></html>", error=None)
    pairs = ingest_urls(tmp_path / "e.db", ["", "  ", "https://z.test/"])
    assert len(pairs) == 1
    assert pairs[0][0] == "https://z.test/"
    assert mock_fetch.call_count == 1
