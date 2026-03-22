"""Tests for finance_ingest.db."""

from __future__ import annotations

import sqlite3

import pytest

from finance_ingest import db as dbmod


def test_content_hash_stable() -> None:
    h = dbmod.content_hash("hello")
    assert len(h) == 64
    assert h == dbmod.content_hash("hello")


def test_upsert_page_insert_and_update(tmp_path) -> None:
    db = tmp_path / "t.db"
    with dbmod.db_session(db) as conn:
        pid1 = dbmod.upsert_page(conn, "https://a.example", "T1", "body1", 200, None)
        pid2 = dbmod.upsert_page(conn, "https://a.example", "T2", "body2", 200, None)
    assert pid1 == pid2
    with dbmod.db_session(db) as conn:
        row = dbmod.get_page_by_id(conn, pid1)
    assert row is not None
    assert row["title"] == "T2"
    assert row["extracted_text"] == "body2"
    assert row["content_hash"] == dbmod.content_hash("body2")


def test_insert_sentiment_and_export_latest(tmp_path) -> None:
    db = tmp_path / "t.db"
    with dbmod.db_session(db) as conn:
        pid = dbmod.upsert_page(conn, "https://b.example", "T", "text", 200, None)
        dbmod.insert_sentiment_prediction(
            conn, pid, "document", "neutral", {"neutral": 0.9}, "m", "v1"
        )
        dbmod.insert_sentiment_prediction(
            conn, pid, "document", "positive", {"positive": 0.8}, "m", "v2"
        )
        rows = dbmod.export_pages_with_latest_prediction(conn)
    assert len(rows) == 1
    assert rows[0]["page_id"] == pid
    assert rows[0]["label"] == "positive"
    assert rows[0]["scores"] == {"positive": 0.8}


def test_iter_pages_for_prediction_skips_empty(tmp_path) -> None:
    db = tmp_path / "t.db"
    with dbmod.db_session(db) as conn:
        p1 = dbmod.upsert_page(conn, "https://c.example", None, "   ", 200, None)
        p2 = dbmod.upsert_page(conn, "https://d.example", None, "ok", 200, None)
        found = list(dbmod.iter_pages_for_prediction(conn))
        ids = {r["id"] for r in found}
    assert p2 in ids
    assert p1 not in ids


def test_iter_pages_for_prediction_filter_ids(tmp_path) -> None:
    db = tmp_path / "t.db"
    with dbmod.db_session(db) as conn:
        p1 = dbmod.upsert_page(conn, "https://e.example", None, "a", 200, None)
        p2 = dbmod.upsert_page(conn, "https://f.example", None, "b", 200, None)
        found = list(dbmod.iter_pages_for_prediction(conn, page_ids=[p2]))
    assert [r["id"] for r in found] == [p2]


def test_pages_to_export_jsonl_rows(tmp_path) -> None:
    db = tmp_path / "t.db"
    with dbmod.db_session(db) as conn:
        pid = dbmod.upsert_page(conn, "https://g.example", "My Title", "Para", 200, None)
        dbmod.insert_sentiment_prediction(conn, pid, "document", "negative", None, "x", "1")
        rows = dbmod.pages_to_export_jsonl_rows(conn)
    assert len(rows) == 1
    assert rows[0]["page_id"] == pid
    assert "My Title" in rows[0]["text"]
    assert "Para" in rows[0]["text"]
    assert rows[0]["sentiment_label"] == "negative"


def test_insert_sentiment_fk_violation(tmp_path) -> None:
    db = tmp_path / "t.db"
    conn = dbmod.connect(db)
    dbmod.init_db(conn)
    with pytest.raises(sqlite3.IntegrityError):
        dbmod.insert_sentiment_prediction(conn, 9999, "document", "neutral", None, "m", "v")
    conn.close()
