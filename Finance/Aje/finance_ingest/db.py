"""SQLite schema and helpers for pages and sentiment_predictions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterable, Optional

DDL = """
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL UNIQUE,
    title TEXT,
    extracted_text TEXT,
    fetched_at TEXT NOT NULL,
    status_code INTEGER,
    error TEXT,
    content_hash TEXT
);

CREATE INDEX IF NOT EXISTS idx_pages_fetched_at ON pages(fetched_at);
CREATE INDEX IF NOT EXISTS idx_pages_content_hash ON pages(content_hash);

CREATE TABLE IF NOT EXISTS sentiment_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id INTEGER NOT NULL,
    granularity TEXT NOT NULL,
    label TEXT NOT NULL,
    scores TEXT,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    predicted_at TEXT NOT NULL,
    FOREIGN KEY (page_id) REFERENCES pages(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sentiment_page_id ON sentiment_predictions(page_id);
CREATE INDEX IF NOT EXISTS idx_sentiment_predicted_at ON sentiment_predictions(predicted_at);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    conn.commit()


@contextmanager
def db_session(db_path: str | Path) -> Generator[sqlite3.Connection, None, None]:
    conn = connect(db_path)
    try:
        init_db(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class PageRow:
    id: Optional[int]
    url: str
    title: Optional[str]
    extracted_text: Optional[str]
    fetched_at: str
    status_code: Optional[int]
    error: Optional[str]
    content_hash: Optional[str]


def upsert_page(
    conn: sqlite3.Connection,
    url: str,
    title: Optional[str],
    extracted_text: Optional[str],
    status_code: Optional[int],
    error: Optional[str],
) -> int:
    now = datetime.now(timezone.utc).isoformat()
    h = content_hash(extracted_text or "") if extracted_text is not None else None
    cur = conn.execute(
        """
        INSERT INTO pages (url, title, extracted_text, fetched_at, status_code, error, content_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            title = excluded.title,
            extracted_text = excluded.extracted_text,
            fetched_at = excluded.fetched_at,
            status_code = excluded.status_code,
            error = excluded.error,
            content_hash = excluded.content_hash
        RETURNING id
        """,
        (url, title, extracted_text, now, status_code, error, h),
    )
    row = cur.fetchone()
    assert row is not None
    return int(row[0])


def list_pages(conn: sqlite3.Connection, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    cur = conn.execute(
        """
        SELECT id, url, title, extracted_text, fetched_at, status_code, error, content_hash
        FROM pages ORDER BY id DESC LIMIT ? OFFSET ?
        """,
        (limit, offset),
    )
    return [dict(r) for r in cur.fetchall()]


def get_page_by_id(conn: sqlite3.Connection, page_id: int) -> Optional[dict[str, Any]]:
    cur = conn.execute(
        """
        SELECT id, url, title, extracted_text, fetched_at, status_code, error, content_hash
        FROM pages WHERE id = ?
        """,
        (page_id,),
    )
    r = cur.fetchone()
    return dict(r) if r else None


def iter_pages_for_prediction(
    conn: sqlite3.Connection,
    page_ids: Optional[Iterable[int]] = None,
) -> Generator[dict[str, Any], None, None]:
    if page_ids is None:
        cur = conn.execute(
            """
            SELECT id, url, title, extracted_text FROM pages
            WHERE extracted_text IS NOT NULL AND length(trim(extracted_text)) > 0
            ORDER BY id
            """
        )
        for r in cur:
            yield dict(r)
    else:
        ids = list(page_ids)
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        cur = conn.execute(
            f"""
            SELECT id, url, title, extracted_text FROM pages
            WHERE id IN ({placeholders})
            AND extracted_text IS NOT NULL AND length(trim(extracted_text)) > 0
            ORDER BY id
            """,
            ids,
        )
        for r in cur:
            yield dict(r)


def insert_sentiment_prediction(
    conn: sqlite3.Connection,
    page_id: int,
    granularity: str,
    label: str,
    scores: Optional[dict[str, float]],
    model_name: str,
    model_version: str,
) -> int:
    now = datetime.now(timezone.utc).isoformat()
    scores_json = json.dumps(scores) if scores is not None else None
    cur = conn.execute(
        """
        INSERT INTO sentiment_predictions
        (page_id, granularity, label, scores, model_name, model_version, predicted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        RETURNING id
        """,
        (page_id, granularity, label, scores_json, model_name, model_version, now),
    )
    row = cur.fetchone()
    assert row is not None
    return int(row[0])


def export_pages_with_latest_prediction(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Join each page with its latest sentiment row (by predicted_at, then id)."""
    cur = conn.execute(
        """
        SELECT
            p.id AS page_id,
            p.url,
            p.title,
            p.extracted_text,
            p.fetched_at,
            p.status_code,
            p.error,
            p.content_hash,
            s.id AS prediction_id,
            s.granularity,
            s.label,
            s.scores,
            s.model_name,
            s.model_version,
            s.predicted_at
        FROM pages p
        LEFT JOIN sentiment_predictions s ON s.id = (
            SELECT s2.id FROM sentiment_predictions s2
            WHERE s2.page_id = p.id
            ORDER BY s2.predicted_at DESC, s2.id DESC
            LIMIT 1
        )
        ORDER BY p.id
        """
    )
    out = []
    for r in cur.fetchall():
        d = dict(r)
        if d.get("scores") and isinstance(d["scores"], str):
            try:
                d["scores"] = json.loads(d["scores"])
            except json.JSONDecodeError:
                pass
        out.append(d)
    return out


def pages_to_export_jsonl_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Rows suitable for JSONL export (text + metadata)."""
    rows = export_pages_with_latest_prediction(conn)
    result = []
    for d in rows:
        text_parts = []
        if d.get("title"):
            text_parts.append(str(d["title"]))
        if d.get("extracted_text"):
            text_parts.append(str(d["extracted_text"]))
        result.append(
            {
                "page_id": d["page_id"],
                "url": d["url"],
                "title": d.get("title"),
                "text": "\n\n".join(text_parts) if text_parts else "",
                "fetched_at": d.get("fetched_at"),
                "sentiment_label": d.get("label"),
                "sentiment_scores": d.get("scores"),
                "model_name": d.get("model_name"),
                "model_version": d.get("model_version"),
                "predicted_at": d.get("predicted_at"),
            }
        )
    return result
