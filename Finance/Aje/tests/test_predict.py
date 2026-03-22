"""Tests for finance_sentiment.predict."""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from finance_sentiment.predict import (
    chunk_text,
    combine_title_body,
    head_tail_truncate,
    predict_document,
)
from finance_sentiment.train import build_pipeline


def test_combine_title_body() -> None:
    assert combine_title_body("T", "B") == "T\n\nB"
    assert combine_title_body(None, "B") == "B"
    assert combine_title_body("", "B") == "B"
    assert combine_title_body("T", "") == "T"


def test_chunk_text() -> None:
    assert chunk_text("", 10) == []
    assert chunk_text("abc", 10) == ["abc"]
    assert chunk_text("abcdefghij", 4) == ["abcd", "efgh", "ij"]


def test_head_tail_truncate() -> None:
    s = "a" * 100
    out = head_tail_truncate(s, 20)
    assert len(out) < len(s)
    assert "..." in out
    assert out.startswith("aa")
    assert out.endswith("aa")


def test_predict_document_chunk_mean() -> None:
    df = pd.DataFrame(
        {
            "text": ["good profit beat"] * 3 + ["loss miss down"] * 3 + ["unchanged flat"] * 3,
            "label": ["positive"] * 3 + ["negative"] * 3 + ["neutral"] * 3,
        }
    )
    pipe: Pipeline = build_pipeline()
    pipe.fit(df["text"], df["label"])
    artifact = {"pipeline": pipe}
    label, scores, note = predict_document(artifact, "strong profit growth outlook", strategy="chunk_mean", max_chunk_chars=500)
    assert label in scores
    assert "chunk_mean" in note


def test_predict_document_head_tail() -> None:
    df = pd.DataFrame({"text": ["up up up", "down down"], "label": ["positive", "negative"]})
    pipe = build_pipeline()
    pipe.fit(df["text"], df["label"])
    artifact = {"pipeline": pipe}
    long = "word " * 500
    label, scores, note = predict_document(artifact, long, strategy="head_tail", max_chunk_chars=80)
    assert label in scores
    assert "head_tail" in note


def test_predict_document_empty_text() -> None:
    df = pd.DataFrame(
        {"text": ["profit growth", "loss decline"], "label": ["positive", "negative"]}
    )
    pipe = build_pipeline()
    pipe.fit(df["text"], df["label"])
    artifact = {"pipeline": pipe}
    label, scores, note = predict_document(artifact, "   ", strategy="chunk_mean")
    assert "empty" in note
    assert abs(sum(scores.values()) - 1.0) < 1e-6
