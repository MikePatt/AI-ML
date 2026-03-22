"""Tests for finance_sentiment.train data loading."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from finance_sentiment.train import LABELS, build_pipeline, load_labeled_rows


def test_load_labeled_rows_jsonl(tmp_path) -> None:
    p = tmp_path / "d.jsonl"
    rows = [
        {"text": "alpha", "label": "positive"},
        {"text": "beta", "label": "NEGATIVE"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    df = load_labeled_rows(p)
    assert list(df["label"]) == ["positive", "negative"]


def test_load_labeled_rows_csv(tmp_path) -> None:
    p = tmp_path / "d.csv"
    p.write_text("text,label\nhello,neutral\n", encoding="utf-8")
    df = load_labeled_rows(p)
    assert len(df) == 1
    assert df.iloc[0]["label"] == "neutral"


def test_load_labeled_rows_bad_label(tmp_path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"text": "x", "label": "spam"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown labels"):
        load_labeled_rows(p)


def test_build_pipeline_fit_predict() -> None:
    df = pd.DataFrame(
        {"text": ["pos words"] * 4 + ["neg words"] * 4, "label": ["positive"] * 4 + ["negative"] * 4}
    )
    pipe = build_pipeline()
    pipe.fit(df["text"], df["label"])
    pred = pipe.predict(["pos words"])
    assert pred[0] in LABELS
