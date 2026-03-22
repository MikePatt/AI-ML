"""Train TF–IDF + logistic regression baseline; save joblib artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

LABELS = ("negative", "neutral", "positive")

DEFAULT_OUT = Path(__file__).resolve().parent / "artifacts" / "sklearn_tfidf_lr.joblib"


def load_labeled_rows(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {suffix} (use .jsonl or .csv)")
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Data must have columns: text, label")
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    bad = set(df["label"].unique()) - set(LABELS)
    if bad:
        raise ValueError(f"Unknown labels: {bad}. Expected subset of {LABELS}")
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50_000,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train finance portrayal baseline (TF-IDF + LR)")
    p.add_argument("data", type=Path, help="JSONL or CSV with columns text, label")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output joblib path",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args(argv)

    df = load_labeled_rows(args.data)
    if len(df) < 2:
        print("Need at least 2 labeled rows for train/test split.", file=sys.stderr)
        return 2
    if len(df) < 6:
        print(f"Warning: only {len(df)} rows; metrics may be unstable.")

    vc = df["label"].value_counts()
    n_classes = int(df["label"].nunique())
    n_test = max(1, int(len(df) * args.test_size))
    can_strat = n_classes > 1 and bool((vc >= 2).all()) and n_test >= n_classes
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["label"] if can_strat else None,
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"accuracy (holdout): {acc:.4f}")
    print(classification_report(y_test, pred, labels=list(LABELS), zero_division=0))

    classes = list(pipe.classes_)
    artifact = {
        "pipeline": pipe,
        "model_name": "sklearn_tfidf_logreg",
        "model_version": "1",
        "label_classes": classes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.out)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
