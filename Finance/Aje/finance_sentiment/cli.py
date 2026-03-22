"""CLI: predict — read pages from SQLite, write sentiment_predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from finance_ingest import db as dbmod
from finance_sentiment.predict import (
    DEFAULT_MODEL,
    combine_title_body,
    load_predictor,
    predict_document,
)


def run_dummy_prediction(page_id: int) -> tuple[str, dict[str, float], str]:
    cycle = ("neutral", "positive", "negative")
    label = cycle[page_id % 3]
    scores = {"negative": 0.15, "neutral": 0.15, "positive": 0.15}
    scores[label] = 0.70
    return label, scores, "dummy_rotating_v1"


def cmd_predict(args: argparse.Namespace) -> int:
    model_path = Path(args.model) if args.model else DEFAULT_MODEL
    page_ids = None
    if args.page_ids:
        page_ids = [int(x.strip()) for x in args.page_ids.split(",") if x.strip()]

    artifact = None
    if not args.dummy:
        try:
            artifact = load_predictor(model_path)
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            print("Hint: use --dummy for smoke tests, or train with finance_sentiment/train.py", file=sys.stderr)
            return 1

    with dbmod.db_session(args.db) as conn:
        rows = list(dbmod.iter_pages_for_prediction(conn, page_ids=page_ids))
        if not rows:
            print("No pages with text to predict.", file=sys.stderr)
            return 0
        for r in rows:
            text = combine_title_body(r.get("title"), r.get("extracted_text") or "")
            if args.dummy:
                label, scores, strat = run_dummy_prediction(int(r["id"]))
                mname, mver = "dummy_rotating", strat
            else:
                assert artifact is not None
                label, scores, strat = predict_document(
                    artifact,
                    text,
                    max_chunk_chars=args.max_chunk_chars,
                    strategy=args.strategy,
                )
                mname = str(artifact.get("model_name", "sklearn_tfidf_logreg"))
                base_ver = str(artifact.get("model_version", "1"))
                mver = f"{base_ver}|{strat}"
            pid = dbmod.insert_sentiment_prediction(
                conn,
                int(r["id"]),
                granularity="document",
                label=label,
                scores=scores,
                model_name=mname,
                model_version=mver,
            )
            print(f"page_id={r['id']} prediction_id={pid} label={label}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="finance-sentiment", description="Finance sentiment CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("predict", help="Run classifier on pages; insert sentiment_predictions")
    pp.add_argument("--db", default="finance_data.db", help="SQLite database path")
    pp.add_argument(
        "--model",
        default=None,
        help=f"Joblib artifact path (default: {DEFAULT_MODEL})",
    )
    pp.add_argument(
        "--dummy",
        action="store_true",
        help="Ignore model; write rotating neutral/pos/neg labels for testing",
    )
    pp.add_argument("--page-ids", help="Comma-separated page ids (default: all pages with text)")
    pp.add_argument("--max-chunk-chars", type=int, default=8000, help="Chunk size for long texts")
    pp.add_argument(
        "--strategy",
        choices=("chunk_mean", "head_tail"),
        default="chunk_mean",
        help="How to handle long texts",
    )
    pp.set_defaults(func=cmd_predict)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
