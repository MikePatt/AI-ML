"""Load saved baseline model; chunk/truncate text; return label + scores."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

DEFAULT_MODEL = Path(__file__).resolve().parent / "artifacts" / "sklearn_tfidf_lr.joblib"


def combine_title_body(title: Optional[str], body: str) -> str:
    parts: list[str] = []
    if title and str(title).strip():
        parts.append(str(title).strip())
    if body and str(body).strip():
        parts.append(str(body).strip())
    return "\n\n".join(parts)


def chunk_text(text: str, max_chunk_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chunk_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + max_chunk_chars])
        start += max_chunk_chars
    return chunks


def head_tail_truncate(text: str, max_chars: int) -> str:
    """If over max_chars, keep head and tail with a separator (recorded in model_version elsewhere)."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n...\n\n" + text[-half:]


def load_predictor(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def predict_document(
    artifact: dict[str, Any],
    text: str,
    *,
    max_chunk_chars: int = 8000,
    strategy: str = "chunk_mean",
) -> tuple[str, dict[str, float], str]:
    """
    strategy: 'chunk_mean' (average proba over chunks) or 'head_tail' (single truncated string).
    Returns (label, scores_dict, strategy_note).
    """
    pipe: Pipeline = artifact["pipeline"]
    classes: list[str] = list(pipe.classes_)

    if strategy == "head_tail":
        combined = head_tail_truncate(text, max_chunk_chars)
        proba = pipe.predict_proba([combined])[0]
        note = f"head_tail_max_{max_chunk_chars}"
    else:
        parts = chunk_text(text, max_chunk_chars)
        if not parts:
            uniform = 1.0 / len(classes)
            return (
                classes[0],
                {c: uniform for c in classes},
                f"empty_text_{strategy}",
            )
        mat = np.vstack([pipe.predict_proba([c]) for c in parts])
        proba = mat.mean(axis=0)
        note = f"chunk_mean_{max_chunk_chars}_n{len(parts)}"

    idx = int(np.argmax(proba))
    label = classes[idx]
    scores = {classes[i]: float(proba[i]) for i in range(len(classes))}
    return label, scores, note
