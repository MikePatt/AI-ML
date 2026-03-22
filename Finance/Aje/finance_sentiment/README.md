# finance_sentiment

**Document-level** financial portrayal labels: `positive`, `negative`, `neutral` (see `labels_schema.md` for definitions). Trains a TF–IDF + logistic regression baseline and writes predictions into SQLite table `sentiment_predictions` (shared with `finance_ingest`).

## Setup

From the **Aje** project root:

```bash
pip install -r requirements-finance.txt
pip install -r requirements-sentiment.txt
```

## Labels and training data

- Annotator guidance: **`labels_schema.md`**
- Training file: **JSONL** or **CSV** with columns **`text`** and **`label`** (`negative` | `neutral` | `positive`)
- Tiny example: `sample_train.jsonl`

## Train

Saves a joblib bundle to `artifacts/sklearn_tfidf_lr.joblib` by default (directory kept in git via `artifacts/.gitkeep`; the `.joblib` file is gitignored).

```bash
python -m finance_sentiment.train path/to/data.jsonl -o finance_sentiment/artifacts/sklearn_tfidf_lr.joblib
```

Holdout metrics and a classification report are printed to the terminal.

## Predict (CLI)

Reads `pages` from the same SQLite path used by ingest, combines `title` + `extracted_text`, runs the classifier (or a dummy mode), and **inserts** one row per page into `sentiment_predictions` (multiple runs per page are allowed).

```bash
# Requires a trained artifact at default path, or pass --model
python -m finance_sentiment.cli predict --db finance_data.db

python -m finance_sentiment.cli predict --db finance_data.db --model path/to/model.joblib

# Smoke test without a model file
python -m finance_sentiment.cli predict --db finance_data.db --dummy

# Subset and long-text handling
python -m finance_sentiment.cli predict --db finance_data.db --page-ids 1,2,3
python -m finance_sentiment.cli predict --db finance_data.db --strategy chunk_mean --max-chunk-chars 8000
python -m finance_sentiment.cli predict --db finance_data.db --strategy head_tail --max-chunk-chars 8000
```

**Long text:** `chunk_mean` splits into chunks of `--max-chunk-chars` and averages class probabilities; `head_tail` keeps the start and end of the string in one window. Strategy is reflected in the stored `model_version` field.

## Library usage

```python
from finance_sentiment.predict import load_predictor, predict_document, combine_title_body

artifact = load_predictor("finance_sentiment/artifacts/sklearn_tfidf_lr.joblib")
text = combine_title_body(title, body)
label, scores, note = predict_document(artifact, text, max_chunk_chars=8000, strategy="chunk_mean")
```

## Testing

From the **Aje** project root:

```bash
pip install -r requirements-dev.txt
pytest
```

## End-to-end with ingest

```bash
python -m finance_ingest.cli ingest --db finance_data.db --url https://example.com/
python -m finance_sentiment.train finance_sentiment/sample_train.jsonl
python -m finance_sentiment.cli predict --db finance_data.db
python -m finance_ingest.cli export-jsonl --db finance_data.db --out pages.jsonl
```
