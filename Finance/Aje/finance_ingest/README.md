# finance_ingest

Fetch finance-related URLs, extract readable text, and store rows in SQLite. This package does **not** run ML; it feeds the same database that `finance_sentiment` reads for predictions.

## Setup

From the **Aje** project root (parent of `finance_ingest/`):

```bash
pip install -r requirements-finance.txt
```

Run CLIs with `python -m …` so imports resolve.

## SQLite schema

- **`pages`** — `id`, `url` (unique), `title`, `extracted_text`, `fetched_at`, `status_code`, `error`, `content_hash`
- **`sentiment_predictions`** — defined in `db.py` for use by the sentiment CLI (foreign key to `pages`)

Schema and helpers live in `db.py` (`init_db`, `upsert_page`, `list_pages`, export helpers, etc.).

## Pipeline

1. **`fetch.py`** — HTTP GET via `httpx` with timeouts; records status and optional error for HTTP ≥ 400.
2. **`extract.py`** — HTML → plain text with BeautifulSoup + `lxml` (drops `script` / `style`, etc.).
3. **`pipeline.py`** — `ingest_urls(db_path, urls, timeout_seconds=…)` runs fetch → extract → `upsert_page`.

## CLI (`python -m finance_ingest.cli`)

| Command | Purpose |
| -------- | -------- |
| `ingest` | Fetch URLs and upsert `pages` |
| `list` | List stored pages |
| `export-jsonl` | Export pages plus **latest** sentiment row per page (if any) |

Examples:

```bash
python -m finance_ingest.cli ingest --db finance_data.db --url https://example.com/
python -m finance_ingest.cli ingest --db finance_data.db --urls-file urls.txt --timeout 45
python -m finance_ingest.cli list --db finance_data.db --limit 50
python -m finance_ingest.cli export-jsonl --db finance_data.db --out export.jsonl
```

## Library usage

```python
from finance_ingest.pipeline import ingest_urls

pairs = ingest_urls("finance_data.db", ["https://example.com/"], timeout_seconds=30.0)
# pairs: list of (url, page_id)
```

## Testing

From the **Aje** project root:

```bash
pip install -r requirements-dev.txt
pytest
```

(`pytest.ini` sets `pythonpath` so `finance_ingest` / `finance_sentiment` import correctly.)

## Notes

- Respect site terms of use; prefer official APIs when required.
- Heavy JavaScript sites may need a browser-based fetch (e.g. Playwright) upstream; this stack expects HTML from a normal GET.
