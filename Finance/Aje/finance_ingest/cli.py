"""CLI: ingest, list, export-jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from finance_ingest import db as dbmod


def cmd_ingest(args: argparse.Namespace) -> int:
    from finance_ingest.pipeline import ingest_urls

    urls: list[str] = []
    if args.url:
        urls.extend(args.url)
    if args.urls_file:
        p = Path(args.urls_file)
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    if not urls:
        print("No URLs: pass --url or --urls-file", file=sys.stderr)
        return 2
    pairs = ingest_urls(args.db, urls, timeout_seconds=args.timeout)
    for u, pid in pairs:
        print(f"ingested page_id={pid} url={u}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    with dbmod.db_session(args.db) as conn:
        rows = dbmod.list_pages(conn, limit=args.limit, offset=args.offset)
    for r in rows:
        tid = (r.get("title") or "")[:80]
        print(f"{r['id']}\t{r['url']}\t{tid!r}")
    return 0


def cmd_export_jsonl(args: argparse.Namespace) -> int:
    with dbmod.db_session(args.db) as conn:
        rows = dbmod.pages_to_export_jsonl_rows(conn)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} lines to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="finance-ingest", description="Finance page ingest CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="Fetch URLs and upsert pages")
    pi.add_argument("--db", default="finance_data.db", help="SQLite database path")
    pi.add_argument("--url", action="append", help="URL (repeatable)")
    pi.add_argument("--urls-file", help="File with one URL per line")
    pi.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds")
    pi.set_defaults(func=cmd_ingest)

    pl = sub.add_parser("list", help="List stored pages")
    pl.add_argument("--db", default="finance_data.db", help="SQLite database path")
    pl.add_argument("--limit", type=int, default=100)
    pl.add_argument("--offset", type=int, default=0)
    pl.set_defaults(func=cmd_list)

    pe = sub.add_parser("export-jsonl", help="Export pages + latest sentiment to JSONL")
    pe.add_argument("--db", default="finance_data.db", help="SQLite database path")
    pe.add_argument("--out", required=True, help="Output JSONL path")
    pe.set_defaults(func=cmd_export_jsonl)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
