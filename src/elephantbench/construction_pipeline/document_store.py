"""Build the full-document SQLite store from raw source shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .source_io import document_id, expand_inputs, extract_text, extract_url, iter_jsonl

DOCS_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    canonical_doc_id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    text TEXT NOT NULL
) WITHOUT ROWID
"""

SHARDS_SCHEMA = """
CREATE TABLE IF NOT EXISTS source_shards (
    shard_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    compressed_bytes INTEGER NOT NULL,
    rows_seen INTEGER NOT NULL,
    rows_inserted INTEGER NOT NULL,
    missing_url_or_text INTEGER NOT NULL,
    completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

METADATA_SCHEMA = """
CREATE TABLE IF NOT EXISTS store_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID
"""


def _shard_id(path: Path) -> str:
    """Identify an immutable Hub shard without storing a machine-specific path."""
    size = path.stat().st_size
    value = f"{path.name}\n{size}".encode()
    return hashlib.sha1(value).hexdigest()


def _remove_database(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        candidate.unlink(missing_ok=True)


def _open_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=120.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-262144")
    connection.execute("PRAGMA wal_autocheckpoint=10000")
    connection.execute(DOCS_SCHEMA)
    connection.execute(SHARDS_SCHEMA)
    connection.execute(METADATA_SCHEMA)
    connection.executemany(
        "INSERT OR REPLACE INTO store_metadata(key,value) VALUES(?,?)",
        (
            ("schema_version", "1"),
            ("document_id", "sha1(url + newline + text)"),
            ("source", "panzs19/ElephantBench-Source"),
        ),
    )
    connection.commit()
    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(docs)").fetchall()}
    required = {"canonical_doc_id", "url", "text"}
    if not required.issubset(columns):
        raise ValueError(f"{path}: docs table is missing columns {sorted(required - columns)}")
    return connection


def _completed_shards(connection: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in connection.execute("SELECT shard_id FROM source_shards")}


def _ingest_shard(
    connection: sqlite3.Connection,
    path: Path,
    *,
    batch_size: int,
) -> dict[str, Any]:
    started = time.time()
    shard_id = _shard_id(path)
    rows_seen = 0
    missing = 0
    batch: list[tuple[str, str, str]] = []
    connection.execute("BEGIN IMMEDIATE")
    changes_before = connection.total_changes
    try:
        for _, record in iter_jsonl(path):
            rows_seen += 1
            url = extract_url(record)
            text = extract_text(record)
            if not url or not text:
                missing += 1
                continue
            batch.append((document_id(record), url, text))
            if len(batch) >= batch_size:
                connection.executemany("INSERT OR IGNORE INTO docs VALUES(?,?,?)", batch)
                batch.clear()
        if batch:
            connection.executemany("INSERT OR IGNORE INTO docs VALUES(?,?,?)", batch)
        rows_inserted = connection.total_changes - changes_before
        connection.execute(
            """
            INSERT INTO source_shards(
                shard_id, filename, compressed_bytes, rows_seen,
                rows_inserted, missing_url_or_text
            ) VALUES(?,?,?,?,?,?)
            """,
            (shard_id, path.name, path.stat().st_size, rows_seen, rows_inserted, missing),
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    return {
        "shard_id": shard_id,
        "filename": path.name,
        "rows_seen": rows_seen,
        "rows_inserted": rows_inserted,
        "missing_url_or_text": missing,
        "elapsed_sec": round(time.time() - started, 3),
    }


def build_document_store(
    inputs: list[str],
    output: Path,
    *,
    batch_size: int = 5000,
    max_shards: int = 0,
    resume: bool = True,
    overwrite: bool = False,
    progress_every: int = 1,
) -> dict[str, Any]:
    """Stream raw shards into a resumable SQLite full-document store."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_shards < 0:
        raise ValueError("max_shards must be non-negative")
    paths = expand_inputs(inputs)
    if max_shards:
        paths = paths[:max_shards]
    if overwrite:
        _remove_database(output)
    elif output.exists() and not resume:
        raise FileExistsError(f"{output} exists; use --resume or --overwrite")

    started = time.time()
    connection = _open_database(output)
    completed = _completed_shards(connection)
    reports: list[dict[str, Any]] = []
    skipped = 0
    try:
        pending = [path for path in paths if _shard_id(path) not in completed]
        skipped = len(paths) - len(pending)
        for index, path in enumerate(pending, 1):
            report = _ingest_shard(connection, path, batch_size=batch_size)
            reports.append(report)
            if progress_every > 0 and index % progress_every == 0:
                print(
                    f"[build-store] {index}/{len(pending)} {path.name} "
                    f"rows={report['rows_seen']} inserted={report['rows_inserted']}",
                    flush=True,
                )
        connection.execute("PRAGMA optimize")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        totals = connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(rows_seen),0),
                   COALESCE(SUM(rows_inserted),0),
                   COALESCE(SUM(missing_url_or_text),0)
            FROM source_shards
            """
        ).fetchone()
    finally:
        connection.close()

    report = {
        "stage": "build_document_store",
        "output": str(output.resolve()),
        "selected_shards": len(paths),
        "processed_shards": len(reports),
        "skipped_shards": skipped,
        "completed_shards": int(totals[0]),
        "rows_seen": int(totals[1]),
        "document_rows": int(totals[2]),
        "missing_url_or_text": int(totals[3]),
        "sqlite_bytes": output.stat().st_size,
        "elapsed_sec": round(time.time() - started, 3),
        "shards": reports,
    }
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def add_document_store_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", action="append", required=True, help="file, directory, or glob")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--max-shards", type=int, default=0, help="0 processes every shard")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)


def run_document_store(args: argparse.Namespace) -> dict[str, Any]:
    return build_document_store(
        args.input,
        args.output,
        batch_size=args.batch_size,
        max_shards=args.max_shards,
        resume=args.resume,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
    )
