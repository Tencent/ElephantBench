from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number}: expected a JSON object")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def load_by_id(path: str | Path, id_field: str = "benchmark_id") -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        record_id = record.get(id_field)
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{path}: missing non-empty {id_field}")
        if record_id in records:
            raise ValueError(f"{path}: duplicate {id_field} {record_id!r}")
        records[record_id] = record
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def completed_ids(path: str | Path) -> set[str]:
    target = Path(path)
    if not target.exists():
        return set()
    return {
        str(record["benchmark_id"])
        for record in read_jsonl(target)
        if record.get("status") in {"success", "error"} and record.get("benchmark_id")
    }
