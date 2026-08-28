from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .io import load_by_id, read_jsonl

ALIASES = {
    "complete": "complete",
    "full": "complete",
    "full_credit": "complete",
    "partial": "partial",
    "partial_credit": "partial",
    "failed": "failed",
    "failure": "failed",
    "no_credit": "failed",
}


def label_of(record: dict[str, Any]) -> str | None:
    judgment = record.get("judgment")
    if isinstance(judgment, dict):
        return ALIASES.get(str(judgment.get("label", "")).lower())
    for field in ("effective_grade", "judge_grade", "grade"):
        grade = record.get(field)
        if isinstance(grade, dict):
            label = ALIASES.get(str(grade.get("credit", "")).lower())
            if label:
                return label
    return None


def summarize(
    benchmark: dict[str, dict[str, Any]], results: list[dict[str, Any]]
) -> dict[str, Any]:
    by_id: dict[str, dict[str, Any]] = {}
    for record in results:
        benchmark_id = record.get("benchmark_id")
        if benchmark_id not in benchmark:
            raise ValueError(f"result ID absent from benchmark: {benchmark_id!r}")
        if benchmark_id in by_id:
            raise ValueError(f"duplicate result ID: {benchmark_id!r}")
        by_id[str(benchmark_id)] = record

    counts: Counter[str] = Counter()
    for benchmark_id in benchmark:
        record = by_id.get(benchmark_id)
        if record is None or record.get("status") != "success":
            counts["failed"] += 1
            counts["missing_or_error"] += 1
            continue
        label = label_of(record)
        if label is None:
            counts["failed"] += 1
            counts["ungraded"] += 1
        else:
            counts[label] += 1

    total = len(benchmark)
    complete, partial, failed = counts["complete"], counts["partial"], counts["failed"]
    if complete + partial + failed != total:
        raise AssertionError("C/P/F counts must sum to the benchmark size")
    available = complete + partial
    return {
        "n": total,
        "counts": {"complete": complete, "partial": partial, "failed": failed},
        "rates": {
            "C": complete / total,
            "P": partial / total,
            "F": failed / total,
            "K": complete / available if available else 0.0,
        },
        "diagnostics": {
            "result_records": len(by_id),
            "missing_or_error": counts["missing_or_error"],
            "ungraded": counts["ungraded"],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute ElephantBench C/P/F/K metrics")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = summarize(load_by_id(args.benchmark), list(read_jsonl(args.results)))
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
