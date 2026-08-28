from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from .io import read_jsonl
from .schema import validate_item


def validate_dataset(path: Path) -> dict[str, object]:
    ids: set[str] = set()
    item_groups: Counter[str] = Counter()
    normalized_questions: dict[str, int] = {}
    errors: list[str] = []
    count = 0
    for line_number, record in enumerate(read_jsonl(path), 1):
        count += 1
        benchmark_id = record.get("benchmark_id")
        if isinstance(benchmark_id, str):
            if benchmark_id in ids:
                errors.append(f"line {line_number}: duplicate benchmark_id {benchmark_id!r}")
            ids.add(benchmark_id)
        item_group_id = record.get("item_group_id")
        if isinstance(item_group_id, str) and item_group_id:
            item_groups[item_group_id] += 1
        for error in validate_item(record):
            errors.append(f"line {line_number} ({benchmark_id!r}): {error}")
        evaluation = record.get("eval") or {}
        question = evaluation.get("question")
        if isinstance(question, str) and question.strip():
            normalized_question = re.sub(r"\s+", " ", question).strip().casefold()
            previous_line = normalized_questions.get(normalized_question)
            if previous_line is not None:
                errors.append(
                    f"line {line_number}: duplicate normalized question "
                    f"(first seen on line {previous_line})"
                )
            else:
                normalized_questions[normalized_question] = line_number
    if count == 0:
        errors.append("dataset contains no records")
    singleton_groups = sorted(group_id for group_id, size in item_groups.items() if size < 2)
    for group_id in singleton_groups:
        errors.append(f"item_group_id {group_id!r} contains fewer than two records")
    return {
        "valid": not errors,
        "records": count,
        "unique_ids": len(ids),
        "unique_questions": len(normalized_questions),
        "item_groups": len(item_groups),
        "singleton_groups": len(singleton_groups),
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an ElephantBench dataset")
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)
    report = validate_dataset(args.dataset)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
