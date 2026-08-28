"""Export externally verified conflict groups as two benchmark questions each."""

from __future__ import annotations

import argparse
import json
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from elephantbench.schema import validate_item


def _read_by_subgraph(path: Path) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            subgraph_id = str(row.get("subgraph_id") or "")
            if subgraph_id:
                selected[subgraph_id] = row
    return selected


def export_records(
    synthesis_path: Path,
    source_validation_path: Path,
    verification_path: Path,
    output: Path,
) -> dict[str, Any]:
    synthesis = _read_by_subgraph(synthesis_path)
    source_validation = _read_by_subgraph(source_validation_path)
    verification = _read_by_subgraph(verification_path)
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for subgraph_id in sorted(verification):
        reviewed = verification[subgraph_id]
        if reviewed.get("status") != "success":
            counts["verification_error"] += 1
            continue
        decision = reviewed.get("verification") or {}
        if decision.get("keep") is not True or decision.get("verdict") != "verified":
            counts["verification_rejected"] += 1
            continue
        source_review = source_validation.get(subgraph_id)
        source_decision = (source_review or {}).get("validation") or {}
        if (
            not source_review
            or source_review.get("status") != "success"
            or source_decision.get("keep") is not True
            or source_decision.get("verdict") != "verified"
        ):
            counts["source_validation_rejected"] += 1
            continue
        synthesized = synthesis.get(subgraph_id)
        if synthesized is None:
            raise ValueError(f"{subgraph_id}: verification has no matching synthesis row")
        qa = synthesized.get("qa") or {}
        if synthesized.get("status") != "success" or qa.get("keep") is not True:
            raise ValueError(f"{subgraph_id}: verification points to unusable synthesis output")
        questions = {
            str(row.get("formulation") or ""): str(row.get("question") or "")
            for row in qa.get("questions") or []
            if isinstance(row, dict)
        }
        if set(questions) != {"named_entity", "clue_based"}:
            raise ValueError(f"{subgraph_id}: synthesis does not contain the two formulations")
        group_id = str(uuid.uuid4())
        shared = {
            "gold_answers": [
                {"value": str(answer.get("value") or "")} for answer in qa.get("gold_answers") or []
            ],
            "preferred_answer": str(qa.get("preferred_answer") or ""),
        }
        for formulation in ("named_entity", "clue_based"):
            benchmark_id = str(uuid.uuid4())
            record = {
                "benchmark_id": benchmark_id,
                "item_group_id": group_id,
                "eval": {
                    "question": questions[formulation],
                    **shared,
                },
            }
            errors = validate_item(record)
            if errors:
                raise ValueError(f"{subgraph_id}: invalid exported record: {errors}")
            if benchmark_id in seen_ids:
                raise ValueError(f"duplicate benchmark ID: {benchmark_id}")
            seen_ids.add(benchmark_id)
            records.append(record)
        counts["groups_exported"] += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {
        "stage": "export_verified_benchmark",
        "synthesis_rows": len(synthesis),
        "source_validation_rows": len(source_validation),
        "verification_rows": len(verification),
        **dict(counts),
        "questions_exported": len(records),
        "output": str(output.resolve()),
    }
    output.with_suffix(output.suffix + ".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def add_export_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--synthesis", type=Path)
    parser.add_argument("--source-validation", type=Path)
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--output", type=Path)


def run_export(args: argparse.Namespace) -> dict[str, Any]:
    return export_records(
        args.synthesis,
        args.source_validation,
        args.verification,
        args.output,
    )
