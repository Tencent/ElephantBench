from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .io import load_by_id, read_jsonl
from .metrics import label_of

RANK = {"failed": 0, "partial": 1, "complete": 2}


def load_labels(path: Path, benchmark: dict[str, dict[str, Any]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for record in read_jsonl(path):
        benchmark_id = record.get("benchmark_id")
        if benchmark_id not in benchmark:
            raise ValueError(f"{path}: unknown benchmark_id {benchmark_id!r}")
        if benchmark_id in labels:
            raise ValueError(f"{path}: duplicate benchmark_id {benchmark_id!r}")
        label = label_of(record) if record.get("status") == "success" else None
        labels[str(benchmark_id)] = label or "failed"
    return labels


def rates(labels: list[str]) -> dict[str, float | int]:
    counts = Counter(labels)
    total = len(labels)
    available = counts["complete"] + counts["partial"]
    return {
        "n": total,
        "C": counts["complete"] / total if total else 0.0,
        "P": counts["partial"] / total if total else 0.0,
        "F": counts["failed"] / total if total else 0.0,
        "K": counts["complete"] / available if available else 0.0,
    }


def oracle_report(
    benchmark: dict[str, dict[str, Any]], systems: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    if not systems:
        return []
    ids = list(benchmark)
    remaining = set(systems)
    selected: list[str] = []
    best = {benchmark_id: "failed" for benchmark_id in ids}
    curve: list[dict[str, Any]] = []
    while remaining:
        winner = max(
            sorted(remaining),
            key=lambda name: sum(
                max(RANK[best[benchmark_id]], RANK[systems[name].get(benchmark_id, "failed")])
                == RANK["complete"]
                for benchmark_id in ids
            ),
        )
        selected.append(winner)
        remaining.remove(winner)
        for benchmark_id in ids:
            candidate = systems[winner].get(benchmark_id, "failed")
            if RANK[candidate] > RANK[best[benchmark_id]]:
                best[benchmark_id] = candidate
        curve.append({"models": len(selected), "added": winner, **rates(list(best.values()))})
    return curve


def analyze(
    benchmark: dict[str, dict[str, Any]], systems: dict[str, dict[str, str]]
) -> dict[str, Any]:
    per_model = {}
    for name, labels in systems.items():
        complete_labels = {
            benchmark_id: labels.get(benchmark_id, "failed") for benchmark_id in benchmark
        }
        per_model[name] = {"overall": rates(list(complete_labels.values()))}
    return {"models": per_model, "greedy_oracle": oracle_report(benchmark, systems)}


def parse_result(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--result must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--result must be LABEL=PATH")
    return label.strip(), Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze ElephantBench model metrics and oracle coverage"
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument(
        "--result", action="append", type=parse_result, required=True, metavar="LABEL=PATH"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    benchmark = load_by_id(args.benchmark)
    systems: dict[str, dict[str, str]] = {}
    for label, path in args.result:
        if label in systems:
            raise SystemExit(f"duplicate result label {label!r}")
        systems[label] = load_labels(path, benchmark)
    report = analyze(benchmark, systems)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
