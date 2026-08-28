"""Build conflict-centered document subgraphs from judged relations."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _confidence(row: dict[str, Any]) -> float:
    try:
        return float((row.get("judgment") or {}).get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def load_relations(
    paths: Iterable[Path],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], Counter[str]]:
    """Load successful relations and build support adjacency with pair-level deduplication."""
    selected: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") not in (None, "success"):
                    counts["errors_skipped"] += 1
                    continue
                relation = str(row.get("relation") or "")
                if relation not in {"support", "conflict", "none"}:
                    counts["invalid_skipped"] += 1
                    continue
                pair_id = str(row.get("pair_id") or "")
                left = str(row.get("doc_a_id") or "")
                right = str(row.get("doc_b_id") or "")
                if not pair_id or not left or not right or left == right:
                    raise ValueError(f"{path}:{line_number}: invalid relation identifiers")
                counts[f"input_{relation}"] += 1
                current = selected.get(pair_id)
                if current is None or _confidence(row) > _confidence(current):
                    selected[pair_id] = row

    conflicts: list[dict[str, Any]] = []
    support_by_endpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair_id in sorted(selected):
        row = selected[pair_id]
        relation = str(row["relation"])
        counts[f"deduplicated_{relation}"] += 1
        if relation == "conflict":
            conflicts.append(row)
        elif relation == "support":
            support_by_endpoint[str(row["doc_a_id"])].append(row)
            support_by_endpoint[str(row["doc_b_id"])].append(row)
    for edges in support_by_endpoint.values():
        edges.sort(key=lambda row: (-_confidence(row), str(row["pair_id"])))
    return conflicts, support_by_endpoint, counts


def select_graph_specs(
    conflicts: Iterable[dict[str, Any]],
    support_by_endpoint: dict[str, list[dict[str, Any]]],
    *,
    support_per_endpoint: int = 2,
    min_support_per_endpoint: int = 1,
    max_subgraphs: int = 0,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Select deterministic conflict seeds and attach bounded support neighbors."""
    coverage: Counter[str] = Counter()
    ranked: list[tuple[int, float, str, dict[str, Any]]] = []
    for conflict in conflicts:
        left, right = str(conflict["doc_a_id"]), str(conflict["doc_b_id"])
        left_count = len(support_by_endpoint.get(left, []))
        right_count = len(support_by_endpoint.get(right, []))
        coverage["conflicts"] += 1
        if left_count:
            coverage["left_supported"] += 1
        if right_count:
            coverage["right_supported"] += 1
        if left_count and right_count:
            coverage["both_supported"] += 1
        eligible = min(left_count, right_count) >= min_support_per_endpoint
        if eligible:
            ranked.append(
                (
                    min(left_count, right_count),
                    _confidence(conflict),
                    str(conflict["pair_id"]),
                    conflict,
                )
            )
        else:
            coverage["insufficient_support"] += 1

    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    if max_subgraphs > 0:
        ranked = ranked[:max_subgraphs]

    graphs: list[dict[str, Any]] = []
    for _, _, _, conflict in ranked:
        left, right = str(conflict["doc_a_id"]), str(conflict["doc_b_id"])
        edges: list[dict[str, Any]] = []
        used: set[str] = set()
        for endpoint in (left, right):
            added = 0
            for edge in support_by_endpoint.get(endpoint, []):
                pair_id = str(edge["pair_id"])
                if pair_id in used:
                    continue
                edges.append(edge)
                used.add(pair_id)
                added += 1
                if support_per_endpoint > 0 and added >= support_per_endpoint:
                    break
        node_ids = {left, right}
        for edge in edges:
            node_ids.update((str(edge["doc_a_id"]), str(edge["doc_b_id"])))
        graphs.append(
            {
                "subgraph_id": f"subgraph-{conflict['pair_id']}",
                "seed_conflict": conflict,
                "support_edges": edges,
                "node_ids": sorted(node_ids),
            }
        )
    coverage["selected_subgraphs"] = len(graphs)
    return graphs, coverage


def fetch_documents(path: Path, doc_ids: set[str]) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)
    documents: dict[str, dict[str, Any]] = {}
    try:
        ordered = sorted(doc_ids)
        for start in range(0, len(ordered), 400):
            batch = ordered[start : start + 400]
            placeholders = ",".join("?" for _ in batch)
            query = (
                "SELECT canonical_doc_id,url,text FROM docs "
                f"WHERE canonical_doc_id IN ({placeholders})"
            )
            for doc_id, url, text in connection.execute(query, batch):
                documents[str(doc_id)] = {
                    "doc_id": str(doc_id),
                    "url": str(url or ""),
                    "text": str(text or ""),
                }
    finally:
        connection.close()
    return documents


def build_subgraphs(
    relation_paths: list[Path],
    doc_store: Path,
    output: Path,
    *,
    seed_relation_paths: list[Path] | None = None,
    support_per_endpoint: int = 2,
    min_support_per_endpoint: int = 1,
    max_subgraphs: int = 0,
) -> dict[str, Any]:
    graph_conflicts, support_by_endpoint, relation_counts = load_relations(relation_paths)
    if seed_relation_paths:
        conflicts, _, seed_counts = load_relations(seed_relation_paths)
    else:
        conflicts, seed_counts = graph_conflicts, Counter()
    specs, coverage = select_graph_specs(
        conflicts,
        support_by_endpoint,
        support_per_endpoint=support_per_endpoint,
        min_support_per_endpoint=min_support_per_endpoint,
        max_subgraphs=max_subgraphs,
    )
    all_doc_ids = {doc_id for graph in specs for doc_id in graph["node_ids"]}
    documents = fetch_documents(doc_store, all_doc_ids)
    missing = sorted(all_doc_ids - documents.keys())
    if missing:
        raise ValueError(
            f"document store is missing {len(missing)} selected documents: {missing[:5]}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for graph in specs:
            graph["documents"] = [documents[doc_id] for doc_id in graph.pop("node_ids")]
            handle.write(json.dumps(graph, ensure_ascii=False, separators=(",", ":")) + "\n")

    report = {
        "stage": "build_conflict_subgraphs",
        "relation_inputs": [str(path.resolve()) for path in relation_paths],
        "seed_relation_inputs": [
            str(path.resolve()) for path in (seed_relation_paths or relation_paths)
        ],
        "relation_counts": dict(relation_counts),
        "seed_relation_counts": dict(seed_counts),
        "coverage": dict(coverage),
        "selected_subgraphs": len(specs),
        "unique_documents": len(all_doc_ids),
        "output": str(output.resolve()),
        "config": {
            "support_per_endpoint": support_per_endpoint,
            "min_support_per_endpoint": min_support_per_endpoint,
            "max_subgraphs": max_subgraphs,
        },
    }
    output.with_suffix(output.suffix + ".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def add_subgraph_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--relations",
        type=Path,
        action="append",
        required=False,
        help="relation JSONL; repeat to merge independently classified shards",
    )
    parser.add_argument(
        "--seed-relations",
        type=Path,
        action="append",
        help="optional relation JSONL defining conflict seeds; defaults to --relations",
    )
    parser.add_argument("--doc-store", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--support-per-endpoint", type=int, default=2)
    parser.add_argument("--min-support-per-endpoint", type=int, default=1)
    parser.add_argument("--max-subgraphs", type=int, default=0, help="0 keeps every eligible seed")


def run_subgraph_build(args: argparse.Namespace) -> dict[str, Any]:
    return build_subgraphs(
        args.relations,
        args.doc_store,
        args.output,
        seed_relation_paths=args.seed_relations,
        support_per_endpoint=args.support_per_endpoint,
        min_support_per_endpoint=args.min_support_per_endpoint,
        max_subgraphs=args.max_subgraphs,
    )
