"""Targeted audits for construction filters using verified document pairs."""

from __future__ import annotations

import concurrent.futures
import json
import re
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .candidates import load_known_pairs
from .prepare import PreparationConfig, discover_shard_pairs, prepare_document

DOC_ID_RE = re.compile(r'"document_id"\s*:\s*"([0-9a-f]{40})"')


def _selected_rows(path: Path, doc_ids: set[str]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            match = DOC_ID_RE.search(line, 0, min(len(line), 120))
            if match is None or match.group(1) not in doc_ids:
                continue
            row = json.loads(line)
            selected[match.group(1)] = row
    return selected


def _scan_known_shard(
    payload: tuple[str, str, str, list[str]],
) -> dict[str, Any]:
    shard, kp_s, ner_s, raw_doc_ids = payload
    doc_ids = set(raw_doc_ids)
    kp_rows = _selected_rows(Path(kp_s), doc_ids)
    ner_rows = _selected_rows(Path(ner_s), doc_ids)
    return {"shard": shard, "kp_rows": kp_rows, "ner_rows": ner_rows}


def audit_preparation_filters(
    kp_dir: Path,
    ner_dir: Path,
    known_pairs_path: Path,
    output: Path,
    *,
    config: PreparationConfig | None = None,
    workers: int = 8,
) -> dict[str, Any]:
    """Apply preparation filters only to documents in a verified-pair audit set."""
    config = config or PreparationConfig()
    known_pairs = load_known_pairs(known_pairs_path)
    known_doc_ids = {doc_id for pair in known_pairs for doc_id in pair}
    payloads = [
        (shard, str(kp_path), str(ner_path), sorted(known_doc_ids))
        for shard, kp_path, ner_path in discover_shard_pairs(kp_dir, ner_dir)
    ]
    started = time.time()
    kp_rows: dict[str, dict[str, Any]] = {}
    ner_rows: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, summary in enumerate(executor.map(_scan_known_shard, payloads), 1):
            kp_rows.update(summary["kp_rows"])
            ner_rows.update(summary["ner_rows"])
            if index % 50 == 0 or index == len(payloads):
                print(f"[audit-prepare] {index}/{len(payloads)} shards", flush=True)
    anchors_by_doc: dict[str, list[dict[str, Any]]] = {}
    filter_reasons: Counter[str] = Counter()
    for doc_id in sorted(known_doc_ids):
        row = kp_rows.get(doc_id)
        if row is None:
            filter_reasons["missing_kp_row"] += 1
            anchors_by_doc[doc_id] = []
            continue
        ner_row = ner_rows.get(doc_id) or {}
        entities = ner_row.get("entities") if isinstance(ner_row.get("entities"), list) else []
        anchors, reason = prepare_document(row, entities, config)
        anchors_by_doc[doc_id] = anchors
        if reason:
            filter_reasons[reason] += 1
    recovered: set[tuple[str, str]] = set()
    missing_reasons: Counter[str] = Counter()
    diagnostics: list[dict[str, Any]] = []
    for left, right in sorted(known_pairs):
        left_anchors = anchors_by_doc.get(left, [])
        right_anchors = anchors_by_doc.get(right, [])
        left_keys = {
            (
                str(anchor["knowledge_point"]["key"]),
                str(anchor["subject"]["norm"]),
                str(anchor["slot"]),
                str(anchor["subject"].get("alias_scope") or "all"),
            )
            for anchor in left_anchors
        }
        right_keys = {
            (
                str(anchor["knowledge_point"]["key"]),
                str(anchor["subject"]["norm"]),
                str(anchor["slot"]),
                str(anchor["subject"].get("alias_scope") or "all"),
            )
            for anchor in right_anchors
        }
        if any(
            left_key[1] == right_key[1]
            and left_key[2] == right_key[2]
            and (left_key[0] == right_key[0] or (left_key[3] == "all" and right_key[3] == "all"))
            for left_key in left_keys
            for right_key in right_keys
        ):
            recovered.add((left, right))
            continue
        if not left_anchors or not right_anchors:
            reason = "endpoint_missing_anchor"
        elif not {key[1] for key in left_keys} & {key[1] for key in right_keys}:
            reason = "no_shared_subject"
        elif not {key[2] for key in left_keys} & {key[2] for key in right_keys}:
            reason = "no_shared_slot"
        else:
            reason = "no_shared_subject_slot_combination"
        missing_reasons[reason] += 1
        diagnostics.append(
            {
                "pair": [left, right],
                "reason": reason,
                "doc_a_keys": [list(key) for key in sorted(left_keys)],
                "doc_b_keys": [list(key) for key in sorted(right_keys)],
            }
        )
    report = {
        "stage": "audit_preparation_filters",
        "known_pairs_path": str(known_pairs_path.resolve()),
        "kp_dir": str(kp_dir.resolve()),
        "ner_dir": str(ner_dir.resolve()),
        "config": asdict(config),
        "known_pairs": len(known_pairs),
        "recovered_pairs": len(recovered),
        "recall": len(recovered) / len(known_pairs) if known_pairs else None,
        "known_documents": len(known_doc_ids),
        "kp_rows_found": len(kp_rows),
        "ner_rows_found": len(ner_rows),
        "documents_with_anchors": sum(bool(value) for value in anchors_by_doc.values()),
        "document_filter_reasons": dict(sorted(filter_reasons.items())),
        "missing_pair_reason_counts": dict(sorted(missing_reasons.items())),
        "missing_pair_diagnostics": diagnostics,
        "elapsed_sec": round(time.time() - started, 3),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
