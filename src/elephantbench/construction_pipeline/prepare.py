"""Prepare one-label, subject--slot anchors from KP and T-NER shards."""

from __future__ import annotations

import concurrent.futures
import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .signals import (
    content_tokens,
    extract_dates,
    extract_numbers,
    extract_slots,
    knowledge_point_key,
    normalize_space,
    select_primary_tag,
    select_subjects,
    source_domain,
)
from .source_io import iter_jsonl

SHARD_RE = re.compile(r"shard_(\d{8})")


@dataclass(frozen=True)
class PreparationConfig:
    min_confidence: float = 0.9
    min_evidence_chars: int = 60
    max_subjects: int = 2
    max_tokens: int = 64
    language: str = "en"
    require_knowledge_bearing: bool = False
    subject_aliases: bool = False


def _subject_variants(subject: dict[str, Any], enabled: bool) -> list[dict[str, Any]]:
    canonical = str(subject.get("norm") or "")
    variants = [dict(subject, canonical_norm=canonical, alias_scope="all")]
    if not enabled:
        return variants
    label = str(subject.get("label") or "")
    tokens = canonical.replace("’", "'").split()
    if label == "person" and 2 <= len(tokens) <= 6:
        surname = tokens[-1].removesuffix("'s").strip("-' ")
        if len(surname) >= 4 and surname != canonical:
            variants.append(
                dict(
                    subject,
                    text=surname,
                    norm=surname,
                    canonical_norm=canonical,
                    alias_kind="person_surname",
                    alias_scope="kp",
                )
            )
    if label == "organization" and 2 <= len(tokens) <= 8:
        acronym = "".join(token[0] for token in tokens if token and token[0].isalnum())
        if 3 <= len(acronym) <= 8 and acronym != canonical:
            variants.append(
                dict(
                    subject,
                    text=acronym.upper(),
                    norm=acronym,
                    canonical_norm=canonical,
                    alias_kind="organization_acronym",
                    alias_scope="all",
                )
            )
    return variants


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _shard_id(path: Path) -> str:
    match = SHARD_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot recover shard id from {path}")
    return match.group(1)


def discover_shard_pairs(kp_dir: Path, ner_dir: Path) -> list[tuple[str, Path, Path]]:
    def discover(directory: Path) -> dict[str, Path]:
        shards: dict[str, Path] = {}
        for path in sorted(directory.iterdir()):
            if not path.is_file() or not (
                path.name.endswith(".jsonl") or path.name.endswith(".jsonl.zstd")
            ):
                continue
            shard = _shard_id(path)
            if shard in shards:
                raise ValueError(
                    f"duplicate shard {shard} under {directory}: {shards[shard].name}, {path.name}"
                )
            shards[shard] = path
        return shards

    kp = discover(kp_dir)
    ner = discover(ner_dir)
    missing_ner = sorted(set(kp) - set(ner))
    missing_kp = sorted(set(ner) - set(kp))
    if missing_ner or missing_kp:
        raise ValueError(
            f"KP/T-NER shard mismatch: missing_ner={missing_ner[:5]}, missing_kp={missing_kp[:5]}"
        )
    if not kp:
        raise ValueError(f"no JSONL or JSONL.zstd shards found under {kp_dir} and {ner_dir}")
    return [(shard, kp[shard], ner[shard]) for shard in sorted(kp)]


def _load_ner(path: Path) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    for line_number, row in iter_jsonl(path):
        doc_id = str(row.get("document_id") or "")
        if not doc_id:
            raise ValueError(f"{path}:{line_number}: missing document_id")
        records[doc_id] = row.get("entities") if isinstance(row.get("entities"), list) else []
    return records


def _load_domains(metadata_db: str, doc_ids: set[str]) -> dict[str, str]:
    if not metadata_db or not doc_ids:
        return {}
    connection = sqlite3.connect(f"file:{metadata_db}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    tables = {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    if "doc_index" in tables:
        table, id_column = "doc_index", "doc_id"
    elif "docs" in tables:
        table, id_column = "docs", "canonical_doc_id"
    else:
        connection.close()
        raise ValueError(f"{metadata_db}: expected a doc_index or docs table")
    domains: dict[str, str] = {}
    ordered = sorted(doc_ids)
    for offset in range(0, len(ordered), 500):
        batch = ordered[offset : offset + 500]
        placeholders = ",".join("?" for _ in batch)
        query = f"SELECT {id_column}, url FROM {table} WHERE {id_column} IN ({placeholders})"
        for doc_id, url in connection.execute(query, batch):
            domains[str(doc_id)] = source_domain(str(url or ""))
    connection.close()
    return domains


def prepare_document(
    row: dict[str, Any],
    entities: list[dict[str, Any]],
    config: PreparationConfig,
) -> tuple[list[dict[str, Any]], str | None]:
    doc_id = str(row.get("document_id") or row.get("doc_id") or "")
    if not doc_id:
        return [], "missing_doc_id"
    if config.require_knowledge_bearing and row.get("is_knowledge_bearing") is not True:
        return [], "not_knowledge_bearing"
    language = str(row.get("language") or "")
    if config.language and language and language != config.language:
        return [], "wrong_language"
    tag = select_primary_tag(row.get("tags"))
    if tag is None:
        return [], "missing_tag"
    confidence = float(tag.get("confidence") or 0.0)
    if confidence < config.min_confidence:
        return [], "low_confidence"
    evidence = normalize_space(str(tag.get("evidence") or ""))
    if len(evidence) < config.min_evidence_chars:
        return [], "short_evidence"
    slots = extract_slots(evidence)
    if not slots:
        return [], "missing_slot"
    subjects = select_subjects(evidence, entities, config.max_subjects)
    if not subjects:
        return [], "missing_subject"
    dates = extract_dates(evidence)
    numbers = extract_numbers(evidence)
    tokens = content_tokens(evidence, config.max_tokens)
    kp = {
        "key": knowledge_point_key(tag),
        "discipline": str(tag.get("discipline") or ""),
        "field": str(tag.get("field") or ""),
        "subfield": str(tag.get("subfield") or ""),
    }
    anchors: list[dict[str, Any]] = []
    for subject in subjects:
        for subject_variant in _subject_variants(subject, config.subject_aliases):
            for slot in slots:
                anchors.append(
                    {
                        "doc_id": doc_id,
                        "knowledge_point": kp,
                        "confidence": round(confidence, 6),
                        "subject": subject_variant,
                        "slot": slot,
                        "evidence": evidence,
                        "dates": dates,
                        "numbers": numbers,
                        "tokens": tokens,
                        "source_domain": "",
                    }
                )
    return anchors, None


def _prepare_shard(payload: tuple[str, str, str, str, str, dict[str, Any], bool]) -> dict[str, Any]:
    shard_id, kp_path_s, ner_path_s, output_s, doc_store, raw_config, overwrite = payload
    kp_path, ner_path, output = Path(kp_path_s), Path(ner_path_s), Path(output_s)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() and manifest_path.exists() and not overwrite:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    started = time.time()
    config = PreparationConfig(**raw_config)
    ner = _load_ner(ner_path)
    anchors: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    rows = 0
    missing_ner = 0
    for _, row in iter_jsonl(kp_path):
        rows += 1
        doc_id = str(row.get("document_id") or row.get("doc_id") or "")
        doc_entities = ner.get(doc_id)
        if doc_entities is None:
            missing_ner += 1
            doc_entities = []
        prepared, reason = prepare_document(row, doc_entities, config)
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        anchors.extend(prepared)
    domains = _load_domains(doc_store, {str(anchor["doc_id"]) for anchor in anchors})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for anchor in anchors:
            anchor["source_domain"] = domains.get(str(anchor["doc_id"]), "")
            handle.write(_dump_json(anchor) + "\n")
    manifest = {
        "shard_id": shard_id,
        "kp_path": str(kp_path.resolve()),
        "ner_path": str(ner_path.resolve()),
        "output": str(output.resolve()),
        "documents_seen": rows,
        "documents_with_anchors": len({anchor["doc_id"] for anchor in anchors}),
        "anchors_written": len(anchors),
        "documents_missing_ner_row": missing_ner,
        "filtered": dict(sorted(reason_counts.items())),
        "elapsed_sec": round(time.time() - started, 3),
        "config": asdict(config),
    }
    _write_json(manifest_path, manifest)
    return manifest


def prepare_anchors(
    kp_dir: Path,
    ner_dir: Path,
    output_dir: Path,
    *,
    config: PreparationConfig | None = None,
    workers: int = 8,
    doc_store: Path | None = None,
    overwrite: bool = False,
    max_shards: int = 0,
) -> dict[str, Any]:
    """Prepare shard-local anchor JSONL files and return an aggregate manifest."""
    config = config or PreparationConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = discover_shard_pairs(kp_dir, ner_dir)
    if max_shards > 0:
        pairs = pairs[:max_shards]
    payloads = [
        (
            shard,
            str(kp_path),
            str(ner_path),
            str(output_dir / f"shard_{shard}.anchors.jsonl"),
            str(doc_store.resolve()) if doc_store else "",
            asdict(config),
            overwrite,
        )
        for shard, kp_path, ner_path in pairs
    ]
    summaries: list[dict[str, Any]] = []
    started = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, summary in enumerate(executor.map(_prepare_shard, payloads, chunksize=1), 1):
            summaries.append(summary)
            if index % 10 == 0 or index == len(payloads):
                print(
                    f"[prepare] {index}/{len(payloads)} shards; "
                    f"anchors={sum(item['anchors_written'] for item in summaries):,}",
                    flush=True,
                )
    manifest = {
        "stage": "prepare_anchors",
        "kp_dir": str(kp_dir.resolve()),
        "ner_dir": str(ner_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "shards": len(summaries),
        "documents_seen": sum(item["documents_seen"] for item in summaries),
        "documents_with_anchors_sum": sum(item["documents_with_anchors"] for item in summaries),
        "anchors_written": sum(item["anchors_written"] for item in summaries),
        "documents_missing_ner_row": sum(item["documents_missing_ner_row"] for item in summaries),
        "filtered": {},
        "elapsed_sec": round(time.time() - started, 3),
        "config": asdict(config),
        "max_shards": max_shards,
    }
    for summary in summaries:
        for reason, count in summary["filtered"].items():
            manifest["filtered"][reason] = manifest["filtered"].get(reason, 0) + int(count)
    manifest["filtered"] = dict(sorted(manifest["filtered"].items()))
    _write_json(output_dir / "manifest.json", manifest)
    return manifest
