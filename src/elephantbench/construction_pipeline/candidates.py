"""Build bounded KP and cross-KP T-NER candidate pairs without quadratic expansion."""

from __future__ import annotations

import concurrent.futures
import hashlib
import itertools
import json
import math
import shutil
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .signals import pair_scores, stable_hash

ANCHOR_PARTITION_VERSION = 2


@dataclass(frozen=True)
class CandidateConfig:
    partitions: int = 256
    all_pairs_max_docs: int = 20
    support_top_k: int = 10
    conflict_top_k: int = 10
    rare_tokens_per_doc: int = 8
    max_token_posting: int = 200
    candidate_pool_size: int = 200
    max_bucket_docs: int = 5000
    bucket_overlap_docs: int = 64
    max_cross_cluster_docs: int = 200000
    near_duplicate_threshold: float = 0.95


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary, value)
    temporary.replace(path)


def _partition_index(value: str, partitions: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % partitions


def _kp_key(anchor: dict[str, Any]) -> str:
    return str((anchor.get("knowledge_point") or {}).get("key") or "")


def _subject_norm(anchor: dict[str, Any]) -> str:
    return str((anchor.get("subject") or {}).get("norm") or "")


def _bucket_key(anchor: dict[str, Any], route: str) -> str:
    subject, slot = _subject_norm(anchor), str(anchor.get("slot") or "")
    if route == "kp":
        return "\0".join((_kp_key(anchor), subject, slot))
    return subject


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def _pair_id(left: str, right: str) -> str:
    first, second = _pair_key(left, right)
    return f"pair-{stable_hash(first + chr(0) + second, 20)}"


def _known_pair_id_set(known_pairs: set[tuple[str, str]]) -> set[str]:
    return {_pair_id(*pair) for pair in known_pairs}


def partition_anchors(
    anchor_paths: list[Path],
    work_dir: Path,
    partitions: int,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Hash-partition prepared anchors for the KP and T-NER retrieval routes."""
    manifest_path = work_dir / "anchor_partitions.manifest.json"
    progress_path = work_dir / "anchor_partitions.progress.json"
    if manifest_path.exists() and not overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("partition_key_version") != ANCHOR_PARTITION_VERSION:
            raise ValueError(
                "candidate partition keys changed; rerun candidate retrieval with --overwrite"
            )
        return manifest
    partition_root = work_dir / "anchor_partitions"
    input_fingerprint = stable_hash(
        "\n".join(str(path.resolve()) for path in anchor_paths), length=40
    )
    progress: dict[str, Any] = {}
    if progress_path.exists() and not overwrite:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("input_fingerprint") != input_fingerprint:
            raise ValueError(
                "prepared-anchor inputs changed after partitioning began; rerun with --overwrite"
            )
        if int(progress.get("partitions") or 0) != partitions:
            raise ValueError(
                "partition count changed after partitioning began; rerun with --overwrite"
            )
        if progress.get("partition_key_version") != ANCHOR_PARTITION_VERSION:
            raise ValueError(
                "candidate partition keys changed after partitioning began; rerun with --overwrite"
            )
    elif partition_root.exists():
        shutil.rmtree(partition_root)
    if overwrite and progress_path.exists():
        progress_path.unlink()
    (partition_root / "kp").mkdir(parents=True)
    (partition_root / "ner").mkdir(parents=True)
    started = time.time()
    rows = int(progress.get("anchor_rows") or 0)
    route_rows = Counter(progress.get("route_rows") or {})
    completed_files = int(progress.get("completed_anchor_files") or 0)
    raw_offsets = progress.get("byte_offsets") or {}
    output_paths = {
        (route, part): partition_root / route / f"part_{part:04d}.jsonl"
        for route in ("kp", "ner")
        for part in range(partitions)
    }
    for path in output_paths.values():
        path.touch(exist_ok=True)
        offset = int(raw_offsets.get(str(path.relative_to(partition_root))) or 0)
        with path.open("r+b") as handle:
            handle.truncate(offset)
    with ExitStack() as stack:
        handles = {
            (route, part): stack.enter_context(output_paths[(route, part)].open("ab"))
            for route in ("kp", "ner")
            for part in range(partitions)
        }
        for path_index, path in enumerate(anchor_paths[completed_files:], completed_files + 1):
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    anchor = json.loads(line)
                    rows += 1
                    for route in ("kp", "ner"):
                        alias_scope = str((anchor.get("subject") or {}).get("alias_scope") or "all")
                        if route == "ner" and alias_scope == "kp":
                            continue
                        key = _bucket_key(anchor, route)
                        if not key.strip("\0"):
                            continue
                        part = _partition_index(key, partitions)
                        serialized = line if line.endswith("\n") else line + "\n"
                        handles[(route, part)].write(serialized.encode("utf-8"))
                        route_rows[route] += 1
            if path_index % 10 == 0 or path_index == len(anchor_paths):
                for handle in handles.values():
                    handle.flush()
                checkpoint = {
                    "stage": "partition_anchors_in_progress",
                    "partition_key_version": ANCHOR_PARTITION_VERSION,
                    "input_fingerprint": input_fingerprint,
                    "anchor_file_count": len(anchor_paths),
                    "completed_anchor_files": path_index,
                    "anchor_rows": rows,
                    "route_rows": dict(route_rows),
                    "partitions": partitions,
                    "partition_root": str(partition_root.resolve()),
                    "byte_offsets": {
                        str(output_paths[key].relative_to(partition_root)): handle.tell()
                        for key, handle in handles.items()
                    },
                }
                _write_json_atomic(progress_path, checkpoint)
            if path_index % 50 == 0 or path_index == len(anchor_paths):
                print(f"[partition] {path_index}/{len(anchor_paths)} anchor shards", flush=True)
    manifest = {
        "stage": "partition_anchors",
        "partition_key_version": ANCHOR_PARTITION_VERSION,
        "anchor_files": len(anchor_paths),
        "anchor_rows": rows,
        "route_rows": dict(route_rows),
        "partitions": partitions,
        "partition_root": str(partition_root.resolve()),
        "elapsed_sec": round(time.time() - started, 3),
    }
    _write_json_atomic(manifest_path, manifest)
    progress_path.unlink(missing_ok=True)
    return manifest


def _eligible(left: dict[str, Any], right: dict[str, Any], route: str) -> bool:
    if left["doc_id"] == right["doc_id"]:
        return False
    return route != "ner" or _kp_key(left) != _kp_key(right)


def _near_duplicate(
    left: dict[str, Any], right: dict[str, Any], scores: dict[str, Any], threshold: float
) -> bool:
    if float(scores["token_jaccard"]) < threshold:
        return False
    return set(left.get("dates") or []) == set(right.get("dates") or []) and set(
        left.get("numbers") or []
    ) == set(right.get("numbers") or [])


def _doc_ref(anchor: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": str(anchor["doc_id"]),
        "knowledge_point": anchor.get("knowledge_point") or {},
        "confidence": float(anchor.get("confidence") or 0.0),
        "evidence": str(anchor.get("evidence") or ""),
        "dates": list(anchor.get("dates") or []),
        "numbers": list(anchor.get("numbers") or []),
        "source_domain": str(anchor.get("source_domain") or ""),
    }


def _contribution(
    left: dict[str, Any],
    right: dict[str, Any],
    route: str,
    scores: dict[str, Any],
) -> dict[str, Any]:
    if str(left["doc_id"]) > str(right["doc_id"]):
        left, right = right, left
    signal = "same_kp_subject_slot" if route == "kp" else "tner_cross_kp_entity"
    subject = left.get("subject") or right.get("subject") or {}
    return {
        "pair_id": _pair_id(str(left["doc_id"]), str(right["doc_id"])),
        "doc_a": _doc_ref(left),
        "doc_b": _doc_ref(right),
        "retrieval_signal": signal,
        "subject": subject,
        "slot": str(left.get("slot") or right.get("slot") or "") if route == "kp" else "",
        "scores": scores,
    }


def _all_pair_contributions(
    members: list[dict[str, Any]], route: str, config: CandidateConfig, stats: Counter
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for left, right in itertools.combinations(members, 2):
        if not _eligible(left, right, route):
            stats["ineligible_pairs"] += 1
            continue
        scores = pair_scores(left, right)
        if _near_duplicate(left, right, scores, config.near_duplicate_threshold):
            stats["near_duplicates_filtered"] += 1
            continue
        output.append(_contribution(left, right, route, scores))
    return output


def _deterministic_ring(indices: list[int], position: int, limit: int) -> list[int]:
    if len(indices) <= 1 or limit <= 0:
        return []
    output: list[int] = []
    distance = 1
    while len(output) < min(limit, len(indices) - 1):
        for candidate in (
            (position - distance) % len(indices),
            (position + distance) % len(indices),
        ):
            value = indices[candidate]
            if value != indices[position] and value not in output:
                output.append(value)
            if len(output) >= limit:
                break
        distance += 1
    return output


def _topk_contributions(
    members: list[dict[str, Any]], route: str, config: CandidateConfig, stats: Counter
) -> list[dict[str, Any]]:
    token_postings: dict[str, list[int]] = defaultdict(list)
    token_sets = [set(member.get("tokens") or []) for member in members]
    for index, tokens in enumerate(token_sets):
        for token in tokens:
            token_postings[token].append(index)
    ordered_indices = sorted(
        range(len(members)), key=lambda idx: stable_hash(str(members[idx]["doc_id"]))
    )
    positions = {member_index: position for position, member_index in enumerate(ordered_indices)}
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for index, member in enumerate(members):
        rare_tokens = sorted(
            token_sets[index], key=lambda token: (len(token_postings[token]), token)
        )[: config.rare_tokens_per_doc]
        pool: set[int] = set()
        for token in rare_tokens:
            posting = token_postings[token]
            if len(posting) <= config.max_token_posting:
                pool.update(posting)
        pool.discard(index)
        if len(pool) > config.candidate_pool_size:
            pool = set(
                sorted(
                    pool,
                    key=lambda other: (
                        -len(token_sets[index] & token_sets[other]),
                        stable_hash(str(members[other]["doc_id"])),
                    ),
                )[: config.candidate_pool_size]
            )
        ring_limit = max(0, config.candidate_pool_size - len(pool))
        pool.update(_deterministic_ring(ordered_indices, positions[index], ring_limit))
        candidates: list[tuple[float, float, str, int, dict[str, Any]]] = []
        for other_index in pool:
            other = members[other_index]
            if not _eligible(member, other, route):
                continue
            scores = pair_scores(member, other)
            if _near_duplicate(member, other, scores, config.near_duplicate_threshold):
                stats["near_duplicates_filtered"] += 1
                continue
            candidates.append(
                (
                    float(scores["support_score"]),
                    float(scores["conflict_score"]),
                    str(other["doc_id"]),
                    other_index,
                    scores,
                )
            )
        support = sorted(candidates, key=lambda item: (-item[0], item[2]))[: config.support_top_k]
        conflict = sorted(candidates, key=lambda item: (-item[1], item[2]))[: config.conflict_top_k]
        for _, _, _, other_index, scores in [*support, *conflict]:
            other = members[other_index]
            key = _pair_key(str(member["doc_id"]), str(other["doc_id"]))
            contribution = _contribution(member, other, route, scores)
            current = selected.get(key)
            if current is None or float(contribution["scores"]["conflict_score"]) > float(
                current["scores"]["conflict_score"]
            ):
                selected[key] = contribution
    return [selected[key] for key in sorted(selected)]


def _bucket_chunks(
    members: list[dict[str, Any]], config: CandidateConfig
) -> list[list[dict[str, Any]]]:
    if len(members) <= config.max_bucket_docs:
        return [members]
    ordered = sorted(members, key=lambda row: stable_hash(str(row["doc_id"])))
    step = max(1, config.max_bucket_docs - config.bucket_overlap_docs)
    return [
        ordered[start : start + config.max_bucket_docs] for start in range(0, len(ordered), step)
    ]


def _reduce_anchor_partition(
    payload: tuple[str, str, str, dict[str, Any], list[list[str]]],
) -> dict[str, Any]:
    route, input_s, output_s, raw_config, raw_known_pairs = payload
    input_path, output_path = Path(input_s), Path(output_s)
    config = CandidateConfig(**raw_config)
    known_pairs = {_pair_key(*pair) for pair in raw_known_pairs}
    known_pair_ids = _known_pair_id_set(known_pairs)
    known_partners: dict[str, set[str]] = defaultdict(set)
    for left, right in known_pairs:
        known_partners[left].add(right)
        known_partners[right].add(left)
    blocked_known_pairs: set[str] = set()
    recovered_known_pairs: set[str] = set()
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    rows = 0
    with input_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            anchor = json.loads(line)
            rows += 1
            key = _bucket_key(anchor, route)
            doc_id = str(anchor["doc_id"])
            current = groups[key].get(doc_id)
            if current is None or float(anchor.get("confidence") or 0.0) > float(
                current.get("confidence") or 0.0
            ):
                groups[key][doc_id] = anchor
    stats: Counter = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as destination:
        for key in sorted(groups):
            members = list(groups[key].values())
            member_count = len(members)
            stats["buckets"] += 1
            stats["bucket_members"] += member_count
            stats["max_bucket_docs"] = max(stats["max_bucket_docs"], member_count)
            if member_count < 2:
                stats["singleton_buckets"] += 1
                continue
            stats["multi_document_buckets"] += 1
            raw_pairs = member_count * (member_count - 1) // 2
            stats["unbounded_pairs"] += raw_pairs
            if route == "ner":
                kp_sizes = Counter(_kp_key(member) for member in members)
                same_kp_pairs = sum(size * (size - 1) // 2 for size in kp_sizes.values())
                stats["same_kp_pairs_excluded"] += same_kp_pairs
                stats["eligible_pairs_before_bounding"] += raw_pairs - same_kp_pairs
            else:
                stats["eligible_pairs_before_bounding"] += raw_pairs
            member_ids = set(groups[key])
            for doc_id in member_ids & known_partners.keys():
                for partner in member_ids & known_partners[doc_id]:
                    left, right = _pair_key(doc_id, partner)
                    if route == "ner" and _kp_key(groups[key][left]) == _kp_key(groups[key][right]):
                        continue
                    blocked_known_pairs.add(_pair_id(left, right))
            if route == "ner" and member_count > config.max_cross_cluster_docs:
                stats["overfrequent_cross_cluster_buckets"] += 1
                stats["overfrequent_cross_cluster_pairs_skipped"] += raw_pairs
                continue
            chunks = _bucket_chunks(members, config)
            if len(chunks) > 1:
                stats["split_buckets"] += 1
                stats["bucket_chunks"] += len(chunks)
            for chunk in chunks:
                if len(chunk) <= config.all_pairs_max_docs:
                    stats["all_pair_chunks"] += 1
                    contributions = _all_pair_contributions(chunk, route, config, stats)
                else:
                    stats["topk_chunks"] += 1
                    contributions = _topk_contributions(chunk, route, config, stats)
                for contribution in contributions:
                    destination.write(_dumps(contribution) + "\n")
                    stats["pair_contributions"] += 1
                    pair_id = str(contribution["pair_id"])
                    if pair_id in known_pair_ids:
                        recovered_known_pairs.add(pair_id)
    return {
        "route": route,
        "input": str(input_path),
        "output": str(output_path),
        "anchor_rows": rows,
        "known_pairs_after_blocking": sorted(blocked_known_pairs),
        "known_pairs_recovered": sorted(recovered_known_pairs),
        **dict(stats),
    }


def reduce_anchor_partitions(
    partition_root: Path,
    work_dir: Path,
    config: CandidateConfig,
    *,
    workers: int,
    known_pairs: set[tuple[str, str]] | None = None,
    overwrite: bool = False,
) -> tuple[list[Path], dict[str, Any]]:
    manifest_path = work_dir / "route_candidates.manifest.json"
    contribution_dir = work_dir / "route_candidates"
    known_pairs = known_pairs or set()
    known_pair_ids = sorted(_known_pair_id_set(known_pairs))
    serialized_known_pairs = [list(pair) for pair in sorted(known_pairs)]
    known_fingerprint = stable_hash("\n".join(known_pair_ids), length=40)
    if manifest_path.exists() and not overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config") != asdict(config):
            raise ValueError("candidate-retrieval settings changed; rerun with --rebuild-retrieval")
        if manifest.get("known_pairs_fingerprint") != known_fingerprint:
            raise ValueError("known-pair audit input changed; rerun with --rebuild-retrieval")
        return [Path(path) for path in manifest["outputs"]], manifest
    if contribution_dir.exists():
        shutil.rmtree(contribution_dir)
    contribution_dir.mkdir(parents=True)
    payloads = []
    for route in ("kp", "ner"):
        for part in range(config.partitions):
            source = partition_root / route / f"part_{part:04d}.jsonl"
            output = contribution_dir / f"{route}_part_{part:04d}.jsonl"
            payloads.append(
                (route, str(source), str(output), asdict(config), serialized_known_pairs)
            )
    summaries: list[dict[str, Any]] = []
    started = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, summary in enumerate(
            executor.map(_reduce_anchor_partition, payloads, chunksize=1), 1
        ):
            summaries.append(summary)
            if index % 16 == 0 or index == len(payloads):
                print(f"[retrieve] {index}/{len(payloads)} route partitions", flush=True)
    outputs = [Path(summary["output"]) for summary in summaries]
    totals: Counter = Counter()
    for summary in summaries:
        for key, value in summary.items():
            if isinstance(value, int):
                if key == "max_bucket_docs":
                    totals[key] = max(totals[key], value)
                else:
                    totals[key] += value
    recovered_known_ids = {
        pair_id for summary in summaries for pair_id in summary["known_pairs_recovered"]
    }
    blocked_known_ids = {
        pair_id for summary in summaries for pair_id in summary["known_pairs_after_blocking"]
    }
    manifest = {
        "stage": "retrieve_candidates",
        "outputs": [str(path) for path in outputs],
        "totals": dict(totals),
        "elapsed_sec": round(time.time() - started, 3),
        "config": asdict(config),
        "known_pairs_fingerprint": known_fingerprint,
        "known_pair_blocking_recall": {
            "known_pairs": len(known_pair_ids),
            "recovered_pairs": len(blocked_known_ids),
            "recall": len(blocked_known_ids) / len(known_pair_ids) if known_pair_ids else None,
            "missing_pair_ids": sorted(set(known_pair_ids) - blocked_known_ids)[:20],
        },
        "known_pair_recall": {
            "known_pairs": len(known_pair_ids),
            "recovered_pairs": len(recovered_known_ids),
            "recall": len(recovered_known_ids) / len(known_pair_ids) if known_pair_ids else None,
            "missing_pair_ids": sorted(set(known_pair_ids) - recovered_known_ids)[:20],
        },
    }
    _write_json(manifest_path, manifest)
    return outputs, manifest


def partition_contributions(
    contribution_paths: list[Path],
    work_dir: Path,
    partitions: int,
    *,
    pair_sample_rate: float = 1.0,
    overwrite: bool = False,
) -> tuple[list[Path], dict[str, Any]]:
    if not 0.0 < pair_sample_rate <= 1.0:
        raise ValueError("pair_sample_rate must be in (0, 1]")
    manifest_path = work_dir / "pair_partitions.manifest.json"
    pair_root = work_dir / "pair_partitions"
    if manifest_path.exists() and not overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if float(manifest.get("pair_sample_rate") or 1.0) != pair_sample_rate:
            raise ValueError("pair sampling rate changed; rerun with --rebuild-pair-partitions")
        return [Path(path) for path in manifest["outputs"]], manifest
    if pair_root.exists():
        shutil.rmtree(pair_root)
    pair_root.mkdir(parents=True)
    outputs = [pair_root / f"part_{part:04d}.jsonl" for part in range(partitions)]
    rows_seen = 0
    rows_kept = 0
    with ExitStack() as stack:
        handles = [stack.enter_context(path.open("w", encoding="utf-8")) for path in outputs]
        for path_index, path in enumerate(contribution_paths, 1):
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = "\0".join(
                        _pair_key(str(row["doc_a"]["doc_id"]), str(row["doc_b"]["doc_id"]))
                    )
                    rows_seen += 1
                    sample_value = int(stable_hash(key, 16), 16) / float(2**64)
                    if sample_value >= pair_sample_rate:
                        continue
                    handles[_partition_index(key, partitions)].write(
                        line if line.endswith("\n") else line + "\n"
                    )
                    rows_kept += 1
            if path_index % 32 == 0 or path_index == len(contribution_paths):
                print(
                    f"[dedupe-partition] {path_index}/{len(contribution_paths)} files", flush=True
                )
    manifest = {
        "stage": "partition_pair_contributions",
        "rows_seen": rows_seen,
        "rows_kept": rows_kept,
        "pair_sample_rate": pair_sample_rate,
        "partitions": partitions,
        "outputs": [str(path) for path in outputs],
    }
    _write_json(manifest_path, manifest)
    return outputs, manifest


def _merge_state(current: dict[str, Any], row: dict[str, Any]) -> None:
    current["retrieval_signals"].add(str(row["retrieval_signal"]))
    subject = row.get("subject") or {}
    subject_norm = str(subject.get("norm") or "")
    if subject_norm:
        current["subjects"][subject_norm] = subject
    slot = str(row.get("slot") or "")
    if slot:
        current["slots"].add(slot)
    scores = row.get("scores") or {}
    if float(scores.get("support_score") or -math.inf) > float(
        current["best_support"].get("support_score") or -math.inf
    ):
        current["best_support"] = scores
    if float(scores.get("conflict_score") or -math.inf) > float(
        current["best_conflict"].get("conflict_score") or -math.inf
    ):
        current["best_conflict"] = scores


def _merge_pair_partition(payload: tuple[str, str]) -> dict[str, Any]:
    input_s, output_s = payload
    input_path, output_path = Path(input_s), Path(output_s)
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    rows = 0
    with input_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            left, right = str(row["doc_a"]["doc_id"]), str(row["doc_b"]["doc_id"])
            key = _pair_key(left, right)
            current = pairs.get(key)
            if current is None:
                current = {
                    "pair_id": _pair_id(*key),
                    "doc_a": row["doc_a"] if left == key[0] else row["doc_b"],
                    "doc_b": row["doc_b"] if right == key[1] else row["doc_a"],
                    "retrieval_signals": set(),
                    "subjects": {},
                    "slots": set(),
                    "best_support": {},
                    "best_conflict": {},
                }
                pairs[key] = current
            _merge_state(current, row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as destination:
        for key in sorted(pairs):
            current = pairs[key]
            record = {
                "pair_id": current["pair_id"],
                "doc_a": current["doc_a"],
                "doc_b": current["doc_b"],
                "retrieval_signals": sorted(current["retrieval_signals"]),
                "subjects": [current["subjects"][name] for name in sorted(current["subjects"])],
                "slots": sorted(current["slots"]),
                "retrieval_scores": {
                    "best_support": current["best_support"],
                    "best_conflict": current["best_conflict"],
                },
            }
            destination.write(_dumps(record) + "\n")
    return {
        "input": str(input_path),
        "output": str(output_path),
        "contributions": rows,
        "unique_pairs": len(pairs),
    }


def merge_pair_partitions(
    pair_parts: list[Path],
    output: Path,
    work_dir: Path,
    *,
    workers: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    merged_root = work_dir / "merged_pairs"
    if output.exists() and manifest_path.exists() and not overwrite:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if merged_root.exists():
        shutil.rmtree(merged_root)
    merged_root.mkdir(parents=True)
    payloads = [
        (str(path), str(merged_root / f"part_{index:04d}.jsonl"))
        for index, path in enumerate(pair_parts)
    ]
    summaries: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, summary in enumerate(executor.map(_merge_pair_partition, payloads), 1):
            summaries.append(summary)
            if index % 32 == 0 or index == len(payloads):
                print(f"[dedupe] {index}/{len(payloads)} pair partitions", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as destination:
        for summary in summaries:
            with Path(summary["output"]).open("r", encoding="utf-8") as source:
                shutil.copyfileobj(source, destination)
    manifest = {
        "stage": "merge_candidates",
        "output": str(output.resolve()),
        "contributions": sum(item["contributions"] for item in summaries),
        "unique_pairs": sum(item["unique_pairs"] for item in summaries),
        "pair_partitions": len(pair_parts),
        "next_stage": "full-document relation classification (not run)",
    }
    _write_json(manifest_path, manifest)
    return manifest


def _known_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    def doc_id(value: Any) -> str:
        if isinstance(value, dict):
            return str(value.get("doc_id") or value.get("canonical_doc_id") or "")
        return str(value or "")

    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    edge = (
        provenance.get("conflict_edge") if isinstance(provenance.get("conflict_edge"), dict) else {}
    )
    left = doc_id(
        row.get("doc_a")
        or row.get("src_doc_id")
        or row.get("left_doc_id")
        or edge.get("src_doc_id")
    )
    right = doc_id(
        row.get("doc_b")
        or row.get("dst_doc_id")
        or row.get("right_doc_id")
        or edge.get("dst_doc_id")
    )
    return _pair_key(left, right) if left and right and left != right else None


def evaluate_known_pair_recall(candidate_path: Path, known_pairs_path: Path) -> dict[str, Any]:
    known = load_known_pairs(known_pairs_path)
    found: set[tuple[str, str]] = set()
    found_by_signal: Counter[str] = Counter()
    with candidate_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            pair = _known_pair(row)
            if pair in known:
                found.add(pair)
                for signal in row.get("retrieval_signals") or []:
                    found_by_signal[str(signal)] += 1
    return {
        "known_pairs": len(known),
        "recovered_pairs": len(found),
        "recall": len(found) / len(known) if known else None,
        "recovered_by_signal": dict(sorted(found_by_signal.items())),
        "missing_examples": [list(pair) for pair in sorted(known - found)[:20]],
    }


def load_known_pairs(path: Path) -> set[tuple[str, str]]:
    known: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip() and (pair := _known_pair(json.loads(line))) is not None:
                known.add(pair)
    return known


def _audit_anchor_partition(
    payload: tuple[str, str, list[list[str]]],
) -> dict[str, Any]:
    route, input_s, raw_known_pairs = payload
    input_path = Path(input_s)
    known_pairs = {_pair_key(*pair) for pair in raw_known_pairs}
    known_doc_ids = {doc_id for pair in known_pairs for doc_id in pair}
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    rows = 0
    matched_rows = 0
    with input_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            rows += 1
            prefix = '{"doc_id":"'
            if line.startswith(prefix):
                doc_id = line[len(prefix) : len(prefix) + 40]
                if doc_id not in known_doc_ids:
                    continue
                anchor = json.loads(line)
            else:
                anchor = json.loads(line)
                doc_id = str(anchor.get("doc_id") or "")
                if doc_id not in known_doc_ids:
                    continue
            matched_rows += 1
            key = _bucket_key(anchor, route)
            groups[key][doc_id] = anchor
    recovered: set[tuple[str, str]] = set()
    seen_docs: set[str] = set()
    for members in groups.values():
        member_ids = set(members)
        seen_docs.update(member_ids)
        for pair in known_pairs:
            if pair[0] not in member_ids or pair[1] not in member_ids:
                continue
            if route == "ner" and _kp_key(members[pair[0]]) == _kp_key(members[pair[1]]):
                continue
            recovered.add(pair)
    return {
        "route": route,
        "input": str(input_path),
        "anchor_rows": rows,
        "matched_anchor_rows": matched_rows,
        "seen_known_docs": sorted(seen_docs),
        "recovered_pairs": [list(pair) for pair in sorted(recovered)],
        "matched_anchors": [
            {
                "doc_id": doc_id,
                "knowledge_point": _kp_key(anchor),
                "subject": _subject_norm(anchor),
                "slot": str(anchor.get("slot") or ""),
            }
            for members in groups.values()
            for doc_id, anchor in members.items()
        ],
    }


def audit_known_pair_blocking(
    partition_root: Path,
    known_pairs_path: Path,
    output: Path,
    *,
    partitions: int = 256,
    workers: int = 8,
) -> dict[str, Any]:
    """Measure the recall ceiling imposed by KP/subject/slot blocking alone."""
    known_pairs = load_known_pairs(known_pairs_path)
    serialized = [list(pair) for pair in sorted(known_pairs)]
    payloads = [
        (
            route,
            str(partition_root / route / f"part_{part:04d}.jsonl"),
            serialized,
        )
        for route in ("kp", "ner")
        for part in range(partitions)
    ]
    started = time.time()
    summaries: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, summary in enumerate(executor.map(_audit_anchor_partition, payloads), 1):
            summaries.append(summary)
            if index % 32 == 0 or index == len(payloads):
                print(f"[audit-blocking] {index}/{len(payloads)} partitions", flush=True)
    recovered = {tuple(pair) for summary in summaries for pair in summary["recovered_pairs"]}
    seen_docs = {doc_id for summary in summaries for doc_id in summary["seen_known_docs"]}
    known_docs = {doc_id for pair in known_pairs for doc_id in pair}
    anchors_by_doc: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for summary in summaries:
        for anchor in summary["matched_anchors"]:
            anchors_by_doc[str(anchor["doc_id"])].add(
                (
                    str(anchor["knowledge_point"]),
                    str(anchor["subject"]),
                    str(anchor["slot"]),
                )
            )
    reason_counts: Counter[str] = Counter()
    diagnostics: list[dict[str, Any]] = []
    for left, right in sorted(known_pairs - recovered):
        left_anchors = anchors_by_doc.get(left, set())
        right_anchors = anchors_by_doc.get(right, set())
        if not left_anchors or not right_anchors:
            reason = "endpoint_missing_anchor"
        else:
            left_subjects = {anchor[1] for anchor in left_anchors}
            right_subjects = {anchor[1] for anchor in right_anchors}
            left_slots = {anchor[2] for anchor in left_anchors}
            right_slots = {anchor[2] for anchor in right_anchors}
            if not left_subjects & right_subjects:
                reason = "no_shared_subject"
            elif not left_slots & right_slots:
                reason = "no_shared_slot"
            else:
                reason = "no_shared_subject_slot_combination"
        reason_counts[reason] += 1
        diagnostics.append(
            {
                "pair": [left, right],
                "reason": reason,
                "doc_a_anchors": [list(anchor) for anchor in sorted(left_anchors)],
                "doc_b_anchors": [list(anchor) for anchor in sorted(right_anchors)],
            }
        )
    report = {
        "stage": "audit_known_pair_blocking",
        "known_pairs_path": str(known_pairs_path.resolve()),
        "partition_root": str(partition_root.resolve()),
        "known_pairs": len(known_pairs),
        "recovered_pairs": len(recovered),
        "recall": len(recovered) / len(known_pairs) if known_pairs else None,
        "known_documents": len(known_docs),
        "known_documents_with_anchor": len(seen_docs),
        "missing_documents": sorted(known_docs - seen_docs),
        "missing_pairs": [list(pair) for pair in sorted(known_pairs - recovered)],
        "missing_pair_reason_counts": dict(sorted(reason_counts.items())),
        "missing_pair_diagnostics": diagnostics,
        "anchor_rows_scanned": sum(item["anchor_rows"] for item in summaries),
        "matched_anchor_rows": sum(item["matched_anchor_rows"] for item in summaries),
        "elapsed_sec": round(time.time() - started, 3),
    }
    _write_json_atomic(output, report)
    return report


def build_candidates(
    anchors_dir: Path,
    output: Path,
    *,
    config: CandidateConfig | None = None,
    workers: int = 8,
    work_dir: Path | None = None,
    known_pairs: Path | None = None,
    overwrite: bool = False,
    rebuild_retrieval: bool = False,
    rebuild_pair_partitions: bool = False,
    pair_sample_rate: float = 1.0,
) -> dict[str, Any]:
    """Run bounded retrieval, global deduplication, and optional recall audit."""
    config = config or CandidateConfig()
    work_dir = work_dir or output.parent / "candidate_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    anchor_paths = sorted(anchors_dir.glob("shard_*.anchors.jsonl"))
    if not anchor_paths:
        raise ValueError(f"no prepared anchor shards under {anchors_dir}")
    started = time.time()
    known_pair_set = load_known_pairs(known_pairs) if known_pairs else set()
    partition_manifest = partition_anchors(
        anchor_paths, work_dir, config.partitions, overwrite=overwrite
    )
    contribution_paths, route_manifest = reduce_anchor_partitions(
        Path(partition_manifest["partition_root"]),
        work_dir,
        config,
        workers=workers,
        known_pairs=known_pair_set,
        overwrite=overwrite or rebuild_retrieval,
    )
    pair_parts, pair_partition_manifest = partition_contributions(
        contribution_paths,
        work_dir,
        config.partitions,
        pair_sample_rate=pair_sample_rate,
        overwrite=overwrite or rebuild_retrieval or rebuild_pair_partitions,
    )
    merge_manifest = merge_pair_partitions(
        pair_parts,
        output,
        work_dir,
        workers=workers,
        overwrite=overwrite or rebuild_retrieval or rebuild_pair_partitions,
    )
    report = {
        "stage": "pre_llm_candidates_complete",
        "anchors": partition_manifest,
        "retrieval": route_manifest,
        "pair_partitioning": pair_partition_manifest,
        "merge": merge_manifest,
        "elapsed_sec": round(time.time() - started, 3),
        "config": asdict(config),
        "pair_sample_rate": pair_sample_rate,
        "known_pair_recall": None,
        "next_stage": "full-document relation classification (awaiting service configuration)",
    }
    if known_pairs:
        report["known_pair_recall"] = evaluate_known_pair_recall(output, known_pairs)
    _write_json(output.parent / "candidate_report.json", report)
    return report
