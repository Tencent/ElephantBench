"""Run a token-classification T-NER model over raw web-document shards."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import re
import string
import time
import traceback
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .source_io import (
    completed_document_ids,
    document_id,
    expand_inputs,
    extract_text,
    extract_url,
    open_text,
    selected_jsonl_records,
    shard_output_name,
    write_jsonl_row,
)

DEFAULT_MODEL_REVISION = "98a5818d383710eaf2d6f3d4a578e0b0ba98f14a"

LABEL_MAP = {
    "PER": "person",
    "PERSON": "person",
    "ORG": "organization",
    "LOC": "location",
    "GPE": "location",
    "FAC": "facility",
    "MISC": "misc",
    "NORP": "norp",
    "PRODUCT": "product",
    "EVENT": "event",
    "WORK_OF_ART": "work_of_art",
    "LAW": "law",
    "LANGUAGE": "language",
}
NAVIGATION_NOISE = {
    "about",
    "advertisement",
    "archive",
    "back",
    "blog",
    "browse",
    "click",
    "comment",
    "comments",
    "contact",
    "copyright",
    "download",
    "email",
    "facebook",
    "follow",
    "forums",
    "gallery",
    "home",
    "image",
    "images",
    "instagram",
    "login",
    "menu",
    "more",
    "next",
    "photo",
    "photos",
    "previous",
    "print",
    "privacy",
    "read",
    "rss",
    "search",
    "share",
    "subscribe",
    "terms",
    "twitter",
    "video",
    "videos",
    "view",
}
ALIAS_NORMALIZATION = {
    "u.s": "united states",
    "u.s.": "united states",
    "us": "united states",
    "u.s.a": "united states",
    "u.s.a.": "united states",
    "usa": "united states",
    "u.k": "united kingdom",
    "u.k.": "united kingdom",
    "uk": "united kingdom",
}
TWO_LETTER_WHITELIST = {"ai", "bc", "dc", "eu", "uk", "un", "us"}
TRIM_CHARS = string.whitespace + string.punctuation + "“”‘’·•…—–"


def _tagging_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "T-NER requires the 'construction' extra: pip install 'elephantbench[construction]'"
        ) from exc
    return torch, AutoModelForTokenClassification, AutoTokenizer


def normalize_entity(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).replace("\u2019", "'")
    value = re.sub(r"https?://\S+|\b[A-Za-z]+://", " ", value)
    value = re.sub(r"\s+", " ", value).strip(TRIM_CHARS)
    value = re.sub(r"'s\b", "", value, flags=re.I).strip(TRIM_CHARS).casefold()
    value = re.sub(r"\s+([,.;:/)])", r"\1", value)
    value = re.sub(r"([(])\s+", r"\1", value).strip(TRIM_CHARS)
    return ALIAS_NORMALIZATION.get(value, value)


def reasonable_entity(surface: str, normalized: str) -> bool:
    if not normalized or len(normalized) > 120 or normalized in NAVIGATION_NOISE:
        return False
    if len(normalized) < 2 or normalized.isdigit() or len(normalized.split()) > 12:
        return False
    if len(normalized) <= 2:
        compact = re.sub(r"[^A-Za-z]", "", surface)
        if normalized not in TWO_LETTER_WHITELIST and not (len(compact) == 2 and compact.isupper()):
            return False
    if re.fullmatch(r"[\W_]+", normalized) or re.fullmatch(r"\d+([:/.-]\d+)+", normalized):
        return False
    alpha = sum(character.isalpha() for character in normalized)
    if alpha == 0 or alpha / max(len(normalized), 1) < 0.35 or "\n" in surface:
        return False
    first_alpha = next((character for character in surface if character.isalpha()), "")
    if first_alpha and first_alpha.islower():
        first_token = re.split(r"\s+", surface.strip(), maxsplit=1)[0]
        tail = surface[surface.index(first_alpha) + 1 :]
        if len(first_token) <= 3 or not any(character.isupper() for character in tail):
            return False
    return True


def token_chunks(
    text: str, tokenizer: Any, max_length: int, overlap_chars: int, max_chars: int
) -> list[tuple[int, str]]:
    if not text:
        return []
    if max_chars > 0:
        text = text[:max_chars]
    usable = max(1, max_length - tokenizer.num_special_tokens_to_add(pair=False))
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets = [(start, end) for start, end in encoded.get("offset_mapping") or [] if end > start]
    if not offsets:
        return [(0, text)] if text.strip() else []
    chunks: list[tuple[int, str]] = []
    token_start = 0
    while token_start < len(offsets):
        window = offsets[token_start : token_start + usable]
        char_start, char_end = window[0][0], window[-1][1]
        if text[char_start:char_end].strip():
            chunks.append((char_start, text[char_start:char_end]))
        token_end = token_start + len(window)
        if token_end >= len(offsets):
            break
        desired = max(char_start + 1, char_end - overlap_chars)
        next_start = token_end
        for index in range(token_start + 1, token_end):
            if offsets[index][0] >= desired:
                next_start = index
                break
        token_start = next_start if next_start > token_start else token_end
    return chunks


def _flush_mention(
    output: list[dict[str, Any]],
    text: str,
    entity_type: str | None,
    start: int | None,
    end: int | None,
    chunk_start: int,
) -> None:
    if entity_type is None or start is None or end is None or end <= start:
        return
    surface = text[start:end].strip(TRIM_CHARS)
    normalized = normalize_entity(surface)
    if reasonable_entity(surface, normalized):
        output.append(
            {
                "text": surface,
                "norm": normalized,
                "label": entity_type,
                "start": chunk_start + start,
                "end": chunk_start + end,
            }
        )


def decode_mentions(
    texts: list[str],
    starts: list[int],
    offsets: Any,
    predictions: Any,
    id_to_label: dict[int, str],
    allowed_labels: set[str] | None,
) -> list[list[dict[str, Any]]]:
    decoded: list[list[dict[str, Any]]] = []
    for text, chunk_start, token_offsets, token_predictions in zip(
        texts,
        starts,
        offsets.cpu().tolist(),
        predictions.cpu().tolist(),
        strict=True,
    ):
        mentions: list[dict[str, Any]] = []
        current_type: str | None = None
        current_start: int | None = None
        current_end: int | None = None
        previous_end: int | None = None
        for (start, end), prediction in zip(token_offsets, token_predictions, strict=True):
            if start == end:
                continue
            raw_label = id_to_label[int(prediction)]
            if raw_label == "O" or "-" not in raw_label:
                _flush_mention(
                    mentions, text, current_type, current_start, current_end, chunk_start
                )
                current_type = current_start = current_end = previous_end = None
                continue
            prefix, raw_type = raw_label.split("-", 1)
            entity_type = LABEL_MAP.get(raw_type, raw_type.lower())
            if allowed_labels is not None and entity_type not in allowed_labels:
                _flush_mention(
                    mentions, text, current_type, current_start, current_end, chunk_start
                )
                current_type = current_start = current_end = previous_end = None
                continue
            adjacent = (
                current_type == entity_type
                and previous_end is not None
                and start <= previous_end + 1
            )
            starts_new = (
                current_type is None
                or current_type != entity_type
                or (prefix == "B" and not adjacent)
                or (previous_end is not None and start > previous_end + 2)
            )
            if starts_new:
                _flush_mention(
                    mentions, text, current_type, current_start, current_end, chunk_start
                )
                current_type, current_start, current_end = entity_type, start, end
            else:
                current_end = end
            previous_end = end
        _flush_mention(mentions, text, current_type, current_start, current_end, chunk_start)
        decoded.append(mentions)
    return decoded


def aggregate_mentions(
    mentions: list[dict[str, Any]], max_entities: int, max_surfaces: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for mention in mentions:
        key = (mention["label"], mention["norm"])
        item = grouped.setdefault(
            key,
            {
                "text": mention["text"],
                "norm": mention["norm"],
                "label": mention["label"],
                "count": 0,
                "first_start": mention["start"],
                "surfaces": Counter(),
            },
        )
        item["count"] += 1
        item["first_start"] = min(item["first_start"], mention["start"])
        item["surfaces"][mention["text"]] += 1
    selected = sorted(
        grouped.values(), key=lambda item: (-item["count"], item["first_start"], item["norm"])
    )[:max_entities]
    for entity in selected:
        surfaces = entity.pop("surfaces")
        entity["surface_forms"] = [
            surface
            for surface, _ in sorted(
                surfaces.items(), key=lambda item: (-item[1], len(item[0]), item[0])
            )[:max_surfaces]
        ]
        if entity["surface_forms"]:
            entity["text"] = entity["surface_forms"][0]
    return selected


def infer_chunks(
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    chunks: list[tuple[int, int, str]],
    *,
    batch_size: int,
    max_length: int,
    fp16: bool,
    allowed_labels: set[str] | None,
) -> dict[int, list[dict[str, Any]]]:
    mentions_by_document: dict[int, list[dict[str, Any]]] = defaultdict(list)
    id_to_label = {int(key): value for key, value in model.config.id2label.items()}
    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset : offset + batch_size]
        document_indices = [item[0] for item in batch]
        starts = [item[1] for item in batch]
        texts = [item[2] for item in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        token_offsets = encoded.pop("offset_mapping")
        encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
        with torch.inference_mode():
            if fp16 and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    predictions = model(**encoded).logits.argmax(dim=-1)
            else:
                predictions = model(**encoded).logits.argmax(dim=-1)
        decoded = decode_mentions(
            texts, starts, token_offsets, predictions, id_to_label, allowed_labels
        )
        for document_index, mentions in zip(document_indices, decoded, strict=True):
            mentions_by_document[document_index].extend(mentions)
    return mentions_by_document


def _run_ner_shard(
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    remaining: int,
) -> dict[str, Any]:
    if args.overwrite and output_path.exists():
        output_path.unlink()
    completed = completed_document_ids(output_path) if args.resume else set()
    allowed_labels = {
        value.strip() for value in args.allowed_labels.split(",") if value.strip()
    } or None
    mode = "a" if output_path.exists() else "w"
    counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    started = time.time()
    records: list[tuple[int, dict[str, Any]]] = []
    chunks: list[tuple[int, int, str]] = []
    submitted = 0

    with open_text(output_path, mode) as output:

        def flush() -> None:
            nonlocal records, chunks
            if not records:
                return
            mentions_by_document = infer_chunks(
                model,
                tokenizer,
                torch,
                device,
                chunks,
                batch_size=args.batch_size,
                max_length=args.max_length,
                fp16=args.fp16,
                allowed_labels=allowed_labels,
            )
            for document_index, (line_number, record) in enumerate(records):
                entities = aggregate_mentions(
                    mentions_by_document.get(document_index, []),
                    args.max_entities_per_doc,
                    args.max_surface_forms,
                )
                for entity in entities:
                    label_counts[entity["label"]] += 1
                write_jsonl_row(
                    output,
                    {
                        "document_id": document_id(record),
                        "status": "success",
                        "url": extract_url(record),
                        "entities": entities,
                        "source": {"file": input_path.name, "line_number": line_number},
                        "model": args.model_name,
                        "model_revision": args.model_revision,
                        "chunking": {
                            "max_length": args.max_length,
                            "overlap_chars": args.overlap_chars,
                            "max_chars_per_doc": args.max_chars_per_doc,
                        },
                    },
                )
                counts["success"] += 1
            output.flush()
            records, chunks = [], []

        for line_number, record in selected_jsonl_records(input_path, completed, remaining):
            document_index = len(records)
            document_chunks = token_chunks(
                extract_text(record),
                tokenizer,
                args.max_length,
                args.overlap_chars,
                args.max_chars_per_doc,
            )
            records.append((line_number, record))
            chunks.extend((document_index, start, text) for start, text in document_chunks)
            submitted += 1
            if len(records) >= args.doc_batch_size or len(chunks) >= args.chunk_batch_limit:
                flush()
            if args.progress_every > 0 and submitted % args.progress_every == 0:
                print(
                    f"[{input_path.name}] submitted={submitted} success={counts['success']}",
                    flush=True,
                )
        flush()

    report = {
        "stage": "tag_tner",
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "model": args.model_name,
        "model_revision": args.model_revision,
        "device": str(device),
        "resumed_successes": len(completed),
        "submitted": submitted,
        "success": counts["success"],
        "label_counts": dict(label_counts),
        "elapsed_sec": round(time.time() - started, 3),
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _parse_devices(device: str, devices: str) -> list[str]:
    selected = [value.strip() for value in devices.split(",") if value.strip()]
    if not selected:
        selected = [device.strip()]
    if not all(selected):
        raise ValueError("at least one non-empty device is required")
    if len(set(selected)) != len(selected):
        raise ValueError("--devices must not contain duplicates")
    return selected


def _input_jobs(
    inputs: list[Path], max_docs: int, max_docs_per_shard: int
) -> list[tuple[Path, int]]:
    if max_docs > 0 and max_docs_per_shard > 0:
        raise ValueError("use only one of --max-docs and --max-docs-per-shard")
    if max_docs_per_shard > 0:
        return [(path, max_docs_per_shard) for path in inputs]
    if max_docs <= 0:
        return [(path, 0) for path in inputs]
    base, remainder = divmod(max_docs, len(inputs))
    return [
        (path, base + (1 if index < remainder else 0))
        for index, path in enumerate(inputs)
        if base + (1 if index < remainder else 0) > 0
    ]


def _run_ner_jobs(
    args: argparse.Namespace, jobs: list[tuple[Path, int]], device_name: str
) -> list[dict[str, Any]]:
    torch, model_class, tokenizer_class = _tagging_dependencies()
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device_name}")
    device = torch.device(device_name)
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if not tokenizer.is_fast:
        raise ValueError("T-NER requires a fast tokenizer with offset mappings")
    model = model_class.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
    )
    model.eval().to(device)
    reports: list[dict[str, Any]] = []
    for input_path, limit in jobs:
        output_path = args.output_dir / shard_output_name(input_path, "ner", args.compress)
        report = _run_ner_shard(
            input_path,
            output_path,
            args,
            model,
            tokenizer,
            torch,
            device,
            limit,
        )
        reports.append(report)
    return reports


def _ner_device_worker(
    args: argparse.Namespace,
    jobs: list[tuple[Path, int]],
    device_name: str,
    result_queue: Any,
) -> None:
    try:
        result_queue.put({"device": device_name, "reports": _run_ner_jobs(args, jobs, device_name)})
    except BaseException:
        result_queue.put({"device": device_name, "error": traceback.format_exc()})


def _split_pending_jobs(
    args: argparse.Namespace, jobs: list[tuple[Path, int]]
) -> tuple[list[tuple[Path, int]], list[dict[str, Any]]]:
    pending: list[tuple[Path, int]] = []
    completed_reports: list[dict[str, Any]] = []
    for input_path, limit in jobs:
        output_path = args.output_dir / shard_output_name(input_path, "ner", args.compress)
        if args.overwrite or not args.resume or not output_path.exists():
            pending.append((input_path, limit))
            continue
        completed = completed_document_ids(output_path)
        if next(selected_jsonl_records(input_path, completed, limit), None) is not None:
            pending.append((input_path, limit))
            continue
        completed_reports.append(
            {
                "stage": "tag_tner",
                "input": str(input_path.resolve()),
                "output": str(output_path.resolve()),
                "model": args.model_name,
                "model_revision": args.model_revision,
                "device": "not_loaded",
                "resumed_successes": len(completed),
                "submitted": 0,
                "success": 0,
                "label_counts": {},
                "elapsed_sec": 0.0,
            }
        )
    return pending, completed_reports


def run_ner_tagging(args: argparse.Namespace) -> dict[str, Any]:
    inputs = expand_inputs(args.input)
    devices = _parse_devices(args.device, args.devices)
    jobs = _input_jobs(inputs, args.max_docs, args.max_docs_per_shard)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs, reports = _split_pending_jobs(args, jobs)
    assignments = [jobs[index :: len(devices)] for index in range(len(devices))]
    active = [
        (device, assigned)
        for device, assigned in zip(devices, assignments, strict=True)
        if assigned
    ]
    if len(active) == 1:
        device, assigned = active[0]
        reports.extend(_run_ner_jobs(args, assigned, device))
    elif active:
        context = mp.get_context("spawn")
        result_queue = context.Queue()
        processes = [
            context.Process(
                target=_ner_device_worker,
                args=(args, assigned, device, result_queue),
            )
            for device, assigned in active
        ]
        for process in processes:
            process.start()
        results: list[dict[str, Any]] = []
        while len(results) < len(processes):
            try:
                results.append(result_queue.get(timeout=1))
            except queue.Empty:
                if all(not process.is_alive() for process in processes):
                    break
        for process in processes:
            process.join()
        errors = [str(item["error"]) for item in results if item.get("error")]
        if len(results) != len(processes):
            errors.append(
                f"received {len(results)} worker reports for {len(processes)} device processes"
            )
        for process in processes:
            if process.exitcode not in (0, None):
                errors.append(f"device worker {process.pid} exited with {process.exitcode}")
        if errors:
            raise RuntimeError("multi-device T-NER failed:\n" + "\n".join(errors))
        for item in results:
            reports.extend(item["reports"])
    reports.sort(key=lambda item: item["input"])
    summary = {
        "stage": "tag_tner",
        "inputs": len(reports),
        "submitted": sum(int(report.get("submitted") or 0) for report in reports),
        "success": sum(int(report.get("success") or 0) for report in reports),
        "model": args.model_name,
        "model_revision": args.model_revision,
        "devices": [device for device, _ in active],
        "output_dir": str(args.output_dir.resolve()),
        "shards": reports,
    }
    (args.output_dir / "tag_ner.report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def add_ner_tagging_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", action="append", required=True, help="file, directory, or glob")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-name", default="tner/deberta-v3-large-ontonotes5")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--devices",
        default="",
        help="comma-separated devices; input shards are assigned round-robin",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--doc-batch-size", type=int, default=512)
    parser.add_argument("--chunk-batch-limit", type=int, default=4096)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--overlap-chars", type=int, default=100)
    parser.add_argument("--max-chars-per-doc", type=int, default=0)
    parser.add_argument("--max-entities-per-doc", type=int, default=64)
    parser.add_argument("--max-surface-forms", type=int, default=3)
    parser.add_argument(
        "--allowed-labels",
        default=(
            "person,organization,location,facility,product,event,work_of_art,law,language,norp,misc"
        ),
    )
    parser.add_argument("--max-docs", type=int, default=0, help="0 processes every document")
    parser.add_argument(
        "--max-docs-per-shard",
        type=int,
        default=0,
        help="limit each input shard independently (mutually exclusive with --max-docs)",
    )
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--compress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=5000)
