"""Tag raw web documents with the bundled SuperGPQA++ taxonomy."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from elephantbench.client import parse_headers

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
from .synthesis import parse_json_object

CONTENT_TYPES = {
    "academic",
    "technical",
    "medical",
    "legal",
    "educational",
    "news",
    "reference",
    "commercial",
    "personal",
    "boilerplate",
    "other",
}
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def load_taxonomy(path: Path) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    compact: list[dict[str, Any]] = []
    path_strings: list[str] = []
    path_to_tuple: dict[str, tuple[str, str, str]] = {}
    for discipline in source.get("disciplines") or []:
        discipline_name = str(discipline["name"])
        fields: list[dict[str, Any]] = []
        for field in discipline.get("fields") or []:
            field_name = str(field["name"])
            subfields = [str(item["name"]) for item in field.get("subfields") or []]
            fields.append({"field": field_name, "subfields": subfields})
            for subfield in subfields:
                value = (discipline_name, field_name, subfield)
                key = " -> ".join(value)
                path_strings.append(key)
                path_to_tuple[key] = value
        compact.append({"discipline": discipline_name, "fields": fields})
    if not path_strings:
        raise ValueError(f"taxonomy contains no complete paths: {path}")
    return {
        "source": str(path.resolve()),
        "compact_json": json.dumps(compact, ensure_ascii=False, separators=(",", ":")),
        "paths": sorted(path_strings),
        "path_to_tuple": path_to_tuple,
    }


def response_schema(taxonomy: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "supergpqa_knowledge_tags",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "is_knowledge_bearing": {"type": "boolean"},
                    "content_type": {"type": "string", "enum": sorted(CONTENT_TYPES)},
                    "language": {"type": "string", "maxLength": 32},
                    "tag_paths": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "path": {"type": "string", "enum": taxonomy["paths"]},
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                                "evidence_start_sentence": {"type": "integer", "minimum": 1},
                                "evidence_end_sentence": {"type": "integer", "minimum": 1},
                            },
                            "required": [
                                "path",
                                "confidence",
                                "evidence_start_sentence",
                                "evidence_end_sentence",
                            ],
                        },
                    },
                    "reason": {"type": "string", "maxLength": 500},
                },
                "required": [
                    "is_knowledge_bearing",
                    "content_type",
                    "language",
                    "tag_paths",
                    "reason",
                ],
            },
        },
    }


def split_sentences(text: str, max_sentence_chars: int) -> list[tuple[int, str]]:
    pieces: list[str] = []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for block in (value.strip() for value in re.split(r"\n+", normalized)):
        if not block:
            continue
        for sentence in re.split(r"(?<=[。！？；;])\s*|(?<=[.!?])\s+", block):
            sentence = sentence.strip()
            if not sentence:
                continue
            if max_sentence_chars <= 0 or len(sentence) <= max_sentence_chars:
                pieces.append(sentence)
            else:
                pieces.extend(
                    sentence[start : start + max_sentence_chars].strip()
                    for start in range(0, len(sentence), max_sentence_chars)
                )
    return [(index, value) for index, value in enumerate(pieces, 1) if value]


def visible_document(
    record: dict[str, Any], max_sentence_chars: int
) -> tuple[dict[str, str], dict[int, str]]:
    sentences = split_sentences(extract_text(record), max_sentence_chars)
    sentence_map = dict(sentences)
    lines = [f"S{number:06d}: {sentence}" for number, sentence in sentences]
    return {"url": extract_url(record), "text": "\n".join(lines)}, sentence_map


def render_user_prompt(taxonomy: dict[str, Any], document: dict[str, str]) -> str:
    return (
        "Classify this document with the provided taxonomy tags.\n\n"
        "Rules:\n"
        "- A valid tag is one exact discipline -> field -> subfield path.\n"
        "- Return exactly one path: the most specific applicable label.\n"
        "- Use the text, not the URL, as primary evidence.\n"
        "- The text contains numbered sentence lines. Return the shortest inclusive sentence "
        "range that directly supports the selected path.\n"
        "- Do not invent, translate, abbreviate, or alter taxonomy labels.\n\n"
        f"TAXONOMY_JSON:\n{taxonomy['compact_json']}\n\n"
        "DOCUMENT_JSON:\n" + json.dumps(document, ensure_ascii=False, separators=(",", ":"))
    )


def validate_tag_result(
    value: dict[str, Any], taxonomy: dict[str, Any], sentence_map: dict[int, str]
) -> list[str]:
    errors: list[str] = []
    if not isinstance(value.get("is_knowledge_bearing"), bool):
        errors.append("is_knowledge_bearing must be boolean")
    if value.get("content_type") not in CONTENT_TYPES:
        errors.append("invalid content_type")
    if not isinstance(value.get("language"), str) or not value["language"].strip():
        errors.append("language must be a non-empty string")
    if not isinstance(value.get("reason"), str):
        errors.append("reason must be a string")
    paths = value.get("tag_paths")
    if not isinstance(paths, list) or len(paths) != 1:
        errors.append("tag_paths must contain exactly one item")
        paths = []
    for index, item in enumerate(paths):
        if not isinstance(item, dict):
            errors.append(f"tag_paths[{index}] must be an object")
            continue
        if item.get("path") not in taxonomy["path_to_tuple"]:
            errors.append(f"tag_paths[{index}] has an invalid taxonomy path")
        confidence = item.get("confidence")
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0.0 <= float(confidence) <= 1.0
        ):
            errors.append(f"tag_paths[{index}] confidence must be in [0, 1]")
        start, end = item.get("evidence_start_sentence"), item.get("evidence_end_sentence")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            errors.append(f"tag_paths[{index}] evidence range must contain integers")
            continue
        low, high = min(start, end), max(start, end)
        missing = [number for number in range(low, high + 1) if number not in sentence_map]
        if missing:
            errors.append(f"tag_paths[{index}] cites unavailable sentence lines")
        if high - low + 1 > max(1, len(sentence_map) // 2):
            errors.append(f"tag_paths[{index}] evidence range is too broad")
    return errors


def normalize_tag_result(
    value: dict[str, Any], taxonomy: dict[str, Any], sentence_map: dict[int, str]
) -> dict[str, Any]:
    tags: list[dict[str, Any]] = []
    for item in value["tag_paths"]:
        discipline, field, subfield = taxonomy["path_to_tuple"][item["path"]]
        low = min(item["evidence_start_sentence"], item["evidence_end_sentence"])
        high = max(item["evidence_start_sentence"], item["evidence_end_sentence"])
        tags.append(
            {
                "confidence": round(float(item["confidence"]), 6),
                "discipline": discipline,
                "field": field,
                "subfield": subfield,
                "evidence": " ".join(sentence_map[number] for number in range(low, high + 1)),
                "evidence_start_sentence": low,
                "evidence_end_sentence": high,
            }
        )
    return {
        "is_knowledge_bearing": value["is_knowledge_bearing"],
        "content_type": value["content_type"],
        "language": value["language"],
        "tags": tags,
        "reason": value["reason"],
    }


class KnowledgeTagClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float,
        retries: int,
        max_tokens: int,
        headers: dict[str, str] | None = None,
    ) -> None:
        base = base_url.rstrip("/")
        self.endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.max_tokens = max_tokens
        self.headers = headers or {}

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        correction: str = "",
        previous_content: str = "",
    ) -> tuple[str, dict[str, Any]]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if correction:
            if previous_content:
                messages.append({"role": "assistant", "content": previous_content})
            messages.append({"role": "user", "content": correction})
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "stream": False,
            "response_format": schema,
        }
        if self.model:
            payload["model"] = self.model
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        headers.update(self.headers)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                self.endpoint, data=body, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                choices = data.get("choices") or []
                content = (choices[0].get("message") or {}).get("content") if choices else None
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("response has no assistant text")
                return content, data.get("usage") or {}
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in RETRYABLE_STATUS:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(min(8.0, 2**attempt + random.random()))
        raise RuntimeError(f"knowledge tag request failed: {last_error}")


def tag_document(
    task: tuple[Path, int, dict[str, Any]],
    client: KnowledgeTagClient,
    taxonomy: dict[str, Any],
    schema: dict[str, Any],
    system_prompt: str,
    *,
    max_sentence_chars: int,
    validation_retries: int,
) -> dict[str, Any]:
    source_path, line_number, record = task
    started = time.time()
    doc_id = document_id(record)
    document, sentence_map = visible_document(record, max_sentence_chars)
    if not sentence_map:
        return {
            "document_id": doc_id,
            "status": "error",
            "error": "document text is empty",
            "url": extract_url(record),
            "tags": [],
            "source": {"file": source_path.name, "line_number": line_number},
            "elapsed_sec": round(time.time() - started, 3),
        }
    user_prompt = render_user_prompt(taxonomy, document)
    correction = ""
    previous_content = ""
    usage: dict[str, Any] = {}
    last_error = "tagger did not return a valid result"
    for attempt in range(validation_retries + 1):
        try:
            content, usage = client.complete(
                system_prompt,
                user_prompt,
                schema,
                correction,
                previous_content,
            )
            previous_content = content
            value = parse_json_object(content)
            errors = validate_tag_result(value, taxonomy, sentence_map)
            if errors:
                raise ValueError("; ".join(errors))
            normalized = normalize_tag_result(value, taxonomy, sentence_map)
            return {
                "document_id": doc_id,
                "status": "success",
                "url": extract_url(record),
                **normalized,
                "source": {"file": source_path.name, "line_number": line_number},
                "usage": usage,
                "attempts": attempt + 1,
                "elapsed_sec": round(time.time() - started, 3),
            }
        except Exception as exc:
            last_error = str(exc)
            max_range = max(1, len(sentence_map) // 2)
            correction = (
                "The previous JSON response failed local validation:\n- "
                + last_error.replace("; ", "\n- ")
                + "\nReturn the complete corrected JSON object with exactly one legal taxonomy "
                "path. "
                "Use only numbered sentence lines present in DOCUMENT_JSON.text, and limit each "
                f"inclusive evidence range to at most {max_range} sentence(s)."
            )
    return {
        "document_id": doc_id,
        "status": "error",
        "error": last_error,
        "url": extract_url(record),
        "tags": [],
        "source": {"file": source_path.name, "line_number": line_number},
        "usage": usage,
        "attempts": validation_retries + 1,
        "elapsed_sec": round(time.time() - started, 3),
    }


def _tasks(
    input_path: Path, completed: set[str], limit: int
) -> Iterator[tuple[Path, int, dict[str, Any]]]:
    for line_number, record in selected_jsonl_records(input_path, completed, limit):
        yield input_path, line_number, record


def _run_shard(
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    client: KnowledgeTagClient,
    taxonomy: dict[str, Any],
    schema: dict[str, Any],
    system_prompt: str,
    limit: int,
) -> dict[str, Any]:
    if args.overwrite and output_path.exists():
        output_path.unlink()
    completed = completed_document_ids(output_path) if args.resume else set()
    mode = "a" if output_path.exists() else "w"
    counts: Counter[str] = Counter()
    started = time.time()
    iterator = _tasks(input_path, completed, limit)
    pending: set[concurrent.futures.Future[dict[str, Any]]] = set()
    submitted = 0
    with open_text(output_path, mode) as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < max(1, args.workers * 2):
                    try:
                        task = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.add(
                        executor.submit(
                            tag_document,
                            task,
                            client,
                            taxonomy,
                            schema,
                            system_prompt,
                            max_sentence_chars=args.max_sentence_chars,
                            validation_retries=args.validation_retries,
                        )
                    )
                    submitted += 1
                if not pending:
                    continue
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    row = future.result()
                    write_jsonl_row(handle, row)
                    handle.flush()
                    counts[row["status"]] += 1
                    finished = counts["success"] + counts["error"]
                    if args.progress_every > 0 and finished % args.progress_every == 0:
                        print(
                            f"[{input_path.name}] completed={finished}/{submitted} "
                            f"success={counts['success']} error={counts['error']}",
                            flush=True,
                        )
    report = {
        "stage": "tag_supergpqa_plus",
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "resumed_successes": len(completed),
        "submitted": submitted,
        **dict(counts),
        "elapsed_sec": round(time.time() - started, 3),
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def run_knowledge_tagging(args: argparse.Namespace) -> dict[str, Any]:
    if not args.base_url:
        raise ValueError("--base-url or OPENAI_BASE_URL is required")
    if args.max_docs > 0 and args.max_docs_per_shard > 0:
        raise ValueError("use only one of --max-docs and --max-docs-per-shard")
    inputs = expand_inputs(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy(args.taxonomy)
    schema = response_schema(taxonomy)
    system_prompt = args.prompt.read_text(encoding="utf-8").strip()
    client = KnowledgeTagClient(
        args.base_url,
        args.api_key,
        args.model,
        timeout=args.timeout,
        retries=args.request_retries,
        max_tokens=args.max_tokens,
        headers=parse_headers(args.header),
    )
    reports: list[dict[str, Any]] = []
    if args.max_docs_per_shard > 0:
        jobs = [(path, args.max_docs_per_shard) for path in inputs]
    elif args.max_docs > 0:
        base, remainder = divmod(args.max_docs, len(inputs))
        jobs = [
            (path, base + (1 if index < remainder else 0))
            for index, path in enumerate(inputs)
            if base + (1 if index < remainder else 0) > 0
        ]
    else:
        jobs = [(path, 0) for path in inputs]
    for input_path, limit in jobs:
        output_path = args.output_dir / shard_output_name(input_path, "kp", args.compress)
        report = _run_shard(
            input_path,
            output_path,
            args,
            client,
            taxonomy,
            schema,
            system_prompt,
            limit,
        )
        reports.append(report)
    summary = {
        "stage": "tag_supergpqa_plus",
        "inputs": len(reports),
        "submitted": sum(int(report.get("submitted") or 0) for report in reports),
        "success": sum(int(report.get("success") or 0) for report in reports),
        "error": sum(int(report.get("error") or 0) for report in reports),
        "taxonomy": taxonomy["source"],
        "output_dir": str(args.output_dir.resolve()),
        "shards": reports,
    }
    (args.output_dir / "tag_kp.report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def add_knowledge_tagging_arguments(parser: argparse.ArgumentParser) -> None:
    package = Path(__file__).resolve().parents[1]
    parser.add_argument("--input", action="append", required=True, help="file, directory, or glob")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=package / "taxonomy" / "supergpqa_plus_taxonomy.json",
    )
    parser.add_argument(
        "--prompt", type=Path, default=package / "prompts" / "knowledge_tagging.txt"
    )
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", ""))
    parser.add_argument("--header", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--validation-retries", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-sentence-chars", type=int, default=1200)
    parser.add_argument("--max-docs", type=int, default=0, help="0 processes every document")
    parser.add_argument(
        "--max-docs-per-shard",
        type=int,
        default=0,
        help="limit each input shard independently (mutually exclusive with --max-docs)",
    )
    parser.add_argument("--compress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
