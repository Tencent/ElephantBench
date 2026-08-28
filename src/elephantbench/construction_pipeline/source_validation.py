"""Validate synthesized answers against every full document in a conflict subgraph."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from elephantbench.client import OpenAIChatClient, parse_headers

from .synthesis import parse_json_object, validate_qa


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().casefold()


def validate_source_result(
    result: dict[str, Any], qa: dict[str, Any], subgraph: dict[str, Any]
) -> list[str]:
    """Check coverage, identifiers, and verbatim evidence without external knowledge."""
    errors: list[str] = []
    if not isinstance(result.get("keep"), bool):
        errors.append("keep must be boolean")
    if result.get("verdict") not in {"verified", "rejected"}:
        errors.append("verdict must be verified or rejected")
    for key in ("same_subject", "same_attribute", "same_fact_context", "qa_valid"):
        if not isinstance(result.get(key), bool):
            errors.append(f"{key} must be boolean")
    if not isinstance(result.get("reason"), str) or not result["reason"].strip():
        errors.append("reason must be a non-empty string")

    expected = {
        _normalize(str(answer.get("value") or "")): str(answer.get("value") or "")
        for answer in qa.get("gold_answers") or []
    }
    documents = {
        str(document.get("doc_id") or ""): _normalize(str(document.get("text") or ""))
        for document in subgraph.get("documents") or []
        if isinstance(document, dict) and document.get("doc_id")
    }
    checks = result.get("answer_verifications")
    if not isinstance(checks, list):
        errors.append("answer_verifications must be a list")
        checks = []
    seen: set[str] = set()
    all_supported = True
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            errors.append(f"answer verification {index} must be an object")
            all_supported = False
            continue
        normalized = _normalize(str(check.get("value") or ""))
        if normalized not in expected:
            errors.append(f"answer verification {index} has an unknown value")
        if normalized in seen:
            errors.append(f"answer verification {index} duplicates a value")
        seen.add(normalized)
        supported = check.get("supported") is True
        if not isinstance(check.get("supported"), bool):
            errors.append(f"answer verification {index} supported must be boolean")
        all_supported = all_supported and supported
        spans = check.get("evidence_spans")
        if not isinstance(spans, list):
            errors.append(f"answer verification {index} evidence_spans must be a list")
            spans = []
        if supported and not spans:
            errors.append(f"answer verification {index} claims support without evidence")
        for span in spans:
            if not isinstance(span, dict):
                errors.append(f"answer verification {index} has an invalid evidence span")
                continue
            doc_id = str(span.get("doc_id") or "")
            quote = _normalize(str(span.get("quote") or ""))
            if doc_id not in documents:
                errors.append(f"answer verification {index} cites an unknown document")
            elif not quote or quote not in documents[doc_id]:
                errors.append(f"answer verification {index} evidence is not verbatim")
        if not isinstance(check.get("reason"), str) or not check["reason"].strip():
            errors.append(f"answer verification {index} reason must be non-empty")

    if seen != set(expected):
        errors.append("answer_verifications must cover every gold answer exactly once")
    if result.get("keep") is True:
        if result.get("verdict") != "verified":
            errors.append("keep=true requires verdict=verified")
        if not all(
            result.get(key) is True
            for key in ("same_subject", "same_attribute", "same_fact_context", "qa_valid")
        ):
            errors.append("keep=true requires every comparability and QA check to pass")
        if not all_supported or seen != set(expected):
            errors.append("keep=true requires full-document support for every answer")
    elif result.get("verdict") == "verified":
        errors.append("verdict=verified requires keep=true")
    return errors


def _load_by_subgraph(path: Path, *, allow_retries: bool = False) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            subgraph_id = str(row.get("subgraph_id") or "")
            if not subgraph_id:
                raise ValueError(f"{path}:{line_number}: missing subgraph_id")
            if subgraph_id in selected and not allow_retries:
                raise ValueError(f"{path}:{line_number}: duplicate subgraph_id {subgraph_id}")
            selected[subgraph_id] = row
    return selected


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "success" and row.get("subgraph_id"):
                completed.add(str(row["subgraph_id"]))
    return completed


def validate_one(
    synthesis: dict[str, Any],
    subgraph: dict[str, Any],
    client: OpenAIChatClient,
    system_prompt: str,
    max_tokens: int,
    retries: int,
) -> dict[str, Any]:
    started = time.time()
    subgraph_id = str(synthesis["subgraph_id"])
    qa = synthesis["qa"]
    qa_errors = validate_qa(qa, subgraph)
    if qa_errors:
        return {
            "subgraph_id": subgraph_id,
            "status": "error",
            "error": "synthesis/subgraph mismatch: " + "; ".join(qa_errors),
            "attempts": 0,
            "elapsed_sec": round(time.time() - started, 3),
        }
    payload = {
        "qa": qa,
        "seed_conflict": subgraph.get("seed_conflict"),
        "support_edges": subgraph.get("support_edges") or [],
        "documents": subgraph.get("documents") or [],
    }
    prompt = "Validate this synthesized QA record against the supplied documents:\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )
    last_error = "source validator did not return a valid result"
    usage: dict[str, Any] = {}
    for attempt in range(retries + 1):
        try:
            response = client.chat(system_prompt, prompt, temperature=0.0, max_tokens=max_tokens)
            if response.get("finish_reason") not in (None, "stop"):
                raise ValueError(f"unexpected finish_reason={response.get('finish_reason')!r}")
            usage = response.get("usage") or {}
            result = parse_json_object(str(response.get("content") or ""))
            errors = validate_source_result(result, qa, subgraph)
            if errors:
                raise ValueError("; ".join(errors))
            return {
                "subgraph_id": subgraph_id,
                "status": "success",
                "validation": result,
                "attempts": attempt + 1,
                "elapsed_sec": round(time.time() - started, 3),
                "usage": usage,
            }
        except Exception as exc:
            last_error = str(exc)
            if attempt < retries:
                prompt += (
                    "\n\nThe previous response failed local validation: "
                    + last_error
                    + "\nReturn one complete corrected JSON object only."
                )
    return {
        "subgraph_id": subgraph_id,
        "status": "error",
        "error": last_error,
        "attempts": retries + 1,
        "elapsed_sec": round(time.time() - started, 3),
        "usage": usage,
    }


def run_source_validation(args: argparse.Namespace) -> dict[str, Any]:
    if not args.base_url:
        raise ValueError("--base-url or OPENAI_BASE_URL is required")
    synthesis = _load_by_subgraph(args.synthesis, allow_retries=True)
    subgraphs = _load_by_subgraph(args.subgraphs)
    completed = _completed_ids(args.output) if args.resume else set()
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    skipped = 0
    for subgraph_id in sorted(synthesis):
        row = synthesis[subgraph_id]
        qa = row.get("qa") or {}
        if row.get("status") != "success" or qa.get("keep") is not True:
            skipped += 1
            continue
        if subgraph_id in completed:
            continue
        subgraph = subgraphs.get(subgraph_id)
        if subgraph is None:
            raise ValueError(f"{subgraph_id}: synthesis has no matching subgraph")
        rows.append((row, subgraph))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        args.output.unlink()
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    client = OpenAIChatClient(
        args.base_url,
        args.api_key,
        args.model,
        timeout=args.timeout,
        retries=args.request_retries,
        extra_headers=parse_headers(args.header),
    )
    counts: Counter[str] = Counter()
    lock = threading.Lock()
    mode = "a" if args.output.exists() and args.resume else "w"
    with args.output.open(mode, encoding="utf-8", newline="\n") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [
                executor.submit(
                    validate_one,
                    row,
                    subgraph,
                    client,
                    prompt,
                    args.max_tokens,
                    args.validation_retries,
                )
                for row, subgraph in rows
            ]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                row = future.result()
                counts[str(row["status"])] += 1
                if row["status"] == "success":
                    decision = row["validation"]
                    counts["kept" if decision["keep"] else "rejected"] += 1
                with lock:
                    serialized = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                    handle.write(serialized + "\n")
                    handle.flush()
                if args.progress_every > 0 and index % args.progress_every == 0:
                    print(f"completed={index}/{len(rows)} counts={dict(counts)}", flush=True)
    report = {
        "stage": "full_document_source_validation",
        "submitted": len(rows),
        "synthesis_rows_skipped": skipped,
        "resumed": len(completed),
        **dict(counts),
        "output": str(args.output.resolve()),
        "request_includes_model_selection": bool(args.model),
    }
    args.output.with_suffix(args.output.suffix + ".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def add_source_validation_arguments(parser: argparse.ArgumentParser) -> None:
    default_prompt = Path(__file__).resolve().parents[1] / "prompts" / "source_validation.txt"
    parser.add_argument("--synthesis", type=Path)
    parser.add_argument("--subgraphs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompt", type=Path, default=default_prompt)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_SOURCE_VALIDATION_MODEL")
        or os.environ.get("OPENAI_MODEL", ""),
    )
    parser.add_argument("--header", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--validation-retries", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
