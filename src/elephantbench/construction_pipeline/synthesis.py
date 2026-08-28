"""Synthesize paired closed-book questions from conflict-centered subgraphs."""

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


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.I)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("response contains no JSON object")
        text = text[start : end + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("response JSON is not an object")
    return value


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def validate_qa(qa: dict[str, Any], subgraph: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(qa.get("keep"), bool):
        return ["keep must be boolean"]
    if qa["keep"] is False:
        if not str(qa.get("rejection_reason") or "").strip():
            errors.append("keep=false requires rejection_reason")
        if qa.get("questions") not in (None, []):
            errors.append("keep=false requires empty questions")
        if qa.get("gold_answers") not in (None, []):
            errors.append("keep=false requires empty gold_answers")
        return errors

    for field in ("subject", "attribute", "preferred_answer"):
        if not isinstance(qa.get(field), str) or not qa[field].strip():
            errors.append(f"{field} must be a non-empty string")
    questions = qa.get("questions")
    answers = qa.get("gold_answers")
    if not isinstance(questions, list) or len(questions) != 2:
        errors.append("exactly two questions are required")
        questions = []
    formulations = {
        str(item.get("formulation") or "") for item in questions if isinstance(item, dict)
    }
    if formulations != {"named_entity", "clue_based"}:
        errors.append("questions must contain named_entity and clue_based")
    if not isinstance(answers, list) or len(answers) < 2:
        errors.append("at least two gold answers are required")
        answers = []

    document_text = {
        str(doc["doc_id"]): _normalize(str(doc.get("text") or ""))
        for doc in subgraph.get("documents") or []
        if isinstance(doc, dict) and doc.get("doc_id")
    }
    known_ids = set(document_text)
    values: list[str] = []
    for index, answer in enumerate(answers):
        if not isinstance(answer, dict):
            errors.append(f"gold answer {index} is not an object")
            continue
        value = str(answer.get("value") or "").strip()
        if not value:
            errors.append(f"gold answer {index} has empty value")
        values.append(value)
        supporting = answer.get("supporting_doc_ids")
        if not isinstance(supporting, list) or not supporting:
            errors.append(f"gold answer {index} has no supporting_doc_ids")
        elif not set(map(str, supporting)) <= known_ids:
            errors.append(f"gold answer {index} references an unknown document")
        spans = answer.get("evidence_spans")
        if not isinstance(spans, list) or not spans:
            errors.append(f"gold answer {index} has no evidence spans")
            continue
        for span in spans:
            if not isinstance(span, dict):
                errors.append(f"gold answer {index} has invalid evidence span")
                continue
            doc_id = str(span.get("doc_id") or "")
            quote = _normalize(str(span.get("quote") or ""))
            if doc_id not in known_ids or not quote:
                errors.append(f"gold answer {index} has invalid evidence reference")
            elif quote not in document_text[doc_id]:
                errors.append(f"gold answer {index} evidence is not verbatim")

    normalized_values = [_normalize(value) for value in values if value]
    if len(normalized_values) != len(set(normalized_values)):
        errors.append("gold answer values must be distinct")
    for item in questions:
        if not isinstance(item, dict) or not str(item.get("question") or "").strip():
            errors.append("question text must be non-empty")
            continue
        question = _normalize(str(item["question"]))
        for value in normalized_values:
            if len(value) >= 3 and value in question:
                errors.append("question leaks a gold answer")
                break
    return errors


def _load_subgraphs(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def synthesize_one(
    subgraph: dict[str, Any],
    client: OpenAIChatClient,
    system_prompt: str,
    max_tokens: int,
    retries: int,
) -> dict[str, Any]:
    started = time.time()
    subgraph_id = str(subgraph["subgraph_id"])
    prompt = "Synthesize a paired QA record from this subgraph:\n" + json.dumps(
        subgraph, ensure_ascii=False, separators=(",", ":")
    )
    last_error = ""
    usage: dict[str, Any] = {}
    for attempt in range(retries + 1):
        try:
            response = client.chat(system_prompt, prompt, temperature=0.0, max_tokens=max_tokens)
            if response.get("finish_reason") not in (None, "stop"):
                raise ValueError(f"unexpected finish_reason={response.get('finish_reason')!r}")
            usage = response.get("usage") or {}
            qa = parse_json_object(str(response.get("content") or ""))
            errors = validate_qa(qa, subgraph)
            if errors:
                raise ValueError("; ".join(errors))
            return {
                "subgraph_id": subgraph_id,
                "status": "success",
                "qa": qa,
                "attempts": attempt + 1,
                "elapsed_sec": round(time.time() - started, 3),
                "usage": usage,
            }
        except Exception as exc:
            last_error = str(exc)
            if attempt < retries:
                prompt += (
                    "\n\nThe previous response failed validation: "
                    + last_error
                    + "\nReturn one corrected JSON object only."
                )
    return {
        "subgraph_id": subgraph_id,
        "status": "error",
        "error": last_error,
        "attempts": retries + 1,
        "elapsed_sec": round(time.time() - started, 3),
        "usage": usage,
    }


def run_synthesis(args: argparse.Namespace) -> dict[str, Any]:
    if not args.base_url:
        raise ValueError("--base-url or OPENAI_BASE_URL is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_ids(args.output) if args.resume else set()
    if args.output.exists() and not args.resume:
        args.output.unlink()
    subgraphs = [
        row
        for row in _load_subgraphs(args.subgraphs)
        if str(row.get("subgraph_id") or "") not in completed
    ]
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
                    synthesize_one, row, client, prompt, args.max_tokens, args.validation_retries
                )
                for row in subgraphs
            ]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                row = future.result()
                counts[str(row["status"])] += 1
                if row["status"] == "success":
                    counts["kept" if row["qa"]["keep"] else "rejected"] += 1
                with lock:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    handle.flush()
                if args.progress_every > 0 and index % args.progress_every == 0:
                    print(f"completed={index}/{len(subgraphs)} counts={dict(counts)}", flush=True)
    report = {
        "stage": "synthesize_paired_questions",
        "input_subgraphs": len(subgraphs),
        "resumed": len(completed),
        **dict(counts),
        "output": str(args.output.resolve()),
        "request_includes_model_selection": bool(args.model),
    }
    args.output.with_suffix(args.output.suffix + ".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def add_synthesis_arguments(parser: argparse.ArgumentParser) -> None:
    default_prompt = Path(__file__).resolve().parents[1] / "prompts" / "qa_synthesis.txt"
    parser.add_argument("--subgraphs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompt", type=Path, default=default_prompt)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", ""))
    parser.add_argument("--header", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--validation-retries", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
