"""Externally verify synthesized answer sets with bounded web tools."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from elephantbench.client import parse_headers

from .synthesis import parse_json_object, validate_qa
from .web_tools import DEFAULT_SEARCH_ENDPOINT, WebTools

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "WebSearch",
        "description": "Search the public web for pages relevant to a factual claim.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

WEB_FETCH_TOOL = {
    "type": "function",
    "function": {
        "name": "WebFetch",
        "description": "Fetch and read a public HTTP(S) page returned by search.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
}


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().casefold()


def _canonical_url(value: str) -> str:
    """Normalize a public URL for exact seed-source exclusion."""
    value = value.strip()
    if not value:
        return ""
    parsed = urllib.parse.urlsplit(value)
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return value.casefold()
    port = parsed.port
    if port is not None and not (
        (parsed.scheme.lower() == "http" and port == 80)
        or (parsed.scheme.lower() == "https" and port == 443)
    ):
        hostname = f"{hostname}:{port}"
    path = parsed.path.rstrip("/") or "/"
    return urllib.parse.urlunsplit(("", hostname, path, parsed.query, ""))


def _seed_urls(subgraph: dict[str, Any]) -> set[str]:
    return {
        canonical
        for document in subgraph.get("documents") or []
        if isinstance(document, dict)
        and (canonical := _canonical_url(str(document.get("url") or "")))
    }


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


class ToolEndpointClient:
    """Small OpenAI-compatible client with optional endpoint-side model selection."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float,
        retries: int,
        max_tokens: int,
        extra_headers: dict[str, str] | None = None,
        disable_thinking: bool = False,
    ) -> None:
        base = base_url.rstrip("/")
        self.endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.max_tokens = max_tokens
        self.extra_headers = extra_headers or {}
        self.disable_thinking = disable_thinking

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools_enabled: bool,
        json_response: bool = False,
    ) -> tuple[dict[str, Any], list[str]]:
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.model:
            payload["model"] = self.model
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if tools_enabled:
            payload["tools"] = [WEB_SEARCH_TOOL, WEB_FETCH_TOOL]
            payload["tool_choice"] = "auto"
        elif json_response:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        headers.update(self.extra_headers)
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
                    value = json.loads(response.read().decode("utf-8"))
                choices = value.get("choices") or []
                if not choices or not isinstance(choices[0].get("message"), dict):
                    raise ValueError("response has no assistant message")
                return choices[0]["message"], sorted(payload)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(8.0, 2**attempt + random.random()))
        raise RuntimeError(f"request failed after {self.retries + 1} attempts: {last_error}")


def validate_verification(
    result: dict[str, Any],
    qa: dict[str, Any],
    fetched_sources: dict[str, dict[str, Any]],
    seed_urls: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(result.get("keep"), bool):
        errors.append("keep must be boolean")
    if result.get("verdict") not in {"verified", "rejected", "insufficient_evidence"}:
        errors.append("invalid verdict")
    for key in ("same_subject", "same_attribute", "same_fact_context", "qa_valid"):
        if not isinstance(result.get(key), bool):
            errors.append(f"{key} must be boolean")
    if not isinstance(result.get("reason"), str) or not result["reason"].strip():
        errors.append("reason must be a non-empty string")

    expected = {
        _normalize(str(item.get("value") or "")): str(item.get("value") or "")
        for item in qa["gold_answers"]
    }
    checks = result.get("answer_verifications")
    if not isinstance(checks, list):
        errors.append("answer_verifications must be a list")
        checks = []
    seen: set[str] = set()
    all_verified = True
    seed_urls = seed_urls or set()
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            errors.append(f"answer verification {index} is not an object")
            all_verified = False
            continue
        normalized = _normalize(str(check.get("value") or ""))
        if normalized not in expected:
            errors.append(f"answer verification {index} has an unknown value")
        if normalized in seen:
            errors.append(f"answer verification {index} duplicates a value")
        seen.add(normalized)
        verified = check.get("verified") is True
        all_verified = all_verified and verified
        sources = check.get("sources")
        if not isinstance(sources, list):
            errors.append(f"answer verification {index} sources must be a list")
            sources = []
        if verified and not sources:
            errors.append(f"answer verification {index} claims verification without a source")
        for source in sources:
            if not isinstance(source, dict):
                errors.append(f"answer verification {index} has an invalid source")
                continue
            source_id = str(source.get("source_id") or "")
            quote = _normalize(str(source.get("quote") or ""))
            fetched = fetched_sources.get(source_id)
            if fetched is None:
                errors.append(f"answer verification {index} cites an unfetched source")
            else:
                fetched_urls = {
                    _canonical_url(str(fetched.get("url") or "")),
                    _canonical_url(str(fetched.get("requested_url") or "")),
                }
                fetched_urls.discard("")
                if fetched_urls & seed_urls:
                    errors.append(
                        f"answer verification {index} reuses a seed document "
                        "as independent evidence"
                    )
                if not quote or quote not in _normalize(str(fetched.get("text") or "")):
                    errors.append(f"answer verification {index} quote is not verbatim fetched text")
    if seen != set(expected):
        errors.append("answer_verifications must cover every gold answer exactly once")

    if result.get("keep") is True:
        if result.get("verdict") != "verified":
            errors.append("keep=true requires verdict=verified")
        if not all(
            result.get(key) is True
            for key in ("same_subject", "same_attribute", "same_fact_context", "qa_valid")
        ):
            errors.append("keep=true requires all comparability and QA checks to pass")
        if not all_verified or len(seen) != len(expected):
            errors.append("keep=true requires independent web evidence for every answer")
    return errors


def _verification_input(
    qa: dict[str, Any], subgraph: dict[str, Any], source_validation: dict[str, Any]
) -> dict[str, Any]:
    return {
        "qa": qa,
        "source_validation": source_validation,
        "seed_conflict": subgraph.get("seed_conflict"),
        "support_edges": subgraph.get("support_edges") or [],
        "seed_documents": subgraph.get("documents") or [],
    }


def _quotation_failure_result(
    qa: dict[str, Any], attempted: dict[str, Any] | None = None
) -> dict[str, Any]:
    attempted = attempted or {}
    reason = "Automatic verification could not provide an exact quotation from the fetched pages."
    return {
        "keep": False,
        "verdict": "insufficient_evidence",
        "same_subject": attempted.get("same_subject") is not False,
        "same_attribute": attempted.get("same_attribute") is not False,
        "same_fact_context": attempted.get("same_fact_context") is not False,
        "qa_valid": attempted.get("qa_valid") is not False,
        "answer_verifications": [
            {
                "value": str(answer.get("value") or ""),
                "verified": False,
                "sources": [],
                "reason": reason,
            }
            for answer in qa["gold_answers"]
        ],
        "reason": reason,
    }


def verify_one(
    row: dict[str, Any],
    subgraph: dict[str, Any],
    source_validation: dict[str, Any],
    client: ToolEndpointClient,
    web: WebTools,
    system_prompt: str,
    *,
    search_budget: int,
    fetch_budget: int,
    max_steps: int,
    final_retries: int,
) -> dict[str, Any]:
    started = time.time()
    subgraph_id = str(row["subgraph_id"])
    qa = row["qa"]
    decision = source_validation.get("validation") or {}
    if (
        source_validation.get("status") != "success"
        or decision.get("keep") is not True
        or decision.get("verdict") != "verified"
    ):
        return {
            "subgraph_id": subgraph_id,
            "status": "error",
            "error": "record did not pass full-document source validation",
            "web_trace": {"searches": [], "fetches": []},
            "request_fields": [],
            "attempts": 0,
            "elapsed_sec": round(time.time() - started, 3),
        }
    qa_errors = validate_qa(qa, subgraph)
    if qa_errors:
        return {
            "subgraph_id": subgraph_id,
            "status": "error",
            "error": "synthesis/subgraph mismatch: " + "; ".join(qa_errors),
            "web_trace": {"searches": [], "fetches": []},
            "request_fields": [],
            "attempts": 0,
            "elapsed_sec": round(time.time() - started, 3),
        }
    required_searches = len(qa["gold_answers"])
    if search_budget < required_searches:
        return {
            "subgraph_id": subgraph_id,
            "status": "error",
            "error": (
                f"search budget {search_budget} is smaller than the "
                f"{required_searches} proposed answers"
            ),
            "web_trace": {"searches": [], "fetches": []},
            "request_fields": [],
            "attempts": 0,
            "elapsed_sec": round(time.time() - started, 3),
        }
    seed_urls = _seed_urls(subgraph)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Find independent public-web evidence for this source-validated QA record:\n"
            + json.dumps(
                _verification_input(qa, subgraph, decision),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]
    searches: list[dict[str, Any]] = []
    fetch_log: list[dict[str, Any]] = []
    fetched_sources: dict[str, dict[str, Any]] = {}
    fetched_url_index: dict[str, str] = {}
    request_fields: set[str] = set()
    correction = 0
    tools_enabled = True
    last_error = "agent did not produce a final result"

    for _ in range(max_steps):
        try:
            message, fields = client.complete(
                messages,
                tools_enabled=tools_enabled,
                json_response=not tools_enabled,
            )
            request_fields.update(fields)
        except Exception as exc:
            last_error = str(exc)
            break
        tool_calls = message.get("tool_calls") or []
        if tool_calls and tools_enabled:
            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": tool_calls,
                }
            )
            budget_exhausted = False
            for call in tool_calls:
                function = call.get("function") or {}
                name = str(function.get("name") or "")
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                try:
                    if name == "WebSearch":
                        if len(searches) >= search_budget:
                            tool_result = {"status": "error", "error": "WebSearch budget exhausted"}
                            budget_exhausted = True
                        else:
                            tool_result = web.search(str(arguments.get("query") or ""))
                            searches.append(
                                {
                                    "query": tool_result["query"],
                                    "result_count": len(tool_result.get("results") or []),
                                    "cache_hit": tool_result.get("cache_hit"),
                                }
                            )
                    elif name == "WebFetch":
                        requested_url = str(arguments.get("url") or "").strip()
                        existing_source = fetched_url_index.get(requested_url)
                        if existing_source:
                            fetched = fetched_sources[existing_source]
                            tool_result = {
                                "source_id": existing_source,
                                "url": fetched.get("url"),
                                "title": fetched.get("title"),
                                "text": fetched.get("text"),
                                "cache_hit": True,
                            }
                        elif len(fetch_log) >= fetch_budget:
                            tool_result = {"status": "error", "error": "WebFetch budget exhausted"}
                            budget_exhausted = True
                        else:
                            fetched = web.fetch(requested_url)
                            source_id = f"source-{len(fetch_log) + 1}"
                            fetched_sources[source_id] = fetched
                            fetched_url_index[requested_url] = source_id
                            fetched_url_index[str(fetched.get("url") or "")] = source_id
                            fetch_log.append(
                                {
                                    "source_id": source_id,
                                    "url": fetched.get("url"),
                                    "title": fetched.get("title"),
                                    "content_sha256": fetched.get("content_sha256"),
                                    "cache_hit": fetched.get("cache_hit"),
                                }
                            )
                            tool_result = {
                                "source_id": source_id,
                                "url": fetched.get("url"),
                                "title": fetched.get("title"),
                                "text": fetched.get("text"),
                            }
                    else:
                        tool_result = {"status": "error", "error": f"unknown tool: {name}"}
                except Exception as exc:
                    tool_result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "name": name or "unknown",
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )
            if budget_exhausted or (
                len(searches) >= search_budget and len(fetch_log) >= fetch_budget
            ):
                tools_enabled = False
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The tool budget is exhausted. Return the final JSON object now."
                        ),
                    }
                )
            continue

        content = _message_text(message.get("content"))
        validation_errors: list[str] | None = None
        try:
            result = parse_json_object(content)
            validation_errors = validate_verification(
                result, qa, fetched_sources, seed_urls=seed_urls
            )
            if len(searches) < required_searches:
                validation_errors.append(
                    "verification must use WebSearch separately for every proposed answer"
                )
            if validation_errors:
                raise ValueError("; ".join(validation_errors))
            return {
                "subgraph_id": subgraph_id,
                "status": "success",
                "verification": result,
                "web_trace": {"searches": searches, "fetches": fetch_log},
                "request_fields": sorted(request_fields),
                "attempts": correction + 1,
                "elapsed_sec": round(time.time() - started, 3),
            }
        except Exception as exc:
            last_error = str(exc)
            if correction >= final_retries:
                if validation_errors and all(
                    "quote is not verbatim fetched text" in error for error in validation_errors
                ):
                    return {
                        "subgraph_id": subgraph_id,
                        "status": "success",
                        "verification": _quotation_failure_result(qa, result),
                        "web_trace": {"searches": searches, "fetches": fetch_log},
                        "request_fields": sorted(request_fields),
                        "attempts": correction + 1,
                        "elapsed_sec": round(time.time() - started, 3),
                    }
                break
            correction += 1
            requires_more_tools = any(
                marker in error
                for error in (validation_errors or [])
                for marker in (
                    "must use WebSearch",
                    "without a source",
                    "unfetched source",
                    "independent web evidence",
                )
            )
            tools_enabled = requires_more_tools and (
                len(searches) < search_budget or len(fetch_log) < fetch_budget
            )
            messages.append({"role": "assistant", "content": content})
            if tools_enabled:
                instruction = (
                    "The result failed validation because the required web research is incomplete. "
                    "Call WebSearch separately for every proposed answer and WebFetch relevant "
                    "pages before returning a corrected final JSON object. Validation errors: "
                    + last_error
                )
            else:
                instruction = (
                    "The final result failed validation: "
                    + last_error
                    + "\nReturn one corrected JSON object using only fetched sources. "
                    "Copy each quote exactly from a WebFetch text field: do not paraphrase, "
                    "splice passages, add ellipses, or change punctuation. If no exact "
                    "supporting span exists, mark that value unverified and return "
                    "keep=false with verdict=insufficient_evidence."
                )
            messages.append({"role": "user", "content": instruction})

    return {
        "subgraph_id": subgraph_id,
        "status": "error",
        "error": last_error,
        "web_trace": {"searches": searches, "fetches": fetch_log},
        "request_fields": sorted(request_fields),
        "attempts": correction + 1,
        "elapsed_sec": round(time.time() - started, 3),
    }


def _load_synthesis(path: Path) -> tuple[list[dict[str, Any]], int]:
    latest: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            subgraph_id = str(row.get("subgraph_id") or "")
            if not subgraph_id:
                raise ValueError(f"{path}:{line_number}: subgraph_id is missing")
            latest[subgraph_id] = row
    rows = [
        latest[subgraph_id]
        for subgraph_id in sorted(latest)
        if latest[subgraph_id].get("status") == "success"
        and (latest[subgraph_id].get("qa") or {}).get("keep") is True
    ]
    skipped = len(latest) - len(rows)
    return rows, skipped


def _load_subgraphs(path: Path) -> dict[str, dict[str, Any]]:
    subgraphs: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            subgraph_id = str(row.get("subgraph_id") or "")
            if not subgraph_id:
                raise ValueError(f"{path}:{line_number}: subgraph_id is missing")
            if subgraph_id in subgraphs:
                raise ValueError(f"{path}:{line_number}: duplicate subgraph_id {subgraph_id}")
            subgraphs[subgraph_id] = row
    return subgraphs


def _load_source_validations(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    latest: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            subgraph_id = str(row.get("subgraph_id") or "")
            if not subgraph_id:
                raise ValueError(f"{path}:{line_number}: subgraph_id is missing")
            latest[subgraph_id] = row
    selected = {
        subgraph_id: row
        for subgraph_id, row in latest.items()
        if row.get("status") == "success"
        and (row.get("validation") or {}).get("keep") is True
        and (row.get("validation") or {}).get("verdict") == "verified"
    }
    skipped = len(latest) - len(selected)
    return selected, skipped


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


def run_verification(args: argparse.Namespace) -> dict[str, Any]:
    if not args.base_url:
        raise ValueError("--base-url or OPENAI_BASE_URL is required")
    if not args.search_api_key:
        raise ValueError("--search-api-key or BRAVE_SEARCH_API_KEY is required")
    rows, synthesis_skipped = _load_synthesis(args.synthesis)
    source_validations, source_validation_skipped = _load_source_validations(args.source_validation)
    subgraphs = _load_subgraphs(args.subgraphs)
    missing_subgraphs = sorted(
        str(row["subgraph_id"]) for row in rows if str(row["subgraph_id"]) not in subgraphs
    )
    if missing_subgraphs:
        preview = ", ".join(missing_subgraphs[:5])
        raise ValueError(f"synthesis rows have no matching subgraph: {preview}")
    missing_validations = sorted(
        str(row["subgraph_id"]) for row in rows if str(row["subgraph_id"]) not in source_validations
    )
    rows = [row for row in rows if str(row["subgraph_id"]) in source_validations]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_ids(args.output) if args.resume else set()
    if args.output.exists() and not args.resume:
        args.output.unlink()
    pending = [row for row in rows if str(row["subgraph_id"]) not in completed]
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    client = ToolEndpointClient(
        args.base_url,
        args.api_key,
        args.model,
        timeout=args.timeout,
        retries=args.request_retries,
        max_tokens=args.max_tokens,
        extra_headers=parse_headers(args.header),
        disable_thinking=args.disable_thinking,
    )
    web = WebTools(
        args.cache_dir,
        search_endpoint=args.search_endpoint,
        search_api_key=args.search_api_key,
        timeout=args.web_timeout,
        search_results=args.search_results,
        search_concurrency=args.search_concurrency,
        fetch_concurrency=args.fetch_concurrency,
        allow_private_network=args.allow_private_network,
    )
    counts: Counter[str] = Counter()
    lock = threading.Lock()
    mode = "a" if args.output.exists() and args.resume else "w"
    with args.output.open(mode, encoding="utf-8", newline="\n") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [
                executor.submit(
                    verify_one,
                    row,
                    subgraphs[str(row["subgraph_id"])],
                    source_validations[str(row["subgraph_id"])],
                    client,
                    web,
                    prompt,
                    search_budget=args.search_budget,
                    fetch_budget=args.fetch_budget,
                    max_steps=args.max_steps,
                    final_retries=args.final_retries,
                )
                for row in pending
            ]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                row = future.result()
                counts[str(row["status"])] += 1
                if row["status"] == "success":
                    verification = row["verification"]
                    counts["decision_kept" if verification["keep"] else "decision_rejected"] += 1
                    counts[f"verdict_{verification['verdict']}"] += 1
                with lock:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    handle.flush()
                if args.progress_every > 0 and index % args.progress_every == 0:
                    print(f"completed={index}/{len(pending)} counts={dict(counts)}", flush=True)
    report = {
        "stage": "external_web_verification",
        "synthesized_kept": len(rows),
        "synthesis_rows_skipped": synthesis_skipped,
        "source_validation_rows_skipped": source_validation_skipped,
        "synthesis_rows_without_verified_sources": len(missing_validations),
        "resumed": len(completed),
        "submitted": len(pending),
        **dict(counts),
        "output": str(args.output.resolve()),
        "request_includes_model_selection": bool(args.model),
    }
    args.output.with_suffix(args.output.suffix + ".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def add_verification_arguments(parser: argparse.ArgumentParser) -> None:
    default_prompt = Path(__file__).resolve().parents[1] / "prompts" / "web_verification.txt"
    parser.add_argument("--synthesis", type=Path)
    parser.add_argument("--source-validation", type=Path)
    parser.add_argument("--subgraphs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--prompt", type=Path, default=default_prompt)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("OPENAI_VERIFY_MODEL", ""))
    parser.add_argument("--header", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--search-budget", type=int, default=3)
    parser.add_argument("--fetch-budget", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--final-retries", type=int, default=1)
    parser.add_argument("--search-endpoint", default=DEFAULT_SEARCH_ENDPOINT)
    parser.add_argument(
        "--search-api-key",
        default=os.environ.get("BRAVE_SEARCH_API_KEY", ""),
        help="Brave Search subscription token (defaults to BRAVE_SEARCH_API_KEY)",
    )
    parser.add_argument("--search-results", type=int, default=5)
    parser.add_argument("--search-concurrency", type=int, default=2)
    parser.add_argument("--fetch-concurrency", type=int, default=8)
    parser.add_argument("--web-timeout", type=float, default=30.0)
    parser.add_argument("--allow-private-network", action="store_true")
    parser.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="send a gateway-specific request to disable reasoning",
    )
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
