"""Full-document relation classification for pre-LLM candidate pairs."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import multiprocessing as mp
import os
import queue
import random
import re
import sqlite3
import ssl
import threading
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 501, 502, 503, 504}
RELATIONS = {"support", "conflict", "none"}


class NonRetryableRequestError(RuntimeError):
    """An HTTP failure that should not consume the retry budget."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def response_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "elephantbench_document_relation",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "relation": {
                        "type": "string",
                        "enum": ["support", "conflict", "none"],
                    },
                    "same_subject": {"type": "boolean"},
                    "same_attribute": {"type": "boolean"},
                    "same_fact_context": {"type": "boolean"},
                    "values_compatible": {"type": "boolean"},
                    "subject": {"type": "string"},
                    "attribute": {"type": "string"},
                    "value_a": {"type": "string"},
                    "value_b": {"type": "string"},
                    "evidence_a": {"type": "string"},
                    "evidence_b": {"type": "string"},
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "reason": {"type": "string"},
                },
                "required": [
                    "relation",
                    "same_subject",
                    "same_attribute",
                    "same_fact_context",
                    "values_compatible",
                    "subject",
                    "attribute",
                    "value_a",
                    "value_b",
                    "evidence_a",
                    "evidence_b",
                    "confidence",
                    "reason",
                ],
            },
        },
    }


def _json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"response contains no JSON object: {text[:300]!r}")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response is not a JSON object")
    return value


def derived_relation(value: dict[str, Any]) -> str:
    comparable = all(
        value.get(key) is True for key in ("same_subject", "same_attribute", "same_fact_context")
    )
    if not comparable:
        return "none"
    return "support" if value.get("values_compatible") is True else "conflict"


def validate_relation(value: dict[str, Any]) -> str:
    relation = value.get("relation")
    if relation not in RELATIONS:
        raise ValueError(f"invalid relation: {relation!r}")
    for key in ("same_subject", "same_attribute", "same_fact_context", "values_compatible"):
        if not isinstance(value.get(key), bool):
            raise ValueError(f"{key} must be a boolean")
    for key in (
        "subject",
        "attribute",
        "value_a",
        "value_b",
        "evidence_a",
        "evidence_b",
        "reason",
    ):
        if not isinstance(value.get(key), str):
            raise ValueError(f"{key} must be a string")
    confidence = value.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ValueError("confidence must be a number")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    return derived_relation(value)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _knowledge_point(document: dict[str, Any]) -> dict[str, str]:
    value = document.get("knowledge_point")
    if not isinstance(value, dict):
        return {}
    return {
        key: str(value.get(key) or "")
        for key in ("discipline", "field", "subfield")
        if value.get(key)
    }


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    doc_a = candidate.get("doc_a") or {}
    doc_b = candidate.get("doc_b") or {}
    return {
        "pair_id": str(candidate.get("pair_id") or ""),
        "doc_a_id": str(doc_a.get("doc_id") or ""),
        "doc_b_id": str(doc_b.get("doc_id") or ""),
        "metadata": {
            "retrieval_signals": candidate.get("retrieval_signals") or [],
            "subjects": candidate.get("subjects") or [],
            "slots": candidate.get("slots") or [],
            "knowledge_point_a": _knowledge_point(doc_a),
            "knowledge_point_b": _knowledge_point(doc_b),
            "dates_a": doc_a.get("dates") or [],
            "dates_b": doc_b.get("dates") or [],
            "numbers_a": doc_a.get("numbers") or [],
            "numbers_b": doc_b.get("numbers") or [],
            "retrieval_evidence_a": str(doc_a.get("evidence") or ""),
            "retrieval_evidence_b": str(doc_b.get("evidence") or ""),
        },
    }


def render_user_prompt(
    task: dict[str, Any],
    url_a: str,
    text_a: str,
    url_b: str,
    text_b: str,
) -> str:
    metadata = json.dumps(
        task["metadata"], ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return (
        "CANDIDATE METADATA (retrieval hints only; not proof):\n"
        f"{metadata}\n\n"
        f"DOCUMENT A\nURL: {url_a}\nFULL TEXT:\n{text_a}\n\n"
        f"DOCUMENT B\nURL: {url_b}\nFULL TEXT:\n{text_b}\n"
    )


def _endpoint_parts(base_url: str) -> tuple[str, str, int | None, str]:
    parsed = urlsplit(base_url.rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"invalid base URL: {base_url!r}")
    path = parsed.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        path += "/chat/completions"
    if parsed.query:
        path += "?" + parsed.query
    return parsed.scheme, parsed.hostname, parsed.port, path


class ThreadLocalHTTPClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.scheme, self.host, self.port, self.path = _endpoint_parts(base_url)
        self.timeout = timeout
        self.local = threading.local()

    def _new_connection(self) -> http.client.HTTPConnection:
        if self.scheme == "https":
            return http.client.HTTPSConnection(
                self.host,
                self.port,
                timeout=self.timeout,
                context=ssl.create_default_context(),
            )
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def _connection(self) -> http.client.HTTPConnection:
        connection = getattr(self.local, "connection", None)
        if connection is None:
            connection = self._new_connection()
            self.local.connection = connection
        return connection

    def close_current(self) -> None:
        connection = getattr(self.local, "connection", None)
        if connection is not None:
            try:
                connection.close()
            finally:
                self.local.connection = None

    def post(self, payload: dict[str, Any], headers: dict[str, str]) -> tuple[int, str]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        connection = self._connection()
        try:
            connection.request("POST", self.path, body=body, headers=headers)
            response = connection.getresponse()
            text = response.read().decode("utf-8", errors="replace")
            self.close_current()
            return response.status, text
        except Exception:
            self.close_current()
            raise


class DocumentStorePool:
    def __init__(self, path: str, size: int) -> None:
        self.connections: queue.Queue[sqlite3.Connection] = queue.Queue(maxsize=size)
        uri = f"file:{Path(path).resolve()}?mode=ro&immutable=1"
        for _ in range(size):
            connection = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=120.0)
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA temp_store=MEMORY")
            self.connections.put(connection)

    def get_pair(self, doc_a_id: str, doc_b_id: str) -> dict[str, tuple[str, str]]:
        connection = self.connections.get()
        try:
            rows = connection.execute(
                "SELECT canonical_doc_id, url, text FROM docs WHERE canonical_doc_id IN (?, ?)",
                (doc_a_id, doc_b_id),
            ).fetchall()
        finally:
            self.connections.put(connection)
        return {str(doc_id): (str(url or ""), str(text or "")) for doc_id, url, text in rows}


@dataclass
class ClassifierSettings:
    base_url: str
    model: str
    api_key: str
    timeout: float
    retries: int
    retry_sleep: float
    max_retry_sleep: float
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int | None
    min_p: float | None
    presence_penalty: float
    repetition_penalty: float | None
    disable_thinking: bool


def _response_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("response has no choices")
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    raise ValueError("response has empty message content")


def classify_task(
    task: dict[str, Any],
    store: DocumentStorePool,
    client: ThreadLocalHTTPClient,
    system_prompt: str,
    settings: ClassifierSettings,
) -> dict[str, Any]:
    started = time.time()
    pair_id = task["pair_id"]
    try:
        documents = store.get_pair(task["doc_a_id"], task["doc_b_id"])
        if task["doc_a_id"] not in documents or task["doc_b_id"] not in documents:
            missing = [
                doc_id for doc_id in (task["doc_a_id"], task["doc_b_id"]) if doc_id not in documents
            ]
            raise ValueError(f"missing canonical documents: {missing}")
        url_a, text_a = documents[task["doc_a_id"]]
        url_b, text_b = documents[task["doc_b_id"]]
        text_a, text_b = _normalize_text(text_a), _normalize_text(text_b)
        if not text_a or not text_b:
            raise ValueError("canonical document text is empty")
        user_prompt = render_user_prompt(task, url_a, text_a, url_b, text_b)
        payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "presence_penalty": settings.presence_penalty,
            "max_tokens": settings.max_tokens,
            "stream": False,
            "response_format": response_schema(),
        }
        if settings.model:
            payload["model"] = settings.model
        if settings.top_k is not None:
            payload["top_k"] = settings.top_k
        if settings.min_p is not None:
            payload["min_p"] = settings.min_p
        if settings.repetition_penalty is not None:
            payload["repetition_penalty"] = settings.repetition_penalty
        if settings.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        delay = settings.retry_sleep
        last_error: Exception | None = None
        for attempt in range(settings.retries + 1):
            try:
                status, response_text = client.post(payload, headers)
                if status >= 400:
                    error = RuntimeError(f"HTTP {status}: {response_text[:500]}")
                    if status not in RETRYABLE_STATUS:
                        raise NonRetryableRequestError(str(error))
                    raise error
                response = json.loads(response_text)
                judgment = _json_object(_response_content(response))
                normalized_relation = validate_relation(judgment)
                model_relation = judgment["relation"]
                judgment["relation"] = normalized_relation
                if model_relation != normalized_relation:
                    judgment["model_relation"] = model_relation
                usage = response.get("usage") or {}
                return {
                    "pair_id": pair_id,
                    "doc_a_id": task["doc_a_id"],
                    "doc_b_id": task["doc_b_id"],
                    "status": "success",
                    "relation": normalized_relation,
                    "judgment": judgment,
                    "model": response.get("model") or settings.model,
                    "latency_sec": round(time.time() - started, 3),
                    "usage": {
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                    },
                    "retrieval_signals": task["metadata"].get("retrieval_signals") or [],
                    "created_at": utc_now(),
                }
            except NonRetryableRequestError as exc:
                last_error = exc
                client.close_current()
                break
            except Exception as exc:
                last_error = exc
                client.close_current()
                if attempt >= settings.retries:
                    break
                time.sleep(min(settings.max_retry_sleep, delay + random.random()))
                delay = min(settings.max_retry_sleep, max(delay * 2, delay + 1))
        raise RuntimeError(f"request failed after {settings.retries + 1} attempts: {last_error}")
    except Exception as exc:
        return {
            "pair_id": pair_id,
            "doc_a_id": task.get("doc_a_id"),
            "doc_b_id": task.get("doc_b_id"),
            "status": "error",
            "error": str(exc),
            "latency_sec": round(time.time() - started, 3),
            "created_at": utc_now(),
        }


def _worker_thread(
    tasks: queue.SimpleQueue[Any],
    outputs: mp.Queue,
    store: DocumentStorePool,
    client: ThreadLocalHTTPClient,
    system_prompt: str,
    settings: ClassifierSettings,
) -> None:
    while True:
        task = tasks.get()
        if task is None:
            return
        outputs.put(classify_task(task, store, client, system_prompt, settings))


def _worker_process(
    worker_id: int,
    inputs: mp.Queue,
    outputs: mp.Queue,
    doc_store: str,
    doc_store_connections: int,
    threads: int,
    system_prompt: str,
    settings: ClassifierSettings,
) -> None:
    store = DocumentStorePool(doc_store, doc_store_connections)
    client = ThreadLocalHTTPClient(settings.base_url, settings.timeout)
    local_tasks: queue.SimpleQueue[Any] = queue.SimpleQueue()
    workers = [
        threading.Thread(
            target=_worker_thread,
            args=(local_tasks, outputs, store, client, system_prompt, settings),
            daemon=True,
        )
        for _ in range(threads)
    ]
    for worker in workers:
        worker.start()
    while True:
        task = inputs.get()
        if task is None:
            break
        local_tasks.put(task)
    for _ in workers:
        local_tasks.put(None)
    for worker in workers:
        worker.join()
    outputs.put({"worker_done": worker_id})


def load_completed_pair_ids(path: Path) -> set[str]:
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
            pair_id = row.get("pair_id")
            if row.get("status") == "success" and isinstance(pair_id, str):
                completed.add(pair_id)
    return completed


def iter_candidate_tasks(
    path: Path, completed: set[str], max_pairs: int
) -> Iterable[dict[str, Any]]:
    selected = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            task = compact_candidate(json.loads(line))
            if not task["pair_id"] or not task["doc_a_id"] or not task["doc_b_id"]:
                continue
            selected += 1
            if max_pairs > 0 and selected > max_pairs:
                return
            if task["pair_id"] not in completed:
                yield task


def _task_shard(pair_id: str, processes: int) -> int:
    digest = hashlib.blake2b(pair_id.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % processes


def run_relation_classification(args: argparse.Namespace) -> dict[str, Any]:
    if not args.base_url:
        raise ValueError("--base-url or OPENAI_BASE_URL is required")
    candidates = args.candidates.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = load_completed_pair_ids(output) if args.resume else set()
    if output.exists() and not args.resume:
        output.unlink()
    system_prompt = args.prompt.read_text(encoding="utf-8").strip()
    settings = ClassifierSettings(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        max_retry_sleep=args.max_retry_sleep,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        disable_thinking=args.disable_thinking,
    )
    context = mp.get_context("fork")
    input_queues = [
        context.Queue(maxsize=max(args.threads_per_process * 4, 16)) for _ in range(args.processes)
    ]
    output_queue = context.Queue(maxsize=max(args.processes * args.threads_per_process * 2, 32))
    submitted = 0
    feeder_error: str | None = None

    def feed() -> None:
        nonlocal submitted, feeder_error
        try:
            for task in iter_candidate_tasks(candidates, completed_ids, args.max_pairs):
                input_queues[_task_shard(task["pair_id"], args.processes)].put(task)
                submitted += 1
        except Exception as exc:
            feeder_error = str(exc)
        finally:
            for input_queue in input_queues:
                input_queue.put(None)

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    processes = [
        context.Process(
            target=_worker_process,
            args=(
                worker_id,
                input_queues[worker_id],
                output_queue,
                str(args.doc_store.resolve()),
                args.doc_store_connections,
                args.threads_per_process,
                system_prompt,
                settings,
            ),
        )
        for worker_id in range(args.processes)
    ]
    for process in processes:
        process.start()

    counts: Counter[str] = Counter()
    done_workers = 0
    started = time.time()
    mode = "a" if output.exists() and args.resume else "w"
    with output.open(mode, encoding="utf-8") as handle:
        while done_workers < args.processes:
            record = output_queue.get()
            if "worker_done" in record:
                done_workers += 1
                continue
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[record.get("status") or "unknown"] += 1
            if record.get("status") == "success":
                counts[record.get("relation") or "unknown"] += 1
            completed_now = counts["success"] + counts["error"]
            if args.progress_every > 0 and completed_now % args.progress_every == 0:
                handle.flush()
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"completed={completed_now} submitted={submitted} "
                    f"support={counts['support']} conflict={counts['conflict']} "
                    f"none={counts['none']} errors={counts['error']} "
                    f"rate={completed_now / elapsed:.2f}/s",
                    flush=True,
                )

    feeder.join()
    worker_errors: list[str] = []
    for process in processes:
        process.join()
        if process.exitcode not in (0, None):
            worker_errors.append(f"worker {process.pid} exited with {process.exitcode}")
    elapsed = max(time.time() - started, 1e-6)
    report = {
        "stage": "relation_classification",
        "candidates": str(candidates),
        "output": str(output),
        "resumed_successes": len(completed_ids),
        "submitted": submitted,
        "completed": counts["success"] + counts["error"],
        "success": counts["success"],
        "errors": counts["error"],
        "relations": {
            "support": counts["support"],
            "conflict": counts["conflict"],
            "none": counts["none"],
        },
        "elapsed_sec": round(elapsed, 3),
        "rate_per_sec": round((counts["success"] + counts["error"]) / elapsed, 4),
        "feeder_error": feeder_error,
        "worker_errors": worker_errors,
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "processes": args.processes,
            "threads_per_process": args.threads_per_process,
            "doc_store_connections": args.doc_store_connections,
            "timeout": args.timeout,
            "retries": args.retries,
            "max_pairs": args.max_pairs,
        },
        "finished_at": utc_now(),
    }
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if feeder_error or worker_errors:
        raise RuntimeError(json.dumps(report, ensure_ascii=False))
    return report


def add_relation_arguments(parser: argparse.ArgumentParser) -> None:
    default_prompt = Path(__file__).resolve().parents[1] / "prompts" / "relation_classifier.txt"
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
    )
    parser.add_argument("--doc-store", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--prompt", type=Path, default=default_prompt)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", ""),
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", ""))
    parser.add_argument("--processes", type=int, default=20)
    parser.add_argument("--threads-per-process", type=int, default=20)
    parser.add_argument("--doc-store-connections", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--max-retry-sleep", type=float, default=30.0)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 processes every candidate")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-p", type=float)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="send a gateway-specific request to disable reasoning",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
