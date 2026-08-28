from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from .client import OpenAIChatClient, parse_headers
from .io import completed_ids, load_by_id, read_jsonl

LABEL_ALIASES = {
    "complete": "complete",
    "full_credit": "complete",
    "partial": "partial",
    "partial_credit": "partial",
    "failed": "failed",
    "no_credit": "failed",
}


def build_judge_input(item: dict[str, Any], answer: str) -> str:
    evaluation = item["eval"]
    payload = {
        "question": evaluation["question"],
        "verified_answers": [gold["value"] for gold in evaluation["gold_answers"]],
        "model_answer": answer,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_judgment(text: str, expected_gold_count: int | None = None) -> dict[str, Any]:
    candidate = text.strip()
    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.DOTALL | re.IGNORECASE
    )
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    value = json.loads(candidate)
    raw_label = str(value.get("label") or value.get("credit") or "").strip().lower()
    label = LABEL_ALIASES.get(raw_label)
    if label is None:
        raise ValueError(f"unknown judge label or credit {raw_label!r}")

    assessments = value.get("gold_assessments")
    contradictions = value.get("material_contradictions")
    if assessments is not None or contradictions is not None:
        if not isinstance(assessments, list):
            raise ValueError("gold_assessments must be an array")
        if expected_gold_count is not None and len(assessments) != expected_gold_count:
            raise ValueError(
                f"gold_assessments has {len(assessments)} entries; expected {expected_gold_count}"
            )
        if not all(
            isinstance(item, dict) and isinstance(item.get("covered"), bool) for item in assessments
        ):
            raise ValueError("each gold assessment requires a boolean covered field")
        if not isinstance(contradictions, list):
            raise ValueError("material_contradictions must be an array")
        covered = sum(bool(item["covered"]) for item in assessments)
        expected_label = (
            "failed"
            if contradictions or covered == 0
            else "complete"
            if covered == len(assessments)
            else "partial"
        )
        if label != expected_label:
            raise ValueError(
                f"judge credit is inconsistent with coverage: {label!r} != {expected_label!r}"
            )

    rationale = str(value.get("rationale") or value.get("reasoning") or "").strip()
    parsed: dict[str, Any] = {"label": label, "rationale": rationale}
    if assessments is not None:
        parsed["gold_assessments"] = assessments
        parsed["material_contradictions"] = contradictions
        parsed["credit"] = raw_label
    return parsed


def _judge_one(
    response: dict[str, Any],
    item: dict[str, Any],
    client: OpenAIChatClient,
    system_prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    started = time.time()
    output = dict(response)
    if response.get("status") != "success" or not isinstance(response.get("answer"), str):
        output["judgment"] = {
            "label": "failed",
            "rationale": "Target-model request failed or produced no scorable answer.",
            "judge_model": client.model,
        }
        return output
    try:
        result = client.chat(
            system_prompt,
            build_judge_input(item, response["answer"]),
            temperature=0.0,
            max_tokens=max_tokens,
        )
        judgment = parse_judgment(
            result["content"], expected_gold_count=len(item["eval"]["gold_answers"])
        )
        judgment.update(
            {
                "judge_model": client.model,
                "raw_response": result["content"],
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
        output["judgment"] = judgment
        output["status"] = "success"
    except Exception as exc:
        output["judgment"] = {
            "label": "failed",
            "rationale": f"Judge error: {type(exc).__name__}: {exc}",
            "judge_model": client.model,
            "judge_error": True,
        }
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Judge saved ElephantBench responses")
    default_prompt = Path(__file__).resolve().parent / "prompts" / "judge_system.txt"
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-prompt", type=Path, default=default_prompt)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--header", action="append", default=[], metavar="NAME=VALUE")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    benchmark = load_by_id(args.benchmark)
    responses = list(read_jsonl(args.responses))
    seen: set[str] = set()
    for response in responses:
        benchmark_id = response.get("benchmark_id")
        if benchmark_id not in benchmark:
            raise SystemExit(f"response ID not found in benchmark: {benchmark_id!r}")
        if benchmark_id in seen:
            raise SystemExit(f"duplicate response ID: {benchmark_id!r}")
        seen.add(str(benchmark_id))
    done = completed_ids(args.output) if args.resume else set()
    pending = [response for response in responses if response["benchmark_id"] not in done]
    client = OpenAIChatClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.judge_model,
        timeout=args.timeout,
        retries=args.retries,
        extra_headers=parse_headers(args.header),
    )
    prompt = args.judge_prompt.read_text(encoding="utf-8").strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and args.output.exists() else "w"
    lock = threading.Lock()
    with args.output.open(mode, encoding="utf-8", newline="\n") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    _judge_one,
                    response,
                    benchmark[response["benchmark_id"]],
                    client,
                    prompt,
                    args.max_tokens,
                )
                for response in pending
            ]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                with lock:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                print(f"{row['benchmark_id']}\t{row['judgment']['label']}")
    print(f"wrote {len(pending)} judged records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
