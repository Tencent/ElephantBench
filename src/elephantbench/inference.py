from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from .client import OpenAIChatClient, parse_headers
from .io import completed_ids, read_jsonl
from .schema import validate_item


def _run_one(
    record: dict[str, Any],
    client: OpenAIChatClient,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    started = time.time()
    benchmark_id = record["benchmark_id"]
    try:
        response = client.chat(
            system_prompt,
            record["eval"]["question"],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "benchmark_id": benchmark_id,
            "status": "success",
            "model": client.model,
            "answer": response["content"],
            "finish_reason": response.get("finish_reason"),
            "usage": response.get("usage"),
            "elapsed_seconds": round(time.time() - started, 3),
        }
    except Exception as exc:
        return {
            "benchmark_id": benchmark_id,
            "status": "error",
            "model": client.model,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.time() - started, 3),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run closed-book ElephantBench inference")
    default_prompt = Path(__file__).resolve().parent / "prompts" / "target_system.txt"
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-prompt", type=Path, default=default_prompt)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--header", action="append", default=[], metavar="NAME=VALUE")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    records = list(read_jsonl(args.input))
    for record in records:
        errors = validate_item(record)
        if errors:
            raise SystemExit(f"invalid item {record.get('benchmark_id')!r}: {'; '.join(errors)}")
    if args.limit > 0:
        records = records[: args.limit]
    done = completed_ids(args.output) if args.resume else set()
    pending = [record for record in records if record["benchmark_id"] not in done]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and args.output.exists() else "w"
    prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    client = OpenAIChatClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
        retries=args.retries,
        extra_headers=parse_headers(args.header),
    )
    lock = threading.Lock()
    with args.output.open(mode, encoding="utf-8", newline="\n") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(_run_one, record, client, prompt, args.temperature, args.max_tokens)
                for record in pending
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                with lock:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                print(f"{result['benchmark_id']}\t{result['status']}")
    print(f"wrote {len(pending)} records to {args.output}; skipped {len(done)} completed records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
