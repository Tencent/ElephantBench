from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


@dataclass
class OpenAIChatClient:
    base_url: str
    api_key: str
    model: str
    timeout: float = 600.0
    retries: int = 2
    extra_headers: dict[str, str] = field(default_factory=dict)

    @property
    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.model:
            payload["model"] = self.model
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                self.endpoint, data=data, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                choices = result.get("choices") or []
                if not choices:
                    raise ValueError("response has no choices")
                message = choices[0].get("message") or {}
                text = _message_text(message.get("content"))
                if not text:
                    raise ValueError("response has empty message content")
                return {
                    "content": text,
                    "finish_reason": choices[0].get("finish_reason"),
                    "usage": result.get("usage"),
                }
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(
            f"chat completion failed after {self.retries + 1} attempts: {last_error}"
        )


def parse_headers(values: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"header must be NAME=VALUE, got {value!r}")
        name, content = value.split("=", 1)
        if not name.strip():
            raise ValueError("header name cannot be empty")
        headers[name.strip()] = content
    return headers
