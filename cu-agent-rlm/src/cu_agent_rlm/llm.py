from __future__ import annotations

import json
import os
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .usage import UsageSummary, cost_for_tokens, pricing_from_env, usage_from_response


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"


class LLMError(RuntimeError):
    pass


class JSONChatClient(Protocol):
    provider_name: str

    def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        raise NotImplementedError


class OpenAIResponsesClient:
    provider_name = "openai"

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENAI_MODEL,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        api_key: str | None = None,
        timeout_seconds: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.timeout_seconds = timeout_seconds
        self.usage_summary = UsageSummary()
        self.pricing = pricing_from_env()
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is required for openai mode.")

    def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "instructions": system,
            "input": f"Return JSON only.\n\n{user}",
            "text": {"format": {"type": "json_object"}},
            "store": False,
        }
        response = post_json(
            f"{self.base_url}/responses",
            payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout_seconds=self.timeout_seconds,
        )
        self.record_usage(response)
        return parse_json_object(extract_responses_text(response))

    def record_usage(self, response: dict[str, Any]) -> None:
        input_tokens, output_tokens, total_tokens = usage_from_response(response)
        self.usage_summary.add_call(
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_for_tokens(input_tokens, output_tokens, self.pricing),
            pricing=self.pricing,
        )


class OpenAICompatibleChatClient:
    provider_name = "openai-compatible"

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENAI_MODEL,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        api_key: str | None = None,
        timeout_seconds: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.timeout_seconds = timeout_seconds
        self.usage_summary = UsageSummary()
        self.pricing = pricing_from_env()
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is required for openai-compatible extractor mode.")

    def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        response = post_json(
            f"{self.base_url}/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout_seconds=self.timeout_seconds,
        )
        self.record_usage(response)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Chat completions response missing message content: {response}") from exc
        return parse_json_object(str(content))

    def record_usage(self, response: dict[str, Any]) -> None:
        input_tokens, output_tokens, total_tokens = usage_from_response(response)
        self.usage_summary.add_call(
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_for_tokens(input_tokens, output_tokens, self.pricing),
            pricing=self.pricing,
        )


def extract_responses_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    parts: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                parts.append(content["text"])
            elif content.get("type") == "refusal":
                raise LLMError(f"OpenAI response refused the request: {content.get('refusal')}")
    if parts:
        return "\n".join(parts)
    raise LLMError(f"OpenAI Responses API response missing output text: {response}")


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: int,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise LLMError(f"LLM HTTP error from {url}: {exc.code} {details}") from exc
    except URLError as exc:
        raise LLMError(f"LLM connection error from {url}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise LLMError(f"LLM endpoint did not return JSON: {url}") from exc


def parse_json_object(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise LLMError(f"LLM response was not a JSON object: {content[:200]}")
        try:
            payload = json.loads(content[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM response contained invalid JSON: {content[:200]}") from exc
    if not isinstance(payload, dict):
        raise LLMError(f"LLM response must be a JSON object, got {type(payload).__name__}")
    return payload
