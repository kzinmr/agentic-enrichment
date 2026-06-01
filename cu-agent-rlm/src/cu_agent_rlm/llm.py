from __future__ import annotations

import json
import os
import threading
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

    def complete_json_with_schema(
        self,
        *,
        system: str,
        user: str,
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError


def empty_call_usage() -> dict[str, Any]:
    return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "total_cost_usd": 0.0}


class OpenAIResponsesClient:
    provider_name = "openai"
    supports_json_schema = True

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
        self._usage_lock = threading.Lock()
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is required for openai mode.")

    def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        payload, _ = self.complete_json_with_usage(system=system, user=user)
        return payload

    def complete_json_with_usage(self, *, system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]:
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
        call_usage = self.record_usage(response)
        return parse_json_object(extract_responses_text(response)), call_usage

    def complete_json_with_schema(
        self,
        *,
        system: str,
        user: str,
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "instructions": system,
            "input": f"Return JSON only.\n\n{user}",
            "text": {"format": response_format},
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

    def record_usage(self, response: dict[str, Any]) -> dict[str, Any]:
        input_tokens, output_tokens, total_tokens = usage_from_response(response)
        cost_usd = cost_for_tokens(input_tokens, output_tokens, self.pricing)
        # Mutations are serialized so concurrent batched extraction calls accumulate safely.
        with self._usage_lock:
            self.usage_summary.add_call(
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                pricing=self.pricing,
            )
        return {
            "calls": 1,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens if total_tokens is not None else input_tokens + output_tokens,
            "total_cost_usd": round(cost_usd, 10),
        }


class OpenAICompatibleChatClient:
    provider_name = "openai-compatible"

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENAI_MODEL,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        api_key: str | None = None,
        timeout_seconds: int = 120,
        supports_json_schema: bool | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.timeout_seconds = timeout_seconds
        self.supports_json_schema = (
            env_flag("OPENAI_COMPATIBLE_JSON_SCHEMA") if supports_json_schema is None else supports_json_schema
        )
        self.usage_summary = UsageSummary()
        self.pricing = pricing_from_env()
        self._usage_lock = threading.Lock()
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is required for openai-compatible extractor mode.")

    def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        payload, _ = self.complete_json_with_usage(system=system, user=user)
        return payload

    def complete_json_with_usage(self, *, system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]:
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
        call_usage = self.record_usage(response)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Chat completions response missing message content: {response}") from exc
        return parse_json_object(str(content)), call_usage

    def complete_json_with_schema(
        self,
        *,
        system: str,
        user: str,
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.supports_json_schema:
            return self.complete_json(system=system, user=user)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {key: value for key, value in response_format.items() if key != "type"},
            },
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

    def record_usage(self, response: dict[str, Any]) -> dict[str, Any]:
        input_tokens, output_tokens, total_tokens = usage_from_response(response)
        cost_usd = cost_for_tokens(input_tokens, output_tokens, self.pricing)
        with self._usage_lock:
            self.usage_summary.add_call(
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                pricing=self.pricing,
            )
        return {
            "calls": 1,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens if total_tokens is not None else input_tokens + output_tokens,
            "total_cost_usd": round(cost_usd, 10),
        }


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


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
