from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from typing import Any


@dataclass
class ModelUsageSummary:
    total_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    def add_call(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int | None = None,
        cost_usd: float = 0.0,
    ) -> None:
        self.total_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += total_tokens if total_tokens is not None else input_tokens + output_tokens
        self.total_cost_usd += cost_usd

    def merge(self, other: "ModelUsageSummary") -> None:
        self.total_calls += other.total_calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.total_cost_usd += other.total_cost_usd


@dataclass
class UsageSummary:
    total_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    pricing: dict[str, Any] = field(default_factory=dict)
    by_model: dict[str, ModelUsageSummary] = field(default_factory=dict)

    def add_call(
        self,
        *,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int | None = None,
        cost_usd: float = 0.0,
        pricing: dict[str, Any] | None = None,
    ) -> None:
        resolved_total = total_tokens if total_tokens is not None else input_tokens + output_tokens
        self.total_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += resolved_total
        self.total_cost_usd += cost_usd
        self.pricing = pricing or self.pricing
        self.by_model.setdefault(model, ModelUsageSummary()).add_call(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=resolved_total,
            cost_usd=cost_usd,
        )

    def merge(self, other: "UsageSummary") -> None:
        self.total_calls += other.total_calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.total_cost_usd += other.total_cost_usd
        self.pricing = other.pricing or self.pricing
        for model, model_usage in other.by_model.items():
            self.by_model.setdefault(model, ModelUsageSummary()).merge(model_usage)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["total_cost_usd"] = round(self.total_cost_usd, 10)
        for model_usage in payload["by_model"].values():
            model_usage["total_cost_usd"] = round(model_usage["total_cost_usd"], 10)
        return payload


def usage_from_response(response: dict[str, Any]) -> tuple[int, int, int | None]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return 0, 0, None
    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    total_tokens = usage.get("total_tokens")
    return input_tokens, output_tokens, int(total_tokens) if isinstance(total_tokens, int | float) else None


def pricing_from_env() -> dict[str, Any]:
    input_rate = _float_env("OPENAI_INPUT_USD_PER_MTOK")
    output_rate = _float_env("OPENAI_OUTPUT_USD_PER_MTOK")
    if input_rate is None or output_rate is None:
        return {"source": "unpriced", "input_usd_per_mtok": None, "output_usd_per_mtok": None}
    return {
        "source": "env",
        "input_usd_per_mtok": input_rate,
        "output_usd_per_mtok": output_rate,
    }


def cost_for_tokens(input_tokens: int, output_tokens: int, pricing: dict[str, Any]) -> float:
    input_rate = pricing.get("input_usd_per_mtok")
    output_rate = pricing.get("output_usd_per_mtok")
    if not isinstance(input_rate, int | float) or not isinstance(output_rate, int | float):
        return 0.0
    return input_tokens * float(input_rate) / 1_000_000 + output_tokens * float(output_rate) / 1_000_000


def usage_summary_from_components(*components: object) -> dict[str, Any]:
    summary = UsageSummary()
    seen: set[int] = set()
    for component in components:
        _merge_component_usage(summary, component, seen)
    return summary.to_dict()


def _merge_component_usage(summary: UsageSummary, component: object, seen: set[int]) -> None:
    if component is None:
        return
    identity = id(component)
    if identity in seen:
        return
    seen.add(identity)
    usage_summary = getattr(component, "usage_summary", None)
    if isinstance(usage_summary, UsageSummary):
        summary.merge(usage_summary)
    for attr in ("llm", "controller", "reranker", "base"):
        nested = getattr(component, attr, None)
        if nested is not None:
            _merge_component_usage(summary, nested, seen)


def _float_env(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None
