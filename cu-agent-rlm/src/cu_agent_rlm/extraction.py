from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import time
from typing import Any, Protocol

from portable_extractor import build_user_payload, validate as portable_validate

from .fields import extract_call_fields, extract_field, next_action, urgency
from .llm import JSONChatClient, LLMError, empty_call_usage
from .models import CallRecord, Chunk, FieldExtraction, FieldSpec
from .prompt_registry import FIELD_EXTRACTION_PROMPT, PromptRender
from .usage import usage_delta, usage_summary_from_components


class ExtractionError(RuntimeError):
    pass


@dataclass
class CallExtractionOutcome:
    """Per-call extraction result with the telemetry the pipeline needs to trace it.

    ``error`` is set only when extraction raised without a fallback (a hard failure the
    guardrail counts); otherwise ``extractions`` holds the validated/fallback rows.
    """

    call: CallRecord
    extractions: list[FieldExtraction]
    latency_ms: float
    usage: dict[str, Any] = field(default_factory=empty_call_usage)
    prompt_hash: str | None = None
    error: str | None = None


class FieldExtractor(Protocol):
    name: str

    def extract_call_fields(
        self,
        call: CallRecord,
        chunks: list[Chunk],
        specs: list[FieldSpec],
    ) -> list[FieldExtraction]:
        raise NotImplementedError


class HeuristicFieldExtractor:
    name = "heuristic"

    def extract_call_fields(
        self,
        call: CallRecord,
        chunks: list[Chunk],
        specs: list[FieldSpec],
    ) -> list[FieldExtraction]:
        legacy_by_name = {extraction.field_name: extraction for extraction in extract_call_fields(call, chunks)}
        results: list[FieldExtraction] = []
        for spec in specs:
            legacy = legacy_by_name.get(spec.name)
            if legacy is not None:
                results.append(legacy)
                continue
            results.append(generic_extract_field(call, chunks, spec))
        return results


class LLMFieldExtractor:
    def __init__(
        self,
        llm: JSONChatClient,
        *,
        fallback: FieldExtractor | None = None,
    ) -> None:
        self.llm = llm
        self.fallback = fallback
        self.name = llm.provider_name
        self.last_prompt: dict[str, Any] | None = None

    def extract_call_fields(
        self,
        call: CallRecord,
        chunks: list[Chunk],
        specs: list[FieldSpec],
    ) -> list[FieldExtraction]:
        outcome = self._extract_one(call, chunks, specs)
        if outcome.error is not None:
            raise ExtractionError(outcome.error)
        return outcome.extractions

    def extract_batch(
        self,
        calls: list[CallRecord],
        chunks_by_call: dict[str, list[Chunk]],
        specs: list[FieldSpec],
        *,
        max_concurrent: int = 1,
    ) -> list[CallExtractionOutcome]:
        # Per-call extraction is embarrassingly parallel: each call's prompt, validation, and
        # fallback are independent. We run them with bounded concurrency while preserving input
        # order so the materialized result is identical to the sequential path.
        if max_concurrent <= 1 or len(calls) <= 1:
            return [self._extract_one(call, chunks_by_call.get(call.call_id, []), specs) for call in calls]
        outcomes: list[CallExtractionOutcome | None] = [None] * len(calls)
        with ThreadPoolExecutor(max_workers=min(max_concurrent, len(calls))) as executor:
            futures = {
                executor.submit(self._extract_one, call, chunks_by_call.get(call.call_id, []), specs): index
                for index, call in enumerate(calls)
            }
            for future in as_completed(futures):
                outcomes[futures[future]] = future.result()
        return [outcome for outcome in outcomes if outcome is not None]

    def _extract_one(
        self,
        call: CallRecord,
        chunks: list[Chunk],
        specs: list[FieldSpec],
    ) -> CallExtractionOutcome:
        prompt = build_extraction_prompt_render(call, chunks, specs)
        metadata = prompt.metadata()
        self.last_prompt = metadata
        prompt_hash = metadata.get("prompt_hash") if isinstance(metadata, dict) else None
        started = time.perf_counter()
        usage = empty_call_usage()
        try:
            payload, usage = self._llm_complete(prompt.system, prompt.user)
            extractions = validate_extraction_payload(payload, call=call, chunks=chunks, specs=specs, extractor=self.name)
            return CallExtractionOutcome(
                call=call,
                extractions=extractions,
                latency_ms=_elapsed_ms(started),
                usage=usage,
                prompt_hash=prompt_hash,
            )
        except (LLMError, ValueError, TypeError) as exc:
            if self.fallback is None:
                return CallExtractionOutcome(
                    call=call,
                    extractions=[],
                    latency_ms=_elapsed_ms(started),
                    usage=usage,
                    prompt_hash=prompt_hash,
                    error=str(exc)[:240],
                )
            fallback_extractions = self.fallback.extract_call_fields(call, chunks, specs)
            for extraction in fallback_extractions:
                extraction.validation_errors.append(f"llm_fallback:{exc}")
                extraction.rationale = f"{extraction.rationale}; LLM fallback because {exc}"
            return CallExtractionOutcome(
                call=call,
                extractions=fallback_extractions,
                latency_ms=_elapsed_ms(started),
                usage=usage,
                prompt_hash=prompt_hash,
            )

    def _llm_complete(self, system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]:
        # Prefer the usage-returning entrypoint so per-call token attribution stays accurate
        # under concurrency; fall back to a usage snapshot for clients that lack it.
        method = getattr(self.llm, "complete_json_with_usage", None)
        if callable(method):
            payload, usage = method(system=system, user=user)
            return payload, usage if isinstance(usage, dict) else empty_call_usage()
        before = usage_summary_from_components(self.llm)
        payload = self.llm.complete_json(system=system, user=user)
        after = usage_summary_from_components(self.llm)
        return payload, usage_delta(after, before)


def run_call_extractions(
    extractor: FieldExtractor,
    calls: list[CallRecord],
    chunks_by_call: dict[str, list[Chunk]],
    specs: list[FieldSpec],
    *,
    max_concurrent: int = 1,
) -> list[CallExtractionOutcome]:
    """Run per-call extraction, using the extractor's native batch path when available.

    Extractors that implement ``extract_batch`` (e.g. the LLM extractor) get bounded
    concurrency; others fall back to a deterministic sequential loop that still yields the
    same per-call telemetry, so the pipeline treats both uniformly.
    """
    native = getattr(extractor, "extract_batch", None)
    if callable(native):
        return native(calls, chunks_by_call, specs, max_concurrent=max_concurrent)
    return [
        _sequential_outcome(extractor, call, chunks_by_call.get(call.call_id, []), specs)
        for call in calls
    ]


def _sequential_outcome(
    extractor: FieldExtractor,
    call: CallRecord,
    chunks: list[Chunk],
    specs: list[FieldSpec],
) -> CallExtractionOutcome:
    before = usage_summary_from_components(extractor)
    started = time.perf_counter()
    try:
        extractions = extractor.extract_call_fields(call, chunks, specs)
        error = None
    except (ExtractionError, ValueError, TypeError) as exc:
        extractions = []
        error = str(exc)[:240]
    return CallExtractionOutcome(
        call=call,
        extractions=extractions,
        latency_ms=_elapsed_ms(started),
        usage=usage_delta(usage_summary_from_components(extractor), before),
        prompt_hash=_prompt_hash(extractor),
        error=error,
    )


def _prompt_hash(component: object) -> str | None:
    prompt = getattr(component, "last_prompt", None)
    if isinstance(prompt, dict):
        value = prompt.get("prompt_hash")
        return str(value) if value else None
    return None


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)


def build_extraction_prompt(
    call: CallRecord,
    chunks: list[Chunk],
    specs: list[FieldSpec],
) -> tuple[str, str]:
    prompt = build_extraction_prompt_render(call, chunks, specs)
    return prompt.system, prompt.user


def build_extraction_prompt_render(
    call: CallRecord,
    chunks: list[Chunk],
    specs: list[FieldSpec],
) -> PromptRender:
    return FIELD_EXTRACTION_PROMPT.render(build_user_payload(specs, chunks, call=call))


def validate_extraction_payload(
    payload: dict[str, Any],
    *,
    call: CallRecord,
    chunks: list[Chunk],
    specs: list[FieldSpec],
    extractor: str,
) -> list[FieldExtraction]:
    rows = portable_validate(
        payload,
        specs,
        {chunk.chunk_id for chunk in chunks},
        call_id=call.call_id,
        extractor=extractor,
    )
    return [FieldExtraction(**row) for row in rows]


def generic_extract_field(call: CallRecord, chunks: list[Chunk], spec: FieldSpec) -> FieldExtraction:
    if spec.name == "urgency":
        value, refs = urgency(call.transcript.lower(), chunks)
        return FieldExtraction(
            call_id=call.call_id,
            field_name=spec.name,
            value=value,
            confidence="high" if refs else "low",
            evidence_refs=refs,
            abstained=value == "not_mentioned",
            rationale="generic urgency heuristic",
        )
    if "action" in spec.name:
        value, refs = next_action(call, chunks)
        return FieldExtraction(
            call_id=call.call_id,
            field_name=spec.name,
            value=value,
            confidence="high" if refs else "low",
            evidence_refs=refs,
            abstained=value == "not_mentioned",
            rationale="generic next-action heuristic",
        )
    if spec.type == "list":
        values: list[str] = []
        refs: list[str] = []
        for allowed in spec.allowed_values:
            matched_refs = refs_for_label(chunks, allowed, limit=2)
            if matched_refs:
                values.append(allowed)
                refs.extend(matched_refs)
        return FieldExtraction(
            call_id=call.call_id,
            field_name=spec.name,
            value=dedupe(values),
            confidence="medium" if refs else "low",
            evidence_refs=dedupe(refs)[:4],
            abstained=not values,
            rationale="generic induced-schema label matcher",
        )
    if spec.type == "enum":
        for allowed in spec.allowed_values:
            if allowed == "not_mentioned":
                continue
            refs = refs_for_label(chunks, allowed, limit=2)
            if refs:
                return FieldExtraction(
                    call_id=call.call_id,
                    field_name=spec.name,
                    value=allowed,
                    confidence="medium",
                    evidence_refs=refs,
                    abstained=False,
                    rationale="generic induced-schema enum matcher",
                )
        return default_extraction(call, spec)
    if spec.type == "boolean":
        refs = refs_for_label(chunks, spec.name, limit=2)
        return FieldExtraction(
            call_id=call.call_id,
            field_name=spec.name,
            value=bool(refs),
            confidence="medium" if refs else "low",
            evidence_refs=refs,
            abstained=not refs,
            rationale="generic induced-schema boolean matcher",
        )
    return default_extraction(call, spec)


def refs_for_label(chunks: list[Chunk], label: str, *, limit: int) -> list[str]:
    terms = [term for term in label.replace("_", " ").split() if len(term) > 2]
    refs: list[str] = []
    for chunk in chunks:
        text = chunk.text.lower()
        if label.replace("_", " ") in text or (terms and all(term in text for term in terms)):
            refs.append(f"chunk:{chunk.chunk_id}")
        if len(refs) >= limit:
            break
    return refs


def validate_field_item(
    item: dict[str, Any],
    *,
    call: CallRecord,
    spec: FieldSpec,
    chunk_refs: set[str],
    extractor: str,
) -> FieldExtraction:
    errors: list[str] = []
    raw_value = item.get("value")
    value = coerce_value(raw_value, spec, errors)
    raw_refs = item.get("evidence_refs", [])
    if raw_refs is None:
        raw_refs = []
    if not isinstance(raw_refs, list):
        errors.append("evidence_refs_not_list")
        raw_refs = []
    refs = [ref for ref in raw_refs if isinstance(ref, str) and ref in chunk_refs]
    invalid_refs = [ref for ref in raw_refs if not isinstance(ref, str) or ref not in chunk_refs]
    if invalid_refs:
        errors.append("invalid_evidence_refs")

    abstained = item.get("abstained", is_empty_value(value, spec))
    if not isinstance(abstained, bool):
        errors.append("abstained_not_boolean")
        abstained = is_empty_value(value, spec)
    if not abstained and not refs:
        errors.append("missing_evidence_refs")

    confidence = item.get("confidence", "low")
    if confidence not in {"low", "medium", "high"}:
        errors.append("invalid_confidence")
        confidence = "low"

    rationale = item.get("rationale", "")
    if not isinstance(rationale, str):
        errors.append("invalid_rationale")
        rationale = ""
    prefix = f"validated:{extractor}"
    rationale = f"{prefix}; {rationale}" if rationale else prefix

    return FieldExtraction(
        call_id=call.call_id,
        field_name=spec.name,
        value=value,
        confidence=confidence,
        evidence_refs=refs,
        abstained=abstained,
        rationale=rationale,
        validation_errors=errors,
    )


def coerce_value(value: Any, spec: FieldSpec, errors: list[str]) -> Any:
    if spec.type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        errors.append("expected_boolean")
        return False
    if spec.type == "list":
        raw_values = value if isinstance(value, list) else ([] if value in (None, "", "not_mentioned") else [value])
        if not isinstance(raw_values, list):
            errors.append("expected_list")
            raw_values = []
        cleaned = []
        for item in raw_values:
            if not isinstance(item, str):
                errors.append("list_item_not_string")
                continue
            if spec.allowed_values and item not in spec.allowed_values:
                errors.append(f"value_not_allowed:{item}")
                continue
            cleaned.append(item)
        return dedupe(cleaned)
    if spec.type == "enum":
        if isinstance(value, str) and value in spec.allowed_values:
            return value
        errors.append(f"value_not_allowed:{value}")
        return "not_mentioned" if "not_mentioned" in spec.allowed_values else spec.allowed_values[0]
    if spec.type == "string":
        if value in (None, ""):
            return "not_mentioned"
        if not isinstance(value, str):
            errors.append("expected_string")
            return str(value)
        return value
    return value


def default_extraction(
    call: CallRecord,
    spec: FieldSpec,
    *,
    validation_errors: list[str] | None = None,
) -> FieldExtraction:
    return FieldExtraction(
        call_id=call.call_id,
        field_name=spec.name,
        value=default_value(spec),
        confidence="low",
        evidence_refs=[],
        abstained=True,
        rationale="defaulted by extraction validator",
        validation_errors=validation_errors or [],
    )


def default_value(spec: FieldSpec) -> Any:
    if spec.type == "boolean":
        return False
    if spec.type == "list":
        return []
    if spec.type == "enum":
        return "not_mentioned" if "not_mentioned" in spec.allowed_values else spec.allowed_values[0]
    return "not_mentioned"


def is_empty_value(value: Any, spec: FieldSpec) -> bool:
    if spec.type == "boolean":
        return value is False
    return value in (None, "", [], "not_mentioned")


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
