from __future__ import annotations

from typing import Any, Protocol

from .fields import extract_call_fields, extract_field, next_action, urgency
from .llm import JSONChatClient, LLMError
from .models import CallRecord, Chunk, FieldExtraction, FieldSpec
from .prompt_registry import FIELD_EXTRACTION_PROMPT, PromptRender


class ExtractionError(RuntimeError):
    pass


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
        prompt = build_extraction_prompt_render(call, chunks, specs)
        self.last_prompt = prompt.metadata()
        try:
            payload = self.llm.complete_json(system=prompt.system, user=prompt.user)
            return validate_extraction_payload(payload, call=call, chunks=chunks, specs=specs, extractor=self.name)
        except (LLMError, ValueError, TypeError) as exc:
            if self.fallback is None:
                raise ExtractionError(str(exc)) from exc
            fallback_extractions = self.fallback.extract_call_fields(call, chunks, specs)
            for extraction in fallback_extractions:
                extraction.validation_errors.append(f"llm_fallback:{exc}")
                extraction.rationale = f"{extraction.rationale}; LLM fallback because {exc}"
            return fallback_extractions


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
    return FIELD_EXTRACTION_PROMPT.render(
        {
            "call": {
                "call_id": call.call_id,
                "customer_id": call.customer_id,
                "account_name": call.account_name,
                "date": call.date,
                "metadata": {
                    "type": call.metadata.get("type"),
                    "is_connected": call.metadata.get("is_connected"),
                    "is_effective_connected": call.metadata.get("is_effective_connected"),
                    "best_texts_length": call.metadata.get("best_texts_length"),
                },
            },
            "fields": [
                {
                    "name": spec.name,
                    "type": spec.type,
                    "description": spec.description,
                    "allowed_values": spec.allowed_values,
                    "downstream_use_cases": spec.downstream_use_cases,
                }
                for spec in specs
            ],
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "evidence_ref": f"chunk:{chunk.chunk_id}",
                    "turn_start": chunk.turn_start,
                    "turn_end": chunk.turn_end,
                    "speaker_set": chunk.speaker_set,
                    "text": chunk.text,
                }
                for chunk in chunks
            ],
        }
    )


def validate_extraction_payload(
    payload: dict[str, Any],
    *,
    call: CallRecord,
    chunks: list[Chunk],
    specs: list[FieldSpec],
    extractor: str,
) -> list[FieldExtraction]:
    raw_fields = payload.get("fields")
    if not isinstance(raw_fields, list):
        raise ValueError("LLM extraction response must contain a fields array")

    spec_by_name = {spec.name: spec for spec in specs}
    chunk_refs = {f"chunk:{chunk.chunk_id}" for chunk in chunks}
    raw_by_name: dict[str, dict[str, Any]] = {}
    for item in raw_fields:
        if not isinstance(item, dict):
            raise ValueError("Each LLM field extraction must be an object")
        name = item.get("name")
        if not isinstance(name, str):
            raise ValueError("Each LLM field extraction needs a string name")
        if name not in spec_by_name:
            continue
        raw_by_name[name] = item

    results: list[FieldExtraction] = []
    for spec in specs:
        item = raw_by_name.get(spec.name)
        if item is None:
            results.append(default_extraction(call, spec, validation_errors=[f"{extractor}:missing_field"]))
            continue
        results.append(validate_field_item(item, call=call, spec=spec, chunk_refs=chunk_refs, extractor=extractor))
    return results


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
