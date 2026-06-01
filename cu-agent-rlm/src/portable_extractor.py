from __future__ import annotations

from typing import Any


EVIDENCE_REF_PREFIX = "chunk:"
FIELD_TYPES = {"boolean", "enum", "list", "string"}
CONFIDENCE_VALUES = ("low", "medium", "high")


def build_user_payload(field_specs: list[Any], chunks: list[Any], call: Any | None = None) -> dict[str, Any]:
    """Build the field-extraction user payload from portable artifact rows.

    The function accepts plain dictionaries or dataclass-like objects so an external job can
    consume ``field_specs.jsonl`` and ``chunks.jsonl`` without importing ``cu_agent_rlm``.
    """

    call_payload = _call_payload(call, chunks)
    return {
        "call": call_payload,
        "fields": [
            {
                "name": str(_get(spec, "name")),
                "type": str(_get(spec, "type")),
                "description": str(_get(spec, "description", "")),
                "allowed_values": list(_get(spec, "allowed_values", []) or []),
                "downstream_use_cases": list(_get(spec, "downstream_use_cases", []) or []),
            }
            for spec in field_specs
        ],
        "chunks": [
            {
                "chunk_id": str(_get(chunk, "chunk_id")),
                "evidence_ref": evidence_ref(str(_get(chunk, "chunk_id"))),
                "turn_start": _get(chunk, "turn_start"),
                "turn_end": _get(chunk, "turn_end"),
                "speaker_set": list(_get(chunk, "speaker_set", []) or []),
                "text": str(_get(chunk, "text", "")),
            }
            for chunk in chunks
        ],
    }


def derive_response_schema(field_specs: list[Any]) -> dict[str, Any]:
    """Derive the strict JSON Schema for the extraction response envelope."""

    field_schemas = [_field_response_schema(spec) for spec in field_specs]
    item_schema: dict[str, Any]
    if len(field_schemas) == 1:
        item_schema = field_schemas[0]
    elif field_schemas:
        item_schema = {"anyOf": field_schemas}
    else:
        item_schema = _generic_field_response_schema()
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["fields"],
        "properties": {
            "fields": {
                "type": "array",
                "items": item_schema,
            }
        },
    }


def validate(
    payload: dict[str, Any],
    field_specs: list[Any],
    chunk_ids: list[str] | set[str] | tuple[str, ...],
    *,
    call_id: str = "",
    extractor: str = "portable_extractor",
) -> list[dict[str, Any]]:
    """Validate and normalize an extraction response into portable row dictionaries."""

    raw_fields = payload.get("fields")
    if not isinstance(raw_fields, list):
        raise ValueError("LLM extraction response must contain a fields array")

    spec_by_name = {str(_get(spec, "name")): spec for spec in field_specs}
    chunk_refs = {_as_evidence_ref(chunk_id) for chunk_id in chunk_ids}
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

    results: list[dict[str, Any]] = []
    for spec in field_specs:
        name = str(_get(spec, "name"))
        item = raw_by_name.get(name)
        if item is None:
            results.append(
                default_extraction(
                    call_id,
                    spec,
                    validation_errors=[f"{extractor}:missing_field"],
                )
            )
            continue
        results.append(
            validate_field_item(
                item,
                call_id=call_id,
                spec=spec,
                chunk_refs=chunk_refs,
                extractor=extractor,
            )
        )
    return results


def validate_field_item(
    item: dict[str, Any],
    *,
    call_id: str,
    spec: Any,
    chunk_refs: set[str],
    extractor: str,
) -> dict[str, Any]:
    errors: list[str] = []
    value = coerce_value(item.get("value"), spec, errors)
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
    if confidence not in CONFIDENCE_VALUES:
        errors.append("invalid_confidence")
        confidence = "low"

    rationale = item.get("rationale", "")
    if not isinstance(rationale, str):
        errors.append("invalid_rationale")
        rationale = ""
    prefix = f"validated:{extractor}"
    rationale = f"{prefix}; {rationale}" if rationale else prefix

    return {
        "call_id": call_id,
        "field_name": str(_get(spec, "name")),
        "value": value,
        "confidence": confidence,
        "evidence_refs": refs,
        "abstained": abstained,
        "rationale": rationale,
        "validation_errors": errors,
    }


def coerce_value(value: Any, spec: Any, errors: list[str]) -> Any:
    field_type = str(_get(spec, "type"))
    allowed_values = list(_get(spec, "allowed_values", []) or [])
    if field_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        errors.append("expected_boolean")
        return False
    if field_type == "list":
        raw_values = value if isinstance(value, list) else ([] if value in (None, "", "not_mentioned") else [value])
        if not isinstance(raw_values, list):
            errors.append("expected_list")
            raw_values = []
        cleaned: list[str] = []
        for item in raw_values:
            if not isinstance(item, str):
                errors.append("list_item_not_string")
                continue
            if allowed_values and item not in allowed_values:
                errors.append(f"value_not_allowed:{item}")
                continue
            cleaned.append(item)
        return _dedupe(cleaned)
    if field_type == "enum":
        if isinstance(value, str) and value in allowed_values:
            return value
        errors.append(f"value_not_allowed:{value}")
        if "not_mentioned" in allowed_values:
            return "not_mentioned"
        return allowed_values[0] if allowed_values else "not_mentioned"
    if field_type == "string":
        if value in (None, ""):
            return "not_mentioned"
        if not isinstance(value, str):
            errors.append("expected_string")
            return str(value)
        return value
    return value


def default_extraction(
    call_id: str,
    spec: Any,
    *,
    validation_errors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "call_id": call_id,
        "field_name": str(_get(spec, "name")),
        "value": default_value(spec),
        "confidence": "low",
        "evidence_refs": [],
        "abstained": True,
        "rationale": "defaulted by extraction validator",
        "validation_errors": validation_errors or [],
    }


def default_value(spec: Any) -> Any:
    field_type = str(_get(spec, "type"))
    allowed_values = list(_get(spec, "allowed_values", []) or [])
    if field_type == "boolean":
        return False
    if field_type == "list":
        return []
    if field_type == "enum":
        if "not_mentioned" in allowed_values:
            return "not_mentioned"
        return allowed_values[0] if allowed_values else "not_mentioned"
    return "not_mentioned"


def is_empty_value(value: Any, spec: Any) -> bool:
    if str(_get(spec, "type")) == "boolean":
        return value is False
    return value in (None, "", [], "not_mentioned")


def evidence_ref(chunk_id: str) -> str:
    return _as_evidence_ref(chunk_id)


def _field_response_schema(spec: Any) -> dict[str, Any]:
    name = str(_get(spec, "name"))
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "value", "confidence", "evidence_refs", "abstained", "rationale"],
        "properties": {
            "name": {"type": "string", "enum": [name]},
            "value": _value_schema(spec),
            "confidence": {"type": "string", "enum": list(CONFIDENCE_VALUES)},
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
            "abstained": {"type": "boolean"},
            "rationale": {"type": "string"},
        },
    }


def _generic_field_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "value", "confidence", "evidence_refs", "abstained", "rationale"],
        "properties": {
            "name": {"type": "string"},
            "value": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "boolean"},
                    {"type": "array", "items": {"type": "string"}},
                ]
            },
            "confidence": {"type": "string", "enum": list(CONFIDENCE_VALUES)},
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
            "abstained": {"type": "boolean"},
            "rationale": {"type": "string"},
        },
    }


def _value_schema(spec: Any) -> dict[str, Any]:
    field_type = str(_get(spec, "type"))
    allowed_values = list(_get(spec, "allowed_values", []) or [])
    if field_type == "boolean":
        return {"type": "boolean"}
    if field_type == "list":
        item_schema: dict[str, Any] = {"type": "string"}
        if allowed_values:
            item_schema["enum"] = allowed_values
        return {"type": "array", "items": item_schema}
    if field_type == "enum":
        schema: dict[str, Any] = {"type": "string"}
        if allowed_values:
            schema["enum"] = allowed_values
        return schema
    return {"type": "string"}


def _call_payload(call: Any | None, chunks: list[Any]) -> dict[str, Any]:
    if call is not None:
        metadata = _get(call, "metadata", {}) or {}
        return {
            "call_id": _get(call, "call_id"),
            "customer_id": _get(call, "customer_id"),
            "account_name": _get(call, "account_name"),
            "date": _get(call, "date"),
            "metadata": {
                "type": _mapping_get(metadata, "type"),
                "is_connected": _mapping_get(metadata, "is_connected"),
                "is_effective_connected": _mapping_get(metadata, "is_effective_connected"),
                "best_texts_length": _mapping_get(metadata, "best_texts_length"),
            },
        }
    call_id = str(_get(chunks[0], "call_id")) if chunks else None
    return {
        "call_id": call_id,
        "customer_id": None,
        "account_name": None,
        "date": None,
        "metadata": {
            "type": None,
            "is_connected": None,
            "is_effective_connected": None,
            "best_texts_length": None,
        },
    }


def _as_evidence_ref(chunk_id: str) -> str:
    return chunk_id if chunk_id.startswith(EVIDENCE_REF_PREFIX) else f"{EVIDENCE_REF_PREFIX}{chunk_id}"


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _mapping_get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else default


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
