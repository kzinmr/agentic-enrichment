from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any


FIELD_CANDIDATE_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["field_candidates"],
    "properties": {
        "field_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["candidate_id", "field_name", "field_type", "intended_queries"],
                "properties": {
                    "candidate_id": {"type": "string"},
                    "field_name": {"type": "string"},
                    "field_type": {"type": "string"},
                    "intended_queries": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}


@dataclass(frozen=True)
class TypedSubAgentCall:
    call_id: str
    parent_id: str
    agent: str
    capability: str
    input_refs: list[str]
    input_payload: dict[str, Any]
    output_schema: dict[str, Any]
    budget: dict[str, Any]
    validator: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TypedSubAgentResult:
    call: TypedSubAgentCall
    status: str
    validation_result: str
    output_refs: list[str] = field(default_factory=list)
    result_summary: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["call"] = self.call.to_dict()
        return payload


def field_candidate_subagent_call(
    *,
    parent_id: str,
    input_refs: list[str],
    field_count: int,
    max_candidates: int,
) -> TypedSubAgentCall:
    payload = {"field_count": field_count, "max_candidates": max_candidates}
    return TypedSubAgentCall(
        call_id=stable_call_id(parent_id, "cu", "field_candidate_proposal", payload),
        parent_id=parent_id,
        agent="cu",
        capability="field_candidate_proposal",
        input_refs=input_refs,
        input_payload=payload,
        output_schema=FIELD_CANDIDATE_OUTPUT_SCHEMA,
        budget={"max_candidates": max_candidates},
        validator="cu.field_candidates.output@2026-06-02.1",
    )


def complete_field_candidate_call(
    call: TypedSubAgentCall,
    candidates: list[dict[str, Any]],
) -> TypedSubAgentResult:
    for candidate in candidates:
        for key in ("candidate_id", "field_name", "field_type", "intended_queries"):
            if key not in candidate:
                return TypedSubAgentResult(
                    call=call,
                    status="error",
                    validation_result="error",
                    error=f"field candidate missing required key: {key}",
                )
    return TypedSubAgentResult(
        call=call,
        status="ok",
        validation_result="ok",
        output_refs=[f"field_candidate:{candidate['candidate_id']}" for candidate in candidates],
        result_summary=f"{len(candidates)} field candidates",
    )


def stable_call_id(parent_id: str, agent: str, capability: str, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {"parent_id": parent_id, "agent": agent, "capability": capability, "payload": payload},
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:12]
    return f"subcall-{digest}"
