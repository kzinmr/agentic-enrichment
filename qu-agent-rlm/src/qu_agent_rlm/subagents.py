from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any


RETRIEVAL_BRANCH_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["chunks"],
    "properties": {
        "chunks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["chunk_id", "call_id"],
                "properties": {
                    "chunk_id": {"type": "string"},
                    "call_id": {"type": "string"},
                    "snippet": {"type": "string"},
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


def retrieval_branch_call(
    *,
    parent_id: str,
    agent: str,
    tool: str,
    query: str,
    filters: dict[str, Any],
    input_refs: list[str],
    max_results: int,
    max_budget_usd: float | None,
) -> TypedSubAgentCall:
    payload = {"tool": tool, "query": query, "filters": filters, "limit": max_results}
    call_id = stable_call_id(parent_id, agent, "retrieval_branch", payload)
    return TypedSubAgentCall(
        call_id=call_id,
        parent_id=parent_id,
        agent=agent,
        capability="retrieval_branch",
        input_refs=input_refs,
        input_payload=payload,
        output_schema=RETRIEVAL_BRANCH_OUTPUT_SCHEMA,
        budget={"max_results": max_results, "max_budget_usd": max_budget_usd},
        validator="qu.retrieval_branch.output@2026-06-02.1",
    )


def complete_retrieval_branch_call(
    call: TypedSubAgentCall,
    chunks: list[dict[str, Any]],
    *,
    known_chunk_ids: set[str],
) -> TypedSubAgentResult:
    output_refs: list[str] = []
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        call_id = str(chunk.get("call_id", "")).strip()
        if not chunk_id or not call_id:
            return TypedSubAgentResult(
                call=call,
                status="error",
                validation_result="error",
                error="retrieval branch returned a chunk without chunk_id or call_id",
            )
        if chunk_id not in known_chunk_ids:
            return TypedSubAgentResult(
                call=call,
                status="error",
                validation_result="error",
                error=f"retrieval branch returned unknown chunk_id: {chunk_id}",
            )
        output_refs.append(f"chunk:{chunk_id}")
    return TypedSubAgentResult(
        call=call,
        status="ok",
        validation_result="ok",
        output_refs=output_refs,
        result_summary=f"{len(output_refs)} chunk refs",
    )


def join_subagent_results(
    *,
    parent_id: str,
    input_call_ids: list[str],
    output_refs: list[str],
    validator: str = "qu.subagent_join.output@2026-06-02.1",
) -> dict[str, Any]:
    return {
        "join_id": stable_call_id(parent_id, "qu", "subagent_join", {"input_call_ids": input_call_ids}),
        "parent_id": parent_id,
        "agent": "qu",
        "capability": "join",
        "input_call_ids": input_call_ids,
        "output_refs": output_refs,
        "validator": validator,
        "validation_result": "ok",
    }


def stable_call_id(parent_id: str, agent: str, capability: str, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {"parent_id": parent_id, "agent": agent, "capability": capability, "payload": payload},
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:12]
    return f"subcall-{digest}"
