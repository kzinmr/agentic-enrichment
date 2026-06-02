from __future__ import annotations

from dataclasses import asdict
from typing import Any, Protocol

from .llm import JSONChatClient, LLMError
from .planner import ColumnRequest, QueryPlan
from .prompt_registry import ANSWER_JUDGE_PROMPT


CONFIDENCE_VALUES = {"low", "medium", "high"}


class AnswerJudge(Protocol):
    name: str

    def judge(
        self,
        *,
        query: str,
        plan: QueryPlan,
        answer: str,
        records: list[dict[str, Any]],
        aggregation: dict[str, Any],
        evidence: list[dict[str, Any]],
        search_calls: list[dict[str, Any]],
        search_failures: list[dict[str, Any]],
        rerank: dict[str, Any],
        column_requests: list[ColumnRequest],
        catalog: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError


class HeuristicAnswerJudge:
    name = "heuristic"

    def judge(
        self,
        *,
        query: str,
        plan: QueryPlan,
        answer: str,
        records: list[dict[str, Any]],
        aggregation: dict[str, Any],
        evidence: list[dict[str, Any]],
        search_calls: list[dict[str, Any]],
        search_failures: list[dict[str, Any]],
        rerank: dict[str, Any],
        column_requests: list[ColumnRequest],
        catalog: dict[str, Any],
    ) -> dict[str, Any]:
        del query, answer, catalog
        operation = plan.operation
        record_count = len(records)
        evidence_count = len(evidence)
        aggregation_count = len(aggregation)
        search_failure_count = len(search_failures)
        column_request_count = len(column_requests)
        answerable = bool(aggregation) if operation == "aggregate" else bool(records)
        evidence_required = bool(plan.retrieve_evidence) and answerable
        evidence_sufficient = not evidence_required or evidence_count > 0
        failure_modes: list[str] = []
        if operation == "aggregate" and not aggregation:
            failure_modes.append("empty_aggregation")
        if operation != "aggregate" and not records:
            failure_modes.append("no_records")
        if evidence_required and not evidence:
            failure_modes.append("missing_evidence")
        if search_failures:
            failure_modes.append("retrieval_failure")
        if column_requests:
            failure_modes.append("schema_gap")
        if operation == "search" and answerable and column_requests:
            failure_modes.append("search_only_schema_gap")

        success = answerable and evidence_sufficient
        needs_cu_feedback = bool(column_requests) or not success or bool(search_failures)
        confidence = confidence_for_result(
            answerable=answerable,
            evidence_sufficient=evidence_sufficient,
            search_failure_count=search_failure_count,
            column_request_count=column_request_count,
            operation=operation,
        )
        return {
            "judge": self.name,
            "answerable": answerable,
            "evidence_sufficient": evidence_sufficient,
            "success": success,
            "needs_cu_feedback": needs_cu_feedback,
            "confidence": confidence,
            "failure_modes": failure_modes,
            "missing_field_requests": [asdict(request) for request in column_requests],
            "rationale": rationale_for_result(
                answerable=answerable,
                evidence_sufficient=evidence_sufficient,
                needs_cu_feedback=needs_cu_feedback,
                failure_modes=failure_modes,
            ),
            "metrics": {
                "record_count": record_count,
                "evidence_count": evidence_count,
                "aggregation_group_count": aggregation_count,
                "search_call_count": len(search_calls),
                "search_failure_count": search_failure_count,
                "column_request_count": column_request_count,
                "embedding_call_count": sum(1 for call in search_calls if "embedding" in str(call.get("tool", ""))),
                "reranked_chunk_count": len(rerank.get("ranked_chunks", [])) if isinstance(rerank, dict) else 0,
            },
        }


class NoopAnswerJudge:
    name = "none"

    def judge(
        self,
        *,
        query: str,
        plan: QueryPlan,
        answer: str,
        records: list[dict[str, Any]],
        aggregation: dict[str, Any],
        evidence: list[dict[str, Any]],
        search_calls: list[dict[str, Any]],
        search_failures: list[dict[str, Any]],
        rerank: dict[str, Any],
        column_requests: list[ColumnRequest],
        catalog: dict[str, Any],
    ) -> dict[str, Any]:
        del query, plan, answer, catalog
        answerable = bool(records) or bool(aggregation)
        evidence_sufficient = bool(evidence) or not answerable
        return {
            "judge": self.name,
            "answerable": answerable,
            "evidence_sufficient": evidence_sufficient,
            "success": answerable and evidence_sufficient,
            "needs_cu_feedback": bool(column_requests or search_failures),
            "confidence": "low",
            "failure_modes": [],
            "missing_field_requests": [asdict(request) for request in column_requests],
            "rationale": "Answer judging was disabled.",
            "metrics": {
                "record_count": len(records),
                "evidence_count": len(evidence),
                "aggregation_group_count": len(aggregation),
                "search_call_count": len(search_calls),
                "search_failure_count": len(search_failures),
                "column_request_count": len(column_requests),
                "reranked_chunk_count": len(rerank.get("ranked_chunks", [])) if isinstance(rerank, dict) else 0,
            },
        }


class LLMAnswerJudge:
    def __init__(self, llm: JSONChatClient, *, fallback: AnswerJudge | None = None) -> None:
        self.llm = llm
        self.fallback = fallback
        self.name = llm.provider_name
        self.last_prompt: dict[str, Any] | None = None

    def judge(
        self,
        *,
        query: str,
        plan: QueryPlan,
        answer: str,
        records: list[dict[str, Any]],
        aggregation: dict[str, Any],
        evidence: list[dict[str, Any]],
        search_calls: list[dict[str, Any]],
        search_failures: list[dict[str, Any]],
        rerank: dict[str, Any],
        column_requests: list[ColumnRequest],
        catalog: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = ANSWER_JUDGE_PROMPT.render(
            {
                "query": query,
                "answer": answer,
                "plan": asdict(plan),
                "records": compact_records_for_judge(records),
                "aggregation": aggregation,
                "evidence": evidence[:8],
                "search_diagnostics": {
                    "calls": compact_search_calls(search_calls),
                    "failures": search_failures[:6],
                },
                "rerank": rerank,
                "column_requests": [asdict(request) for request in column_requests],
                "schema_fields": compact_fields_for_judge(catalog),
            }
        )
        self.last_prompt = prompt.metadata()
        try:
            return normalize_judge_payload(
                self.llm.complete_json(system=prompt.system, user=prompt.user),
                judge=self.name,
                fallback_requests=[asdict(request) for request in column_requests],
                prompt=self.last_prompt,
            )
        except (LLMError, ValueError, TypeError) as exc:
            if self.fallback is None:
                raise
            fallback_result = self.fallback.judge(
                query=query,
                plan=plan,
                answer=answer,
                records=records,
                aggregation=aggregation,
                evidence=evidence,
                search_calls=search_calls,
                search_failures=search_failures,
                rerank=rerank,
                column_requests=column_requests,
                catalog=catalog,
            )
            fallback_result["judge"] = f"{self.name}->fallback:{self.fallback.name}"
            fallback_result["judge_error"] = str(exc)
            fallback_result["prompt"] = self.last_prompt
            return fallback_result


def normalize_judge_payload(
    payload: dict[str, Any],
    *,
    judge: str,
    fallback_requests: list[dict[str, Any]],
    prompt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    answerable = bool(payload.get("answerable"))
    evidence_sufficient = bool(payload.get("evidence_sufficient"))
    success = bool(payload.get("success", answerable and evidence_sufficient))
    needs_cu_feedback = bool(payload.get("needs_cu_feedback", not success))
    confidence = str(payload.get("confidence", "low")).strip().lower()
    if confidence not in CONFIDENCE_VALUES:
        confidence = "low"
    raw_modes = payload.get("failure_modes", [])
    failure_modes = [str(mode).strip() for mode in raw_modes] if isinstance(raw_modes, list) else []
    raw_requests = payload.get("missing_field_requests", [])
    missing_field_requests = raw_requests if isinstance(raw_requests, list) else []
    if not missing_field_requests and needs_cu_feedback:
        missing_field_requests = fallback_requests
    return {
        "judge": judge,
        "answerable": answerable,
        "evidence_sufficient": evidence_sufficient,
        "success": success,
        "needs_cu_feedback": needs_cu_feedback,
        "confidence": confidence,
        "failure_modes": failure_modes,
        "missing_field_requests": missing_field_requests,
        "rationale": str(payload.get("rationale", "")).strip(),
        "metrics": payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {},
        "prompt": prompt or {},
    }


def confidence_for_result(
    *,
    answerable: bool,
    evidence_sufficient: bool,
    search_failure_count: int,
    column_request_count: int,
    operation: str,
) -> str:
    if not answerable or not evidence_sufficient:
        return "low"
    if search_failure_count or column_request_count or operation == "search":
        return "medium"
    return "high"


def rationale_for_result(
    *,
    answerable: bool,
    evidence_sufficient: bool,
    needs_cu_feedback: bool,
    failure_modes: list[str],
) -> str:
    if answerable and evidence_sufficient and not needs_cu_feedback:
        return "The answer is supported by available silver records and evidence."
    if answerable and evidence_sufficient:
        return "The answer is usable, but the trace exposes reusable CU improvements."
    if failure_modes:
        return f"The answer is not fully supported because: {', '.join(failure_modes)}."
    return "The answer is not fully supported by the current schema and evidence."


def compact_records_for_judge(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "call_id": record.get("call_id"),
            "account_name": record.get("account_name"),
            "fields": record.get("fields", {}),
        }
        for record in records[:12]
    ]


def compact_search_calls(search_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "tool": call.get("tool"),
            "query": call.get("query"),
            "result_count": call.get("result_count"),
            "matched_terms": call.get("matched_terms", []),
            "missing_terms": call.get("missing_terms", []),
            "failure_reason": call.get("failure_reason", ""),
        }
        for call in search_calls[-8:]
    ]


def compact_fields_for_judge(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": field.get("name"),
            "type": field.get("type"),
            "description": field.get("description", ""),
            "allowed_values": field.get("allowed_values", [])[:16],
        }
        for field in catalog.get("fields", [])
    ]
