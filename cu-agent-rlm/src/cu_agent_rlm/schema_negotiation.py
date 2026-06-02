from __future__ import annotations

from typing import Any

from .feedback import FeedbackSummary
from .models import FieldSpec


def build_field_candidates(
    specs: list[FieldSpec],
    quality_report: dict[str, Any],
    feedback: FeedbackSummary,
) -> list[dict[str, Any]]:
    quality_by_name = {field["field_name"]: field for field in quality_report.get("fields", [])}
    feedback_by_name = {request["field_name"]: request for request in feedback.requests}
    candidates: list[dict[str, Any]] = []
    for spec in specs:
        quality = quality_by_name.get(spec.name, {})
        request = feedback_by_name.get(spec.name, {})
        support_count = int(quality.get("support_call_count", 0) or 0)
        evidence_coverage = float(quality.get("evidence_coverage", 0.0) or 0.0)
        fill_rate = float(quality.get("fill_rate", 0.0) or 0.0)
        uncertainty = candidate_uncertainty(
            support_count=support_count,
            evidence_coverage=evidence_coverage,
            requested=bool(request),
        )
        candidates.append(
            {
                "candidate_id": f"field:{spec.name}",
                "field_name": spec.name,
                "field_type": spec.type,
                "description": spec.description,
                "allowed_values": spec.allowed_values,
                "source": "qu_feedback" if request else "schema_induction",
                "intended_queries": intended_queries(spec, request),
                "example_filters": example_filters(spec),
                "example_aggregations": example_aggregations(spec),
                "evidence_requirements": {
                    "requires_chunk_refs": True,
                    "minimum_evidence_coverage": quality_report.get("promotion_policy", {}).get("minimum_evidence_coverage", 0.75),
                    "current_evidence_coverage": round(evidence_coverage, 4),
                    "support_call_count": support_count,
                    "fill_rate": round(fill_rate, 4),
                },
                "uncertainty": uncertainty,
                "merge_with_existing": bool(request and request.get("action") in {"add_allowed_values", "improve_extraction"}),
                "rationale": candidate_rationale(spec, quality, request),
            }
        )
    return candidates


def build_schema_negotiation_report(
    field_candidates: list[dict[str, Any]],
    quality_report: dict[str, Any],
    feedback: FeedbackSummary,
) -> dict[str, Any]:
    quality_by_name = {field["field_name"]: field for field in quality_report.get("fields", [])}
    decisions = [
        candidate_decision(candidate, quality_by_name.get(candidate["field_name"], {}), feedback)
        for candidate in field_candidates
    ]
    return {
        "contract": "cu.qu_schema_negotiation@2026-06-02.1",
        "field_candidates_artifact": "field_candidates.json",
        "decision_values": ["promote", "defer", "retrieval_only", "merge_with_existing", "needs_human_review"],
        "policy": {
            "promotion_requires": [
                "candidate emitted with intended query patterns",
                "field quality report passes support and evidence coverage thresholds",
                "QU candidate evaluation may override future promotions with defer/retrieval_only/human_review",
            ],
            "synthetic_queries": "bootstrap probes are evidence for exploration, not final governance approval",
        },
        "decisions": decisions,
    }


def candidate_decision(
    candidate: dict[str, Any],
    quality: dict[str, Any],
    feedback: FeedbackSummary,
) -> dict[str, Any]:
    del feedback
    reasons = list(quality.get("reasons", []))
    status = str(quality.get("recommended_status", "hold"))
    support_count = int(quality.get("support_call_count", 0) or 0)
    evidence_coverage = float(quality.get("evidence_coverage", 0.0) or 0.0)
    if candidate.get("merge_with_existing"):
        decision = "merge_with_existing"
        rationale = "QU feedback targets an existing field contract; merge allowed values or extraction guidance."
    elif candidate.get("uncertainty") == "high" and candidate.get("source") == "qu_feedback":
        decision = "needs_human_review"
        rationale = "High-priority QU feedback lacks enough extraction evidence for automatic promotion."
    elif status == "promote":
        decision = "promote"
        rationale = "Candidate passed CU support and evidence coverage thresholds."
    elif support_count == 0:
        decision = "retrieval_only"
        rationale = "No supported silver values were extracted; keep this concept in retrieval until more evidence accumulates."
    else:
        decision = "defer"
        rationale = "Candidate has partial extraction support but does not yet meet promotion thresholds."
    return {
        "candidate_id": candidate["candidate_id"],
        "field_name": candidate["field_name"],
        "decision": decision,
        "rationale": rationale,
        "quality": {
            "support_call_count": support_count,
            "evidence_coverage": round(evidence_coverage, 4),
            "reasons": reasons,
        },
        "revisit_when": revisit_condition(decision),
    }


def intended_queries(spec: FieldSpec, request: dict[str, Any]) -> list[str]:
    queries = [str(query) for query in request.get("example_queries", []) if str(query).strip()]
    if queries:
        return queries[:6]
    label = spec.name.replace("_", " ")
    examples = [f"Find calls where {label} is mentioned."]
    if spec.aggregatable:
        examples.append(f"Count calls by {label}.")
    if spec.allowed_values:
        examples.append(f"Find calls where {label} includes {spec.allowed_values[0].replace('_', ' ')}.")
    return examples[:6]


def example_filters(spec: FieldSpec) -> list[dict[str, Any]]:
    if not spec.filterable:
        return []
    if spec.type == "boolean":
        return [{"filters": {spec.name: True}}]
    if spec.allowed_values:
        first_value = next((value for value in spec.allowed_values if value != "not_mentioned"), spec.allowed_values[0])
        return [{"filters": {spec.name: first_value}}]
    return []


def example_aggregations(spec: FieldSpec) -> list[dict[str, Any]]:
    if not spec.aggregatable:
        return []
    return [
        {"group_by": spec.name},
        {"expression": f'top_k(records, "{spec.name}", k=5)'},
    ]


def candidate_uncertainty(*, support_count: int, evidence_coverage: float, requested: bool) -> str:
    if support_count == 0:
        return "high"
    if evidence_coverage < 0.75 or (requested and support_count < 2):
        return "medium"
    return "low"


def candidate_rationale(spec: FieldSpec, quality: dict[str, Any], request: dict[str, Any]) -> str:
    if request:
        return str(request.get("reason") or f"QU requested reusable field {spec.name}.")
    reasons = quality.get("reasons", [])
    if reasons:
        return f"Candidate emerged from schema induction; quality issues: {', '.join(reasons)}."
    return "Candidate emerged from schema induction and is available for QU simulation."


def revisit_condition(decision: str) -> str:
    if decision == "promote":
        return "monitor QU failures and extraction validation errors"
    if decision == "retrieval_only":
        return "reconsider after repeated QU failures or new evidence-bearing calls"
    if decision == "needs_human_review":
        return "review before production promotion"
    if decision == "merge_with_existing":
        return "rerun QU tasks against merged allowed values or extraction guidance"
    return "reconsider when support count or evidence coverage improves"
