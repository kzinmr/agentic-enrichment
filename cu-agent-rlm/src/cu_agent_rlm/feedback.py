from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

from .models import CallRecord, Chunk, FieldExtraction, FieldSpec
from .schema import SchemaInducer


VALID_ACTIONS = {"add_field", "add_allowed_values", "improve_extraction"}
VALID_FIELD_TYPES = {"list", "enum", "boolean", "string"}
PRIORITY_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass
class FeedbackSummary:
    source_path: str | None = None
    request_count: int = 0
    search_failure_count: int = 0
    answerability_failure_count: int = 0
    evidence_failure_count: int = 0
    requests: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_requests(self) -> bool:
        return bool(self.requests)

    def manifest_summary(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "request_count": self.request_count,
            "search_failure_count": self.search_failure_count,
            "answerability_failure_count": self.answerability_failure_count,
            "evidence_failure_count": self.evidence_failure_count,
            "requested_fields": [request["field_name"] for request in self.requests],
        }


class FeedbackAwareSchemaInducer:
    def __init__(
        self,
        base: SchemaInducer,
        feedback: FeedbackSummary,
        *,
        max_feedback_fields: int = 12,
    ) -> None:
        self.base = base
        self.feedback = feedback
        self.max_feedback_fields = max_feedback_fields
        self.name = f"{base.name}+feedback" if feedback.has_requests else base.name

    def induce_schema(self, calls: list[CallRecord], chunks: list[Chunk]) -> list[FieldSpec]:
        specs = self.base.induce_schema(calls, chunks)
        if not self.feedback.has_requests:
            return specs
        return apply_feedback_to_specs(specs, self.feedback, max_feedback_fields=self.max_feedback_fields)


def load_feedback_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payloads: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid feedback JSONL at {path}:{line_no}") from exc
            if isinstance(payload, dict):
                payloads.append(payload)
    return payloads


def summarize_feedback(payloads: list[dict[str, Any]], *, source_path: Path | None = None) -> FeedbackSummary:
    merged: dict[str, dict[str, Any]] = {}
    search_failure_count = 0
    answerability_failure_count = 0
    evidence_failure_count = 0
    for payload in payloads:
        query = str(payload.get("query", "")).strip()
        failures = normalize_search_failures(payload.get("search_diagnostics"))
        search_failure_count += len(failures)
        judgement = normalize_judgement(payload.get("judgement"))
        if judgement:
            answerability_failure_count += 0 if judgement["answerable"] else 1
            evidence_failure_count += 0 if judgement["evidence_sufficient"] else 1
        requests = payload.get("column_requests") or []
        if not isinstance(requests, list):
            requests = []
        if judgement:
            requests = extend_with_judgement_requests(requests, judgement["missing_field_requests"])
        if not requests and failures:
            requests = [request_from_failures(query, failures)]
        if not requests and judgement and judgement["needs_cu_feedback"]:
            requests = [request_from_judgement(query, judgement)]
        for raw_request in requests:
            request = normalize_request(raw_request, query=query, failures=failures)
            if request is None:
                continue
            key = request["field_name"]
            if key not in merged:
                merged[key] = request
            else:
                merged[key] = merge_requests(merged[key], request)
    requests = sorted(
        merged.values(),
        key=lambda item: (-PRIORITY_RANK.get(item["priority"], 1), item["field_name"]),
    )
    return FeedbackSummary(
        source_path=str(source_path) if source_path else None,
        request_count=sum(request["count"] for request in requests),
        search_failure_count=search_failure_count,
        answerability_failure_count=answerability_failure_count,
        evidence_failure_count=evidence_failure_count,
        requests=requests,
    )


def empty_feedback_summary() -> FeedbackSummary:
    return FeedbackSummary()


def normalize_search_failures(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    failures = raw.get("failures", [])
    if not isinstance(failures, list):
        return []
    normalized = []
    for failure in failures:
        if not isinstance(failure, dict):
            continue
        reason = str(failure.get("failure_reason", "")).strip()
        if not reason:
            continue
        normalized.append(
            {
                "failure_reason": reason,
                "query_terms": normalize_string_list(failure.get("query_terms", [])),
                "missing_terms": normalize_string_list(failure.get("missing_terms", [])),
                "tool": str(failure.get("tool", "")).strip(),
            }
        )
    return normalized


def normalize_judgement(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    raw_requests = raw.get("missing_field_requests", [])
    return {
        "answerable": bool(raw.get("answerable")),
        "evidence_sufficient": bool(raw.get("evidence_sufficient")),
        "success": bool(raw.get("success")),
        "needs_cu_feedback": bool(raw.get("needs_cu_feedback")) or not bool(raw.get("success")),
        "confidence": str(raw.get("confidence", "")).strip(),
        "failure_modes": normalize_string_list(raw.get("failure_modes", [])),
        "missing_field_requests": raw_requests if isinstance(raw_requests, list) else [],
        "rationale": str(raw.get("rationale", "")).strip(),
    }


def extend_with_judgement_requests(
    requests: list[Any],
    judgement_requests: list[Any],
) -> list[Any]:
    existing = {
        normalize_identifier(str(request.get("field_name", "")))
        for request in requests
        if isinstance(request, dict)
    }
    merged = list(requests)
    for request in judgement_requests:
        if not isinstance(request, dict):
            continue
        field_name = normalize_identifier(str(request.get("field_name", "")))
        if not field_name or field_name in existing:
            continue
        merged.append(request)
        existing.add(field_name)
    return merged


def request_from_failures(query: str, failures: list[dict[str, Any]]) -> dict[str, Any]:
    terms = []
    for failure in failures:
        terms.extend(failure.get("missing_terms", []))
        terms.extend(failure.get("query_terms", []))
    values = normalize_allowed_values(terms, limit=8) or suggested_values_from_query(query)
    field_name = "_".join(values[:3]) if values else "query_signal"
    return {
        "action": "add_field",
        "field_name": field_name,
        "field_type": "list",
        "description": f"Reusable field requested because QU search could not cover: {query}",
        "reason": "; ".join(failure["failure_reason"] for failure in failures[:3]),
        "priority": "medium",
        "suggested_allowed_values": values,
        "example_queries": [query] if query else [],
        "evidence_refs": [],
    }


def request_from_judgement(query: str, judgement: dict[str, Any]) -> dict[str, Any]:
    values = suggested_values_from_query(query)
    field_name = "_".join(values[:3]) if values else "query_signal"
    modes = ", ".join(judgement.get("failure_modes", []))
    reason = judgement.get("rationale") or f"Answer judge requested CU feedback for modes: {modes}"
    return {
        "action": "add_field",
        "field_name": field_name,
        "field_type": "list",
        "description": f"Reusable field requested by answerability judge for: {query}",
        "reason": reason,
        "priority": "high" if not judgement.get("answerable") else "medium",
        "suggested_allowed_values": values,
        "example_queries": [query] if query else [],
        "evidence_refs": [],
    }


def normalize_request(raw: Any, *, query: str, failures: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    field_name = normalize_identifier(str(raw.get("field_name", "")))
    if not field_name:
        return None
    action = str(raw.get("action", "add_field")).strip()
    if action not in VALID_ACTIONS:
        action = "add_field"
    field_type = str(raw.get("field_type", raw.get("type", "list"))).strip()
    if field_type not in VALID_FIELD_TYPES:
        field_type = "list"
    priority = str(raw.get("priority", "medium")).strip()
    if priority not in PRIORITY_RANK:
        priority = "medium"
    reason = str(raw.get("reason", "")).strip()
    failure_reasons = [failure["failure_reason"] for failure in failures if failure.get("failure_reason")]
    return {
        "action": action,
        "field_name": field_name,
        "field_type": field_type,
        "description": str(raw.get("description", "")).strip()
        or f"Field requested by downstream QU feedback for: {query}",
        "reason": reason,
        "priority": priority,
        "suggested_allowed_values": normalize_allowed_values(raw.get("suggested_allowed_values", []), limit=16),
        "example_queries": normalize_string_list(raw.get("example_queries", [query])) or ([query] if query else []),
        "evidence_refs": normalize_string_list(raw.get("evidence_refs", [])),
        "failure_reasons": failure_reasons,
        "source_queries": [query] if query else [],
        "count": 1,
    }


def merge_requests(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    priority = left["priority"]
    if PRIORITY_RANK.get(right["priority"], 1) > PRIORITY_RANK.get(priority, 1):
        priority = right["priority"]
    return {
        **left,
        "action": merge_action(left["action"], right["action"]),
        "field_type": merge_field_type(left["field_type"], right["field_type"]),
        "description": left["description"] if len(left["description"]) >= len(right["description"]) else right["description"],
        "reason": join_unique_text([left.get("reason", ""), right.get("reason", "")]),
        "priority": priority,
        "suggested_allowed_values": dedupe([*left["suggested_allowed_values"], *right["suggested_allowed_values"]])[:16],
        "example_queries": dedupe([*left["example_queries"], *right["example_queries"]])[:12],
        "evidence_refs": dedupe([*left["evidence_refs"], *right["evidence_refs"]])[:24],
        "failure_reasons": dedupe([*left["failure_reasons"], *right["failure_reasons"]])[:12],
        "source_queries": dedupe([*left["source_queries"], *right["source_queries"]])[:12],
        "count": int(left.get("count", 1)) + int(right.get("count", 1)),
    }


def merge_action(left: str, right: str) -> str:
    if "add_field" in {left, right}:
        return "add_field"
    if "improve_extraction" in {left, right}:
        return "improve_extraction"
    return "add_allowed_values"


def merge_field_type(left: str, right: str) -> str:
    if left == right:
        return left
    if "string" in {left, right}:
        return "string"
    if "list" in {left, right}:
        return "list"
    if "enum" in {left, right}:
        return "enum"
    return left


def apply_feedback_to_specs(
    specs: list[FieldSpec],
    feedback: FeedbackSummary,
    *,
    max_feedback_fields: int,
) -> list[FieldSpec]:
    by_name = {spec.name: spec for spec in specs}
    ordered = list(specs)
    added = 0
    for request in feedback.requests:
        field_name = request["field_name"]
        if field_name in by_name:
            updated = merge_spec_with_request(by_name[field_name], request)
            by_name[field_name] = updated
            ordered = [updated if spec.name == field_name else spec for spec in ordered]
            continue
        if added >= max_feedback_fields:
            continue
        spec = spec_from_request(request)
        by_name[field_name] = spec
        ordered.append(spec)
        added += 1
    return ordered


def merge_spec_with_request(spec: FieldSpec, request: dict[str, Any]) -> FieldSpec:
    return FieldSpec(
        name=spec.name,
        type=spec.type,
        description=spec.description,
        allowed_values=dedupe([*spec.allowed_values, *request["suggested_allowed_values"]])[:24],
        downstream_use_cases=dedupe([*spec.downstream_use_cases, "QU feedback refinement"]),
        filterable=spec.filterable,
        facetable=spec.facetable,
        aggregatable=spec.aggregatable,
    )


def spec_from_request(request: dict[str, Any]) -> FieldSpec:
    field_type = request["field_type"]
    allowed_values = list(request["suggested_allowed_values"])
    if field_type == "enum" and not allowed_values:
        field_type = "list"
    if field_type == "enum" and "not_mentioned" not in allowed_values:
        allowed_values = ["not_mentioned", *allowed_values]
    if field_type not in {"list", "enum"}:
        allowed_values = []
    return FieldSpec(
        name=request["field_name"],
        type=field_type,
        description=request["description"],
        allowed_values=allowed_values,
        downstream_use_cases=dedupe(
            [
                "filtering",
                "aggregation" if field_type != "string" else "evidence-backed analysis",
                "QU feedback refinement",
                *request["example_queries"][:2],
            ]
        ),
        filterable=True,
        facetable=field_type != "string",
        aggregatable=field_type != "string",
    )


def build_feedback_report(
    feedback: FeedbackSummary,
    *,
    schema_specs: list[FieldSpec],
    promoted_specs: list[FieldSpec],
    extractions: list[FieldExtraction],
) -> dict[str, Any]:
    schema_names = {spec.name for spec in schema_specs}
    promoted_names = {spec.name for spec in promoted_specs}
    validation_errors: Counter[str] = Counter()
    for extraction in extractions:
        validation_errors.update(extraction.validation_errors)
    return {
        "feedback": feedback.manifest_summary(),
        "requested_fields": [
            {
                **request,
                "accepted_into_schema": request["field_name"] in schema_names,
                "promoted_to_silver": request["field_name"] in promoted_names,
            }
            for request in feedback.requests
        ],
        "validation_error_counts": dict(sorted(validation_errors.items())),
    }


def normalize_allowed_values(raw_values: Any, *, limit: int) -> list[str]:
    return dedupe(normalize_identifier(value) for value in normalize_string_list(raw_values))[:limit]


def normalize_string_list(raw: Any) -> list[str]:
    if raw in (None, ""):
        return []
    values = raw if isinstance(raw, list) else [raw]
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            result.append(text)
    return dedupe(result)


def normalize_identifier(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    normalized = re.sub(r"_+", "_", normalized)
    if normalized and normalized[0].isdigit():
        normalized = f"field_{normalized}"
    return normalized


def suggested_values_from_query(query: str) -> list[str]:
    return [
        value
        for value in normalize_allowed_values(query.replace("-", " ").split(), limit=8)
        if value not in {"which", "calls", "mention", "mentions", "show", "find", "with", "the", "and"}
    ]


def join_unique_text(values: list[str]) -> str:
    return "; ".join(value for value in dedupe([value.strip() for value in values]) if value)


def dedupe(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result
