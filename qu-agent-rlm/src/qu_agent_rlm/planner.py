from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Protocol

from .llm import JSONChatClient, LLMError
from .prompt_registry import QUERY_PLANNER_PROMPT, PromptRender


ALLOWED_STEP_TOOLS = {
    "search_chunks",
    "bm25_search_chunks",
    "embedding_search_chunks",
    "query_silver",
    "aggregate_silver",
    "fetch_chunks",
    "review_schema_gaps",
}

QUERY_STOPWORDS = {
    "and",
    "are",
    "break",
    "breakdown",
    "calls",
    "count",
    "down",
    "find",
    "for",
    "have",
    "how",
    "many",
    "mention",
    "mentions",
    "show",
    "the",
    "which",
    "with",
}


@dataclass
class QueryToolStep:
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    purpose: str = ""


@dataclass
class ColumnRequest:
    action: str
    field_name: str
    field_type: str
    description: str
    reason: str
    priority: str = "medium"
    suggested_allowed_values: list[str] = field(default_factory=list)
    example_queries: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)


@dataclass
class QueryPlan:
    operation: str
    filters: dict[str, Any] = field(default_factory=dict)
    group_by: str | None = None
    retrieve_evidence: bool = True
    ranking_query: str = ""
    planner: str = "heuristic"
    reasoning: str = ""
    steps: list[QueryToolStep] = field(default_factory=list)
    column_requests: list[ColumnRequest] = field(default_factory=list)


class QueryPlanner(Protocol):
    name: str

    def plan(self, query: str, catalog: dict[str, Any]) -> QueryPlan:
        raise NotImplementedError


def plan_query(query: str, catalog: dict[str, Any]) -> QueryPlan:
    return HeuristicQueryPlanner().plan(query, catalog)


class HeuristicQueryPlanner:
    name = "heuristic"

    def plan(self, query: str, catalog: dict[str, Any]) -> QueryPlan:
        return heuristic_plan_query(query, catalog)


class LLMQueryPlanner:
    def __init__(
        self,
        llm: JSONChatClient,
        *,
        fallback: QueryPlanner | None = None,
    ) -> None:
        self.llm = llm
        self.fallback = fallback
        self.name = llm.provider_name
        self.last_prompt: dict[str, Any] | None = None

    def plan(self, query: str, catalog: dict[str, Any]) -> QueryPlan:
        prompt = build_llm_planner_prompt_render(query, catalog)
        self.last_prompt = prompt.metadata()
        try:
            payload = self.llm.complete_json(system=prompt.system, user=prompt.user)
            return plan_from_llm_payload(payload, query, catalog, planner=self.name)
        except (LLMError, ValueError, TypeError) as exc:
            if self.fallback is None:
                raise
            plan = self.fallback.plan(query, catalog)
            plan.planner = f"{self.name}->fallback:{self.fallback.name}"
            plan.reasoning = f"LLM planner failed validation or request: {exc}"
            return plan


def heuristic_plan_query(query: str, catalog: dict[str, Any]) -> QueryPlan:
    text = query.lower()
    fields = {field["name"]: field for field in catalog.get("fields", [])}
    filters: dict[str, Any] = {}

    if contains_any(text, ["security", "sso", "audit", "redaction", "access control", "strict access"]):
        if "security_review_requested" in fields:
            filters["security_review_requested"] = True
    if contains_any(text, ["pricing", "price", "commercial"]):
        if "renewal_risk" in fields:
            filters["renewal_risk"] = "pricing_pushback"
    if contains_any(text, ["coupon"]):
        filters["complaint_theme"] = "coupon_confusion"
    elif contains_any(text, ["refund"]):
        filters["complaint_theme"] = "refund_policy"
    elif contains_any(text, ["shipping"]):
        filters["complaint_theme"] = "shipping_delay"
    elif contains_any(text, ["mobile login"]):
        filters["complaint_theme"] = "mobile_login"

    integration = detect_value(text, ["salesforce", "hubspot", "slack", "jira", "bigquery", "warehouse", "api"])
    if integration and "integration_request" in fields:
        filters["integration_request"] = integration

    if contains_any(text, ["missing data", "duplicate account", "missing opportunity", "campaign names"]):
        if "implementation_blocker" in fields:
            filters["implementation_blocker"] = "missing_data"

    if not filters:
        filters.update(detect_feedback_field_filters(text, fields))

    operation = "aggregate" if is_aggregation_query(text) else ("filter" if filters else "search")
    group_by = detect_group_by(text, fields) if operation == "aggregate" else None
    if operation == "aggregate" and group_by is None:
        group_by = "topic" if "topic" in fields else next(iter(fields), None)
    plan = QueryPlan(
        operation=operation,
        filters=filters,
        group_by=group_by,
        ranking_query=query,
        planner="heuristic",
        reasoning="Rule-based query planning.",
    )
    plan.steps = default_steps_for_plan(plan)
    return plan


def build_llm_planner_prompt(query: str, catalog: dict[str, Any]) -> tuple[str, str]:
    prompt = build_llm_planner_prompt_render(query, catalog)
    return prompt.system, prompt.user


def build_llm_planner_prompt_render(query: str, catalog: dict[str, Any]) -> PromptRender:
    fields = [
        {
            "name": field["name"],
            "type": field["type"],
            "description": field.get("description", ""),
            "allowed_values": field.get("allowed_values", []),
            "filterable": field.get("search", {}).get("filterable", True),
            "aggregatable": field.get("search", {}).get("aggregatable", False),
        }
        for field in catalog.get("fields", [])
    ]
    prompt = QUERY_PLANNER_PROMPT.render(
        {
            "query": query,
            "schema_version": catalog.get("schema_version"),
            "fields": fields,
        }
    )
    return prompt


def plan_from_llm_payload(
    payload: dict[str, Any],
    query: str,
    catalog: dict[str, Any],
    *,
    planner: str,
) -> QueryPlan:
    operation = str(payload.get("operation", "")).strip().lower()
    if operation not in {"filter", "aggregate", "search"}:
        raise ValueError(f"Invalid operation from LLM planner: {operation!r}")

    fields = {field["name"]: field for field in catalog.get("fields", [])}
    filters = validate_filters(payload.get("filters", {}), fields)
    group_by = payload.get("group_by")
    if group_by in ("", "null"):
        group_by = None
    if group_by is not None:
        group_by = str(group_by)
        if group_by not in fields:
            raise ValueError(f"Invalid group_by field from LLM planner: {group_by}")
        if not fields[group_by].get("search", {}).get("aggregatable", False):
            raise ValueError(f"LLM planner selected non-aggregatable group_by field: {group_by}")

    if operation == "aggregate" and group_by is None:
        raise ValueError("LLM planner selected aggregate without group_by")
    if operation == "search":
        filters = {}
        group_by = None

    retrieve_evidence = payload.get("retrieve_evidence", True)
    if not isinstance(retrieve_evidence, bool):
        raise ValueError("retrieve_evidence must be boolean")

    ranking_query = payload.get("ranking_query")
    if not isinstance(ranking_query, str) or not ranking_query.strip():
        ranking_query = query

    reasoning = payload.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = ""

    plan = QueryPlan(
        operation=operation,
        filters=filters,
        group_by=group_by,
        retrieve_evidence=retrieve_evidence,
        ranking_query=ranking_query,
        planner=planner,
        reasoning=reasoning,
        column_requests=validate_column_requests(
            payload.get("column_requests", []),
            query=query,
            existing_fields=set(fields),
        ),
    )
    plan.steps = validate_steps(payload.get("steps", []), plan, fields) or default_steps_for_plan(plan)
    return plan


def default_steps_for_plan(plan: QueryPlan) -> list[QueryToolStep]:
    ranking_query = plan.ranking_query or ""
    if plan.operation == "aggregate":
        steps = [
            QueryToolStep(
                tool="bm25_search_chunks",
                arguments={"query": ranking_query, "filters": {}, "limit": 8},
                purpose="Find broad candidate evidence before structured aggregation.",
            )
        ]
        steps.append(
            QueryToolStep(
                tool="query_silver",
                arguments={"filters": plan.filters, "limit": 50},
                purpose="Load representative silver records before aggregation.",
            )
        )
        steps.extend(
            [
                QueryToolStep(
                    tool="aggregate_silver",
                    arguments={"group_by": plan.group_by, "filters": plan.filters},
                    purpose="Compute the requested grouped count on silver fields.",
                ),
                QueryToolStep(
                    tool="fetch_chunks",
                    arguments={"limit": 8},
                    purpose="Fetch supporting evidence for representative records.",
                ),
                QueryToolStep(
                    tool="review_schema_gaps",
                    arguments={},
                    purpose="Record CU column requests if the query was only partially expressible.",
                ),
            ]
        )
        return steps
    if plan.operation == "filter":
        return [
            QueryToolStep(
                tool="query_silver",
                arguments={"filters": plan.filters, "limit": 50},
                purpose="Use validated silver filters for the primary answer set.",
            ),
            QueryToolStep(
                tool="bm25_search_chunks",
                arguments={"query": ranking_query, "filters": plan.filters, "limit": 8},
                purpose="Look for supporting evidence and possible exceptions.",
            ),
            QueryToolStep(
                tool="fetch_chunks",
                arguments={"limit": 8},
                purpose="Fetch evidence refs for matched records.",
            ),
            QueryToolStep(
                tool="review_schema_gaps",
                arguments={},
                purpose="Record CU column requests if the current schema undercovered the query.",
            ),
        ]
    return [
        QueryToolStep(
            tool="bm25_search_chunks",
            arguments={"query": ranking_query, "filters": {}, "limit": 10, "promote_records": True},
            purpose="Use retrieval because no reliable silver filter covers the query.",
        ),
        QueryToolStep(
            tool="fetch_chunks",
            arguments={"limit": 8},
            purpose="Fetch retrieved chunks as evidence.",
        ),
        QueryToolStep(
            tool="review_schema_gaps",
            arguments={},
            purpose="Turn repeated search-only needs into CU column requests.",
        ),
    ]


def validate_steps(raw_steps: Any, plan: QueryPlan, fields: dict[str, dict[str, Any]]) -> list[QueryToolStep]:
    if raw_steps in (None, ""):
        return []
    if not isinstance(raw_steps, list):
        raise ValueError("steps must be an array")
    steps: list[QueryToolStep] = []
    for raw in raw_steps:
        if not isinstance(raw, dict):
            raise ValueError("Each plan step must be an object")
        tool = str(raw.get("tool", "")).strip()
        if tool not in ALLOWED_STEP_TOOLS:
            raise ValueError(f"Invalid plan step tool: {tool}")
        arguments = raw.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ValueError(f"arguments for {tool} must be an object")
        arguments = validate_step_arguments(tool, arguments, fields)
        purpose = raw.get("purpose", "")
        steps.append(QueryToolStep(tool=tool, arguments=arguments, purpose=purpose if isinstance(purpose, str) else ""))
    if not steps:
        return []
    if plan.operation == "aggregate" and not any(step.tool == "aggregate_silver" for step in steps):
        return []
    if plan.operation == "aggregate" and not any(step.tool == "query_silver" for step in steps):
        aggregate_index = next(
            (index for index, step in enumerate(steps) if step.tool == "aggregate_silver"),
            0,
        )
        steps.insert(
            aggregate_index,
            QueryToolStep(
                tool="query_silver",
                arguments={"filters": plan.filters, "limit": 50},
                purpose="Load representative silver records before aggregation.",
            ),
        )
    if plan.retrieve_evidence and not any(step.tool == "fetch_chunks" for step in steps):
        steps.append(QueryToolStep(tool="fetch_chunks", arguments={"limit": 8}, purpose="Fetch evidence."))
    if not any(step.tool == "review_schema_gaps" for step in steps):
        steps.append(
            QueryToolStep(
                tool="review_schema_gaps",
                arguments={},
                purpose="Record CU column requests if the query exposes schema gaps.",
            )
        )
    return steps


SEARCH_STEP_TOOLS = {"search_chunks", "bm25_search_chunks", "embedding_search_chunks"}
MAX_SUBQUERY_FANOUT = 20


def validate_step_arguments(tool: str, arguments: dict[str, Any], fields: dict[str, dict[str, Any]]) -> dict[str, Any]:
    validated = dict(arguments)
    if "filters" in validated and validated["filters"] not in (None, {}, ""):
        validated["filters"] = validate_filters(validated["filters"], fields)
    if "queries" in validated:
        # A search step may declare a fan-out of subqueries (map-reduce): run each in parallel
        # and merge the results. Bounded to keep fan-out within the "thick prompt, small batch"
        # envelope. Non-search tools cannot fan out, so the key is dropped for them.
        if tool in SEARCH_STEP_TOOLS:
            raw = validated["queries"]
            if not isinstance(raw, list):
                raise ValueError(f"queries for {tool} must be an array")
            subqueries: list[str] = []
            for item in raw:
                text = str(item).strip()
                if text and text not in subqueries:
                    subqueries.append(text)
            if subqueries:
                validated["queries"] = subqueries[:MAX_SUBQUERY_FANOUT]
            else:
                validated.pop("queries")
        else:
            validated.pop("queries")
    if tool == "aggregate_silver":
        group_by = validated.get("group_by")
        if not isinstance(group_by, str) or group_by not in fields:
            raise ValueError(f"aggregate_silver needs a valid group_by field: {group_by}")
        if not fields[group_by].get("search", {}).get("aggregatable", False):
            raise ValueError(f"aggregate_silver selected non-aggregatable field: {group_by}")
    if "limit" in validated:
        try:
            validated["limit"] = max(1, min(int(validated["limit"]), 100))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid limit for {tool}: {validated['limit']!r}") from exc
    return validated


def validate_column_requests(
    raw_requests: Any,
    *,
    query: str,
    existing_fields: set[str] | None = None,
) -> list[ColumnRequest]:
    if raw_requests in (None, ""):
        return []
    if not isinstance(raw_requests, list):
        raise ValueError("column_requests must be an array")
    requests: list[ColumnRequest] = []
    for raw in raw_requests:
        if not isinstance(raw, dict):
            raise ValueError("Each column request must be an object")
        action = str(raw.get("action", "add_field")).strip()
        if action not in {"add_field", "add_allowed_values", "improve_extraction"}:
            action = "add_field"
        field_name = normalize_identifier(str(raw.get("field_name", "")))
        if not field_name:
            continue
        if existing_fields is not None:
            if action == "add_field" and field_name in existing_fields:
                continue
            if action in {"add_allowed_values", "improve_extraction"} and field_name not in existing_fields:
                action = "add_field"
        field_type = str(raw.get("field_type", raw.get("type", "list"))).strip()
        if field_type not in {"list", "enum", "boolean", "string"}:
            field_type = "list"
        priority = str(raw.get("priority", "medium")).strip()
        if priority not in {"low", "medium", "high"}:
            priority = "medium"
        requests.append(
            ColumnRequest(
                action=action,
                field_name=field_name,
                field_type=field_type,
                description=str(raw.get("description", "")).strip() or f"Column requested by query: {query}",
                reason=str(raw.get("reason", "")).strip() or "The query was not fully expressible with current silver fields.",
                priority=priority,
                suggested_allowed_values=normalize_string_list(raw.get("suggested_allowed_values", [])),
                example_queries=normalize_string_list(raw.get("example_queries", [query])) or [query],
                evidence_refs=normalize_string_list(raw.get("evidence_refs", [])),
            )
        )
    return requests


def validate_filters(raw_filters: Any, fields: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if raw_filters in (None, ""):
        return {}
    if not isinstance(raw_filters, dict):
        raise ValueError("filters must be an object")
    filters: dict[str, Any] = {}
    for field_name, value in raw_filters.items():
        if value in (None, "", [], "null"):
            continue
        name = str(field_name)
        if name not in fields:
            raise ValueError(f"Unknown filter field from LLM planner: {name}")
        field = fields[name]
        if not field.get("search", {}).get("filterable", True):
            raise ValueError(f"LLM planner selected non-filterable field: {name}")
        filters[name] = validate_filter_value(name, value, field)
    return filters


def validate_filter_value(field_name: str, value: Any, field: dict[str, Any]) -> Any:
    field_type = field.get("type")
    allowed_values = field.get("allowed_values") or []
    if field_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise ValueError(f"Boolean filter {field_name} must be true or false")
    if field_type == "list":
        if isinstance(value, list):
            return [validate_allowed_value(field_name, item, allowed_values) for item in value]
        return validate_allowed_value(field_name, value, allowed_values)
    if field_type == "enum":
        return validate_allowed_value(field_name, value, allowed_values)
    if field_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"String filter {field_name} must be a string")
        return value
    return value


def validate_allowed_value(field_name: str, value: Any, allowed_values: list[Any]) -> Any:
    if not allowed_values:
        return value
    if value not in allowed_values:
        raise ValueError(f"Invalid value for {field_name}: {value!r}")
    return value


def normalize_identifier(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    normalized = re.sub(r"_+", "_", normalized)
    if normalized and normalized[0].isdigit():
        normalized = f"field_{normalized}"
    return normalized


def normalize_string_list(raw: Any) -> list[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    values: list[str] = []
    seen: set[str] = set()
    for item in raw:
        value = str(item).strip()
        if value and value not in seen:
            values.append(value)
            seen.add(value)
    return values[:12]


def is_aggregation_query(text: str) -> bool:
    if contains_any(text, ["何件", "集計", "別", "break down", "group by", "grouped by"]):
        return True
    return bool(re.search(r"\b(count|aggregate|distribution|breakdown)\b|\bhow many\b", text))


def detect_group_by(text: str, fields: dict[str, Any]) -> str | None:
    aliases = {
        "product_area": ["product area", "product", "area", "プロダクト"],
        "topic": ["topic", "theme", "トピック"],
        "pain_point": ["pain", "pain point", "課題"],
        "urgency": ["urgency", "urgent", "緊急"],
        "implementation_blocker": ["blocker", "implementation blocker"],
        "renewal_risk": ["risk", "renewal"],
        "complaint_theme": ["complaint", "coupon", "refund", "shipping"],
    }
    for field_name, words in aliases.items():
        if field_name in fields and contains_any(text, words):
            return field_name
    for field_name in fields:
        if field_name.replace("_", " ") in text:
            return field_name
    return None


def detect_value(text: str, values: list[str]) -> str | None:
    for value in values:
        if value in text:
            return value
    return None


def detect_feedback_field_filters(text: str, fields: dict[str, dict[str, Any]]) -> dict[str, Any]:
    query_terms = set(content_terms(text))
    if not query_terms:
        return {}
    candidates: list[tuple[int, str, dict[str, Any], Any]] = []
    for field_name, field in fields.items():
        if not field.get("search", {}).get("filterable", True):
            continue
        if not is_feedback_field(field):
            continue
        field_terms = set(content_terms(" ".join([field_name, field.get("description", "")])))
        field_overlap = len(query_terms & field_terms)
        field_type = field.get("type")
        if field_type == "boolean" and field_overlap >= 1:
            candidates.append((field_overlap + 2, field_name, field, True))
            continue
        allowed = field.get("allowed_values") or []
        value = best_allowed_value(text, query_terms, allowed)
        if value is not None:
            value_terms = set(content_terms(str(value)))
            score = field_overlap + len(query_terms & value_terms) + 1
            candidates.append((score, field_name, field, value))
    if not candidates:
        return {}
    _, field_name, _, value = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
    return {field_name: value}


def is_feedback_field(field: dict[str, Any]) -> bool:
    usage = field.get("usage", {})
    good_for = usage.get("good_for", []) if isinstance(usage, dict) else []
    if any(str(value).lower() == "qu feedback refinement" for value in good_for):
        return True
    quality = field.get("quality", {})
    if isinstance(quality, dict) and "feedback" in str(quality.get("reasons", "")).lower():
        return True
    return "downstream queries expressible" in str(field.get("description", "")).lower()


def best_allowed_value(text: str, query_terms: set[str], allowed_values: list[Any]) -> str | None:
    scored: list[tuple[int, int, str]] = []
    for raw_value in allowed_values:
        value = str(raw_value)
        if value == "not_mentioned":
            continue
        value_terms = set(content_terms(value))
        if not value_terms:
            continue
        phrase = value.replace("_", " ").lower()
        phrase_bonus = 2 if phrase in text else 0
        overlap = len(query_terms & value_terms)
        if overlap:
            scored.append((overlap + phrase_bonus, len(value_terms), value))
    if not scored:
        return None
    _, _, value = sorted(scored, key=lambda item: (-item[0], -item[1], item[2]))[0]
    return value


def content_terms(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
        if len(token) > 2 and token not in QUERY_STOPWORDS
    ]


def contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)
