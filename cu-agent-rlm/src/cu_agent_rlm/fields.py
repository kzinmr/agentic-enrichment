from __future__ import annotations

import re
from typing import Any

from .models import CallRecord, Chunk, FieldExtraction, FieldSpec


SCHEMA_VERSION = "silver.rlm.v1"


FIELD_SPECS = [
    FieldSpec(
        name="topic",
        type="list",
        description="Business topic categories useful for routing and broad analysis.",
        allowed_values=[
            "call_search",
            "crm_hygiene",
            "security_governance",
            "pipeline_review",
            "support_analytics",
            "campaign_analytics",
            "customer_journey",
        ],
        downstream_use_cases=["semantic routing", "topic filters", "trend aggregation"],
    ),
    FieldSpec(
        name="pain_point",
        type="list",
        description="Repeated operational pain, blocker, or unmet need mentioned in the call.",
        allowed_values=[
            "manual_work",
            "crm_data_quality",
            "privacy_or_access_control",
            "pricing_objection",
            "integration_gap",
            "complaint_theme",
            "campaign_normalization",
            "evidence_requirement",
        ],
        downstream_use_cases=["pain trend analysis", "support prioritization", "answerability checks"],
    ),
    FieldSpec(
        name="product_area",
        type="list",
        description="Product, workflow, data source, or integration area discussed.",
        allowed_values=[
            "search",
            "crm",
            "security",
            "integrations",
            "analytics_dashboard",
            "mobile_app",
            "campaigns",
            "fulfillment",
            "follow_up",
        ],
        downstream_use_cases=["product area filters", "roadmap analysis", "evidence retrieval"],
    ),
    FieldSpec(
        name="urgency",
        type="enum",
        description="Time sensitivity or escalation level for prioritization.",
        allowed_values=["not_mentioned", "low", "medium", "high"],
        downstream_use_cases=["review prioritization", "escalation filters"],
    ),
    FieldSpec(
        name="implementation_blocker",
        type="enum",
        description="Primary blocker that could prevent rollout or adoption.",
        allowed_values=[
            "not_mentioned",
            "security_review",
            "missing_data",
            "integration",
            "stakeholder_alignment",
            "regulatory_escalation",
        ],
        downstream_use_cases=["blocker analysis", "enablement backlog creation"],
    ),
    FieldSpec(
        name="renewal_risk",
        type="enum",
        description="Commercial or adoption risk signal that needs account review.",
        allowed_values=[
            "not_mentioned",
            "pricing_pushback",
            "low_adoption",
            "unresolved_blocker",
            "competitor_evaluation",
        ],
        downstream_use_cases=["at-risk account search", "risk reason grouping"],
    ),
    FieldSpec(
        name="integration_request",
        type="list",
        description="Named system or integration requested or blocking analysis.",
        allowed_values=["salesforce", "hubspot", "slack", "jira", "bigquery", "warehouse", "api"],
        downstream_use_cases=["integration demand filters", "partner backlog analysis"],
    ),
    FieldSpec(
        name="security_review_requested",
        type="boolean",
        description="Whether the call asks for security, compliance, redaction, audit, or access-control review.",
        allowed_values=[],
        downstream_use_cases=["security review queue", "privacy-control validation"],
    ),
    FieldSpec(
        name="data_gap",
        type="list",
        description="Missing or inconsistent source data that limits search, enrichment, or analysis.",
        allowed_values=[
            "duplicate_account_names",
            "missing_opportunity_id",
            "missing_metadata_join",
            "inconsistent_campaign_name",
            "separate_transcript_store",
            "evidence_span_required",
        ],
        downstream_use_cases=["data quality review", "agent answerability analysis"],
    ),
    FieldSpec(
        name="complaint_theme",
        type="list",
        description="Support or retail complaint category that can be filtered and counted.",
        allowed_values=["shipping_delay", "coupon_confusion", "refund_policy", "mobile_login", "branch_onboarding"],
        downstream_use_cases=["support theme search", "campaign complaint comparison"],
    ),
    FieldSpec(
        name="next_action",
        type="string",
        description="Concrete follow-up, owner action, or review step mentioned in the call.",
        allowed_values=[],
        downstream_use_cases=["follow-up tracking", "handoff review"],
        facetable=False,
        aggregatable=False,
    ),
]


LIST_RULES: dict[str, dict[str, list[str]]] = {
    "topic": {
        "call_search": ["search past", "transcript search", "searchable call", "call archive"],
        "crm_hygiene": ["crm", "salesforce", "hubspot", "account names", "opportunity"],
        "security_governance": ["security", "redaction", "retention", "audit", "access control", "sso"],
        "pipeline_review": ["pipeline", "deal", "prospect", "pricing blocked"],
        "support_analytics": ["support", "complaints", "escalation", "refund", "shipping", "mobile"],
        "campaign_analytics": ["campaign", "coupon", "marketing"],
        "customer_journey": ["journey", "onboarding", "adoption", "expansion"],
    },
    "pain_point": {
        "manual_work": ["manual work", "inconsistent notes", "summarize calls"],
        "crm_data_quality": ["duplicate account", "missing opportunity", "crm hygiene", "right salesforce account"],
        "privacy_or_access_control": ["access controls", "redaction", "regulated", "customer addresses", "raw transcripts"],
        "pricing_objection": ["pricing objection", "pricing blocked", "pricing pushback", "budget"],
        "integration_gap": ["integration", "bigquery", "transcripts are separate", "warehouse", "slack", "jira"],
        "complaint_theme": ["shipping delay", "coupon", "refund", "complaint", "mobile login"],
        "campaign_normalization": ["campaign names inconsistently", "campaign normalization"],
        "evidence_requirement": ["evidence span", "evidence snippets", "leadership will not trust"],
    },
    "product_area": {
        "search": ["search", "archive", "indexing"],
        "crm": ["crm", "salesforce", "hubspot", "opportunity"],
        "security": ["security", "sso", "audit", "redaction", "access control", "retention"],
        "integrations": ["integration", "slack", "jira", "bigquery", "warehouse", "api"],
        "analytics_dashboard": ["dashboard", "trend analysis", "weekly report", "cluster report"],
        "mobile_app": ["mobile app", "mobile login"],
        "campaigns": ["campaign", "coupon", "marketing"],
        "fulfillment": ["shipping", "warehouse", "refund"],
        "follow_up": ["next action", "follow-up", "owner"],
    },
    "integration_request": {
        "salesforce": ["salesforce"],
        "hubspot": ["hubspot"],
        "slack": ["slack"],
        "jira": ["jira"],
        "bigquery": ["bigquery"],
        "warehouse": ["warehouse"],
        "api": ["api"],
    },
    "data_gap": {
        "duplicate_account_names": ["duplicate account"],
        "missing_opportunity_id": ["missing opportunity"],
        "missing_metadata_join": ["metadata", "separate"],
        "inconsistent_campaign_name": ["campaign names inconsistently", "campaign normalization"],
        "separate_transcript_store": ["transcripts are separate"],
        "evidence_span_required": ["evidence span", "evidence snippets", "will not trust"],
    },
    "complaint_theme": {
        "shipping_delay": ["shipping delay"],
        "coupon_confusion": ["coupon", "coupon codes"],
        "refund_policy": ["refund"],
        "mobile_login": ["mobile login"],
        "branch_onboarding": ["branch onboarding"],
    },
}


def extract_call_fields(call: CallRecord, chunks: list[Chunk]) -> list[FieldExtraction]:
    return [extract_field(call, chunks, spec) for spec in FIELD_SPECS]


def extract_field(call: CallRecord, chunks: list[Chunk], spec: FieldSpec) -> FieldExtraction:
    text = call.transcript.lower()
    if spec.name in LIST_RULES:
        value, refs = list_value(spec.name, chunks)
        return extraction(call, spec.name, value, refs, bool(value), "matched keyword-backed signals")
    if spec.name == "urgency":
        value, refs = urgency(text, chunks)
        return extraction(call, spec.name, value, refs, value != "not_mentioned", "derived from urgency and escalation terms")
    if spec.name == "implementation_blocker":
        value, refs = implementation_blocker(text, chunks)
        return extraction(call, spec.name, value, refs, value != "not_mentioned", "ranked blocker signals")
    if spec.name == "renewal_risk":
        value, refs = renewal_risk(text, chunks)
        return extraction(call, spec.name, value, refs, value != "not_mentioned", "commercial/adoption risk signals")
    if spec.name == "security_review_requested":
        refs = refs_for(chunks, ["security", "sso", "audit", "redaction", "access control", "retention", "regulated"])
        return extraction(call, spec.name, bool(refs), refs, bool(refs), "security and privacy-control evidence")
    if spec.name == "next_action":
        value, refs = next_action(call, chunks)
        return extraction(call, spec.name, value, refs, value != "not_mentioned", "turn-level action sentence")
    raise ValueError(f"Unsupported field spec: {spec.name}")


def extraction(
    call: CallRecord,
    field_name: str,
    value: Any,
    refs: list[str],
    supported: bool,
    rationale: str,
) -> FieldExtraction:
    return FieldExtraction(
        call_id=call.call_id,
        field_name=field_name,
        value=value,
        confidence="high" if refs else "low",
        evidence_refs=refs,
        abstained=not supported,
        rationale=rationale,
    )


def list_value(field_name: str, chunks: list[Chunk]) -> tuple[list[str], list[str]]:
    values: list[str] = []
    refs: list[str] = []
    for value, keywords in LIST_RULES[field_name].items():
        value_refs = refs_for(chunks, keywords, limit=2)
        if value_refs:
            values.append(value)
            refs.extend(value_refs)
    return values, dedupe(refs)[:4]


def urgency(text: str, chunks: list[Chunk]) -> tuple[str, list[str]]:
    high_terms = ["urgent", "regulatory", "escalation", "strict access controls", "will not trust"]
    medium_terms = ["need", "blocker", "pilot", "review", "missing", "should not be buried"]
    if contains_any(text, high_terms):
        return "high", refs_for(chunks, high_terms)
    if contains_any(text, medium_terms):
        return "medium", refs_for(chunks, medium_terms)
    return "not_mentioned", []


def implementation_blocker(text: str, chunks: list[Chunk]) -> tuple[str, list[str]]:
    rules = [
        ("regulatory_escalation", ["regulatory", "regulated", "escalation"]),
        ("security_review", ["security review", "sso", "audit logs", "redaction", "access controls"]),
        ("missing_data", ["missing", "duplicate account", "campaign names inconsistently", "transcripts are separate"]),
        ("integration", ["integration", "bigquery", "warehouse", "slack", "jira", "salesforce"]),
        ("stakeholder_alignment", ["leadership will not trust", "reviewers", "owner"]),
    ]
    for value, terms in rules:
        if contains_any(text, terms):
            return value, refs_for(chunks, terms)
    return "not_mentioned", []


def renewal_risk(text: str, chunks: list[Chunk]) -> tuple[str, list[str]]:
    rules = [
        ("pricing_pushback", ["pricing objection", "pricing blocked", "pricing"]),
        ("low_adoption", ["adoption risk", "low adoption"]),
        ("competitor_evaluation", ["competitor", "vendor comparison"]),
        ("unresolved_blocker", ["blocker", "will not trust", "retention"]),
    ]
    for value, terms in rules:
        if contains_any(text, terms):
            return value, refs_for(chunks, terms)
    return "not_mentioned", []


def next_action(call: CallRecord, chunks: list[Chunk]) -> tuple[str, list[str]]:
    action_terms = ["next step", "we will", "will add", "can start", "start with", "review", "please"]
    for turn in call.turns:
        lower = turn.text.lower()
        if contains_any(lower, action_terms):
            return clean_sentence(turn.text), refs_for(chunks, [turn.text[:80]], limit=1)
    return "not_mentioned", []


def refs_for(chunks: list[Chunk], keywords: list[str], limit: int = 3) -> list[str]:
    refs: list[str] = []
    normalized_keywords = [keyword.lower() for keyword in keywords if keyword]
    for chunk in chunks:
        text = chunk.text.lower()
        if any(keyword in text for keyword in normalized_keywords):
            refs.append(f"chunk:{chunk.chunk_id}")
        if len(refs) >= limit:
            break
    return refs


def contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def clean_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
