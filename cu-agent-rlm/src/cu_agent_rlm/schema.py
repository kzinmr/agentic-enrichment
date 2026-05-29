from __future__ import annotations

from collections import Counter
import re
from typing import Any, Protocol

from .fields import FIELD_SPECS
from .llm import JSONChatClient, LLMError
from .models import CallRecord, Chunk, FieldSpec
from .prompt_registry import PromptRender, schema_induction_prompt


class SchemaInductionError(RuntimeError):
    pass


class SchemaInducer(Protocol):
    name: str

    def induce_schema(self, calls: list[CallRecord], chunks: list[Chunk]) -> list[FieldSpec]:
        raise NotImplementedError


class StaticSchemaInducer:
    name = "static"

    def induce_schema(self, calls: list[CallRecord], chunks: list[Chunk]) -> list[FieldSpec]:
        return list(FIELD_SPECS)


class HeuristicSchemaInducer:
    name = "heuristic"

    def __init__(self, *, max_fields: int = 10, max_values_per_field: int = 10) -> None:
        self.max_fields = max_fields
        self.max_values_per_field = max_values_per_field

    def induce_schema(self, calls: list[CallRecord], chunks: list[Chunk]) -> list[FieldSpec]:
        theme_values = top_labels(chunks, max_values=self.max_values_per_field)
        systems = mentioned_systems(chunks, max_values=self.max_values_per_field)
        needs = sentence_labels(chunks, NEED_TERMS, max_values=self.max_values_per_field)
        risks = sentence_labels(chunks, RISK_TERMS, max_values=self.max_values_per_field)
        outputs = sentence_labels(chunks, OUTPUT_TERMS, max_values=self.max_values_per_field)

        specs = [
            FieldSpec(
                name="conversation_topic",
                type="list",
                description="Data-induced recurring topics or themes discussed in the calls.",
                allowed_values=theme_values,
                downstream_use_cases=["topic filtering", "theme aggregation", "routing"],
            ),
            FieldSpec(
                name="customer_need",
                type="list",
                description="Data-induced stated needs, goals, or jobs-to-be-done from customer language.",
                allowed_values=needs,
                downstream_use_cases=["need discovery", "workflow analysis", "roadmap input"],
            ),
            FieldSpec(
                name="risk_or_blocker",
                type="list",
                description="Data-induced risks, blockers, objections, missing prerequisites, or trust barriers.",
                allowed_values=risks,
                downstream_use_cases=["risk filtering", "blocker analysis", "prioritization"],
            ),
            FieldSpec(
                name="mentioned_system_or_tool",
                type="list",
                description="Named systems, tools, platforms, or data stores mentioned in calls.",
                allowed_values=systems,
                downstream_use_cases=["integration analysis", "system filters", "partner backlog"],
            ),
            FieldSpec(
                name="requested_output_or_workflow",
                type="list",
                description="Data-induced requested outputs, reports, dashboards, workflows, or review surfaces.",
                allowed_values=outputs,
                downstream_use_cases=["deliverable analysis", "workflow filters", "solution design"],
            ),
            FieldSpec(
                name="urgency",
                type="enum",
                description="Time sensitivity or escalation level inferred from the call.",
                allowed_values=["not_mentioned", "low", "medium", "high"],
                downstream_use_cases=["review prioritization", "escalation filters"],
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
        return [spec for spec in specs if spec.type in {"enum", "string"} or spec.allowed_values][: self.max_fields]


class LLMSchemaInducer:
    def __init__(
        self,
        llm: JSONChatClient,
        *,
        fallback: SchemaInducer | None = None,
        max_fields: int = 12,
        max_values_per_field: int = 12,
    ) -> None:
        self.llm = llm
        self.fallback = fallback
        self.max_fields = max_fields
        self.max_values_per_field = max_values_per_field
        self.name = llm.provider_name
        self.last_prompt: dict[str, Any] | None = None

    def induce_schema(self, calls: list[CallRecord], chunks: list[Chunk]) -> list[FieldSpec]:
        prompt = build_schema_prompt_render(calls, chunks, self.max_fields, self.max_values_per_field)
        self.last_prompt = prompt.metadata()
        try:
            payload = self.llm.complete_json(system=prompt.system, user=prompt.user)
            return validate_schema_payload(
                payload,
                max_fields=self.max_fields,
                max_values_per_field=self.max_values_per_field,
            )
        except (LLMError, ValueError, TypeError) as exc:
            if self.fallback is None:
                raise SchemaInductionError(str(exc)) from exc
            return self.fallback.induce_schema(calls, chunks)


def build_schema_prompt(
    calls: list[CallRecord],
    chunks: list[Chunk],
    max_fields: int,
    max_values_per_field: int,
) -> tuple[str, str]:
    prompt = build_schema_prompt_render(calls, chunks, max_fields, max_values_per_field)
    return prompt.system, prompt.user


def build_schema_prompt_render(
    calls: list[CallRecord],
    chunks: list[Chunk],
    max_fields: int,
    max_values_per_field: int,
) -> PromptRender:
    sample_chunks = [
        {
            "chunk_id": chunk.chunk_id,
            "call_id": chunk.call_id,
            "snippet": chunk.snippet,
            "text": chunk.text,
        }
        for chunk in chunks[: min(len(chunks), 24)]
    ]
    return schema_induction_prompt(max_fields, max_values_per_field).render(
        {
            "record_count": len(calls),
            "sample_records": [
                {
                    "call_id": call.call_id,
                    "account_name": call.account_name,
                    "date": call.date,
                    "metadata": {
                        "type": call.metadata.get("type"),
                        "is_connected": call.metadata.get("is_connected"),
                        "best_texts_length": call.metadata.get("best_texts_length"),
                    },
                }
                for call in calls[: min(len(calls), 24)]
            ],
            "chunks": sample_chunks,
        }
    )


def validate_schema_payload(
    payload: dict[str, Any],
    *,
    max_fields: int,
    max_values_per_field: int,
) -> list[FieldSpec]:
    raw_fields = payload.get("fields")
    if not isinstance(raw_fields, list):
        raise ValueError("Schema induction response must contain a fields array")
    specs: list[FieldSpec] = []
    seen: set[str] = set()
    for raw in raw_fields:
        if not isinstance(raw, dict):
            continue
        name = normalize_label(str(raw.get("name", "")))
        if not name or name in seen or name in RESERVED_FIELD_NAMES:
            continue
        field_type = str(raw.get("type", "")).strip().lower()
        if field_type not in {"list", "enum", "boolean", "string"}:
            continue
        description = str(raw.get("description", "")).strip()
        if not description:
            description = f"Induced field `{name}`."
        allowed_values = normalize_allowed_values(raw.get("allowed_values", []), limit=max_values_per_field)
        if field_type in {"list", "enum"} and not allowed_values:
            continue
        if field_type == "enum" and "not_mentioned" not in allowed_values:
            allowed_values.insert(0, "not_mentioned")
        use_cases = normalize_use_cases(raw.get("downstream_use_cases", []))
        specs.append(
            FieldSpec(
                name=name,
                type=field_type,
                description=description,
                allowed_values=allowed_values,
                downstream_use_cases=use_cases,
                filterable=bool(raw.get("filterable", True)),
                facetable=bool(raw.get("facetable", field_type != "string")),
                aggregatable=bool(raw.get("aggregatable", field_type != "string")),
            )
        )
        seen.add(name)
        if len(specs) >= max_fields:
            break
    if not specs:
        raise ValueError("Schema induction produced no valid fields")
    return specs


def top_labels(chunks: list[Chunk], *, max_values: int) -> list[str]:
    counter: Counter[str] = Counter()
    for chunk in chunks:
        tokens = content_tokens(chunk.text)
        counter.update(tokens)
        counter.update("_".join(pair) for pair in zip(tokens, tokens[1:]))
    return [label for label, _ in counter.most_common(max_values) if label]


def mentioned_systems(chunks: list[Chunk], *, max_values: int) -> list[str]:
    counter: Counter[str] = Counter()
    for chunk in chunks:
        for match in re.findall(r"\b[A-Z][A-Za-z0-9]*(?:[A-Z][A-Za-z0-9]*)?\b", chunk.text):
            label = normalize_label(match)
            if label and label not in STOPWORDS and len(label) > 2:
                counter[label] += 1
        for token in content_tokens(chunk.text):
            if token in SYSTEM_HINTS:
                counter[token] += 2
    return [label for label, _ in counter.most_common(max_values)]


def sentence_labels(chunks: list[Chunk], terms: set[str], *, max_values: int) -> list[str]:
    counter: Counter[str] = Counter()
    for chunk in chunks:
        for sentence in re.split(r"[\n.!?。]+", chunk.text):
            lower = sentence.lower()
            if not any(term in lower for term in terms):
                continue
            tokens = content_tokens(sentence)
            if len(tokens) >= 2:
                counter["_".join(tokens[:3])] += 1
            for token in tokens[:5]:
                counter[token] += 1
    return [label for label, _ in counter.most_common(max_values) if label]


def content_tokens(text: str) -> list[str]:
    tokens = [normalize_label(token) for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)]
    return [token for token in tokens if token and token not in STOPWORDS and len(token) > 2]


def normalize_allowed_values(raw_values: Any, *, limit: int) -> list[str]:
    if raw_values in (None, ""):
        return []
    values = raw_values if isinstance(raw_values, list) else [raw_values]
    normalized = []
    for value in values:
        label = normalize_label(str(value))
        if label:
            normalized.append(label)
    return dedupe(normalized)[:limit]


def normalize_use_cases(raw_values: Any) -> list[str]:
    if raw_values in (None, ""):
        return ["filtering", "aggregation", "evidence-backed analysis"]
    values = raw_values if isinstance(raw_values, list) else [raw_values]
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    return cleaned or ["filtering", "aggregation", "evidence-backed analysis"]


def normalize_label(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


STOPWORDS = {
    "about",
    "account",
    "accounts",
    "after",
    "agent",
    "also",
    "analysis",
    "analytics",
    "and",
    "any",
    "are",
    "area",
    "ask",
    "before",
    "between",
    "but",
    "call",
    "calls",
    "can",
    "cannot",
    "could",
    "customer",
    "customers",
    "data",
    "did",
    "does",
    "for",
    "from",
    "have",
    "how",
    "if",
    "into",
    "issue",
    "main",
    "many",
    "mentioned",
    "need",
    "needs",
    "next",
    "not",
    "our",
    "past",
    "please",
    "raw",
    "report",
    "review",
    "should",
    "show",
    "some",
    "speaker",
    "start",
    "that",
    "the",
    "their",
    "this",
    "today",
    "turn",
    "understand",
    "use",
    "want",
    "wants",
    "way",
    "we",
    "what",
    "when",
    "where",
    "which",
    "whether",
    "with",
    "would",
    "will",
    "you",
    "your",
    "team",
}

RESERVED_FIELD_NAMES = {
    "account_name",
    "call_date",
    "call_event_date",
    "call_id",
    "call_type",
    "calllog_id",
    "customer_id",
    "date",
    "source_index",
    "source_record_id",
}

NEED_TERMS = {"need", "want", "would", "use", "ask", "asked", "care", "compare", "identify", "detect"}
RISK_TERMS = {"blocker", "risk", "issue", "missing", "strict", "regulated", "objection", "duplicate", "trust", "avoid"}
OUTPUT_TERMS = {"dashboard", "report", "archive", "search", "evidence", "summary", "workflow", "draft", "snippets"}
SYSTEM_HINTS = {
    "api",
    "bigquery",
    "crm",
    "hubspot",
    "jira",
    "salesforce",
    "slack",
    "warehouse",
}
