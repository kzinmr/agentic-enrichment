from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from .llm import JSONChatClient
from .prompt_registry import DOWNSTREAM_QUERY_BOOTSTRAP_PROMPT


QUERY_TEXT_KEYS = ("query", "user_query", "text", "prompt", "message")
VALID_INTENTS = {"filter", "aggregate", "search", "multi_hop", "no_answer", "schema_gap", "unknown"}
VALID_OPERATIONS = {"filter", "aggregate", "search", "unknown"}


@dataclass
class QueryBootstrapReport:
    generation_id: str | None = None
    generator: str = "none"
    model: str | None = None
    prompt: dict[str, Any] = field(default_factory=dict)
    coverage_notes: str = ""
    requested_count: int = 0
    generated_count: int = 0
    fallback_reason: str = ""


class LLMDownstreamQueryGenerator:
    def __init__(self, llm: JSONChatClient) -> None:
        self.llm = llm
        self.last_prompt: dict[str, Any] | None = None

    def generate(
        self,
        *,
        catalog: dict[str, Any],
        manifest: dict[str, Any],
        existing_tasks: list[dict[str, Any]],
        adjusted_for: dict[str, Any],
        max_tasks: int,
        focus: str = "",
    ) -> tuple[list[dict[str, Any]], QueryBootstrapReport]:
        generation_id = f"qgen-{uuid4().hex[:12]}"
        prompt = DOWNSTREAM_QUERY_BOOTSTRAP_PROMPT.render(
            {
                "goal": "Generate downstream primary queries to probe CU schema usefulness for QU.",
                "max_tasks": max_tasks,
                "focus": focus,
                "dataset": compact_manifest(manifest),
                "schema": compact_catalog(catalog),
                "existing_queries": [task["query"] for task in existing_tasks],
                "adjusted_for": adjusted_for,
            }
        )
        self.last_prompt = prompt.metadata()
        payload = self.llm.complete_json(system=prompt.system, user=prompt.user)
        raw_tasks = payload.get("tasks") or payload.get("queries") or []
        if not isinstance(raw_tasks, list):
            raise ValueError("query bootstrap LLM response must include a tasks list")
        tasks = [
            normalize_query_task(
                raw,
                index=index,
                default_source_type="synthetic_llm",
                default_label_status="unlabeled",
                adjusted_for=adjusted_for,
                generation_id=generation_id,
                provenance={
                    "generator": self.llm.provider_name,
                    "model": getattr(self.llm, "model", None),
                    "prompt": prompt.metadata(),
                },
            )
            for index, raw in enumerate(raw_tasks, start=1)
        ]
        tasks = [task for task in tasks if task is not None]
        tasks = dedupe_query_tasks(tasks)[:max_tasks]
        report = QueryBootstrapReport(
            generation_id=generation_id,
            generator=self.llm.provider_name,
            model=getattr(self.llm, "model", None),
            prompt=prompt.metadata(),
            coverage_notes=str(payload.get("coverage_notes", "")).strip(),
            requested_count=max_tasks,
            generated_count=len(tasks),
        )
        return tasks, report


def build_heuristic_bootstrap_tasks(
    *,
    catalog: dict[str, Any],
    adjusted_for: dict[str, Any],
    max_tasks: int,
) -> tuple[list[dict[str, Any]], QueryBootstrapReport]:
    generation_id = f"qgen-heuristic-{uuid4().hex[:8]}"
    fields = {field.get("name"): field for field in catalog.get("fields", []) if isinstance(field, dict)}
    raw_tasks: list[dict[str, Any]] = [
        {
            "query": "Which calls mention security review, SSO, audit logs, redaction, or strict access controls?",
            "intent": "filter",
            "expected_operation": "filter",
            "rationale": "Security review is a common downstream filter and evidence workflow.",
            "targets_schema_gaps": ["security_review_requested", "security_evidence_type"],
        },
        {
            "query": "Which accounts have pricing objections or pricing blocked deals?",
            "intent": "filter",
            "expected_operation": "filter",
            "rationale": "Pricing objections are reusable for renewal risk triage.",
            "targets_schema_gaps": ["renewal_risk"],
        },
        {
            "query": "Which calls mention founder-led sales conversations?",
            "intent": "schema_gap",
            "expected_operation": "search",
            "rationale": "This probes whether the current schema covers a reusable go-to-market signal.",
            "targets_schema_gaps": ["founder_led_sales"],
        },
    ]
    group_field = first_aggregatable_field(fields)
    if group_field:
        raw_tasks.append(
            {
                "query": f"Count calls by {group_field.replace('_', ' ')}.",
                "intent": "aggregate",
                "expected_operation": "aggregate",
                "rationale": "Aggregations test whether induced fields are useful beyond retrieval.",
                "targets_schema_gaps": [group_field],
            }
        )
    filter_field, filter_value = first_filter_value(fields)
    if filter_field and filter_value:
        raw_tasks.append(
            {
                "query": f"Find calls where {filter_field.replace('_', ' ')} includes {filter_value.replace('_', ' ')} and show evidence.",
                "intent": "filter",
                "expected_operation": "filter",
                "rationale": "Field/value filters test exact silver expressibility and evidence retrieval.",
                "targets_schema_gaps": [filter_field],
            }
        )
    raw_tasks.append(
        {
            "query": "Which calls ask for something not present in the transcripts, such as a HIPAA certification timeline?",
            "intent": "no_answer",
            "expected_operation": "search",
            "rationale": "No-answer probes help separate retrieval failure from true absence.",
            "targets_schema_gaps": ["answerability_status"],
        }
    )
    tasks = [
        normalize_query_task(
            raw,
            index=index,
            default_source_type="synthetic_heuristic",
            default_label_status="unlabeled",
            adjusted_for=adjusted_for,
            generation_id=generation_id,
            provenance={"generator": "heuristic"},
        )
        for index, raw in enumerate(raw_tasks[:max_tasks], start=1)
    ]
    tasks = [task for task in tasks if task is not None]
    return tasks, QueryBootstrapReport(
        generation_id=generation_id,
        generator="heuristic",
        requested_count=max_tasks,
        generated_count=len(tasks),
        coverage_notes="Deterministic bootstrap probes over common filter/search/aggregate/no-answer patterns.",
    )


def load_query_tasks(path: Path, *, default_source_type: str = "curated") -> list[dict[str, Any]]:
    records = load_json_records(path)
    tasks = [
        normalize_query_task(
            record,
            index=index,
            default_source_type=default_source_type,
            default_label_status="unlabeled",
            source_path=str(path),
        )
        for index, record in enumerate(records, start=1)
    ]
    return [task for task in tasks if task is not None]


def load_json_records(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("tasks", "queries", "items", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    raise ValueError(f"Unsupported query task payload in {path}")


def normalize_query_task(
    raw: Any,
    *,
    index: int,
    default_source_type: str,
    default_label_status: str,
    source_path: str | None = None,
    adjusted_for: dict[str, Any] | None = None,
    generation_id: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if isinstance(raw, str):
        payload: dict[str, Any] = {"query": raw}
    elif isinstance(raw, dict):
        payload = dict(raw)
    else:
        return None
    query = first_text(payload, QUERY_TEXT_KEYS)
    if not query:
        return None
    source_type = str(payload.get("source_type") or default_source_type).strip() or default_source_type
    label_status = str(payload.get("label_status") or default_label_status).strip() or default_label_status
    intent = normalize_choice(payload.get("intent") or payload.get("task_intent"), VALID_INTENTS, "unknown")
    expected_operation = normalize_choice(
        payload.get("expected_operation") or payload.get("operation"),
        VALID_OPERATIONS,
        "unknown",
    )
    task = {
        "task_id": str(payload.get("task_id") or default_task_id(source_type, query, index)),
        "query": query,
        "intent": intent,
        "source_type": source_type,
        "label_status": label_status,
        "split": str(payload.get("split") or "bootstrap"),
        "weight": float(payload.get("weight", 1.0) or 1.0),
        "expected_operation": expected_operation,
    }
    for key in ("expected_filters", "expected_group_by", "expected_call_ids", "expected_evidence_ids"):
        if key in payload:
            task[key] = payload[key]
    rationale = str(payload.get("rationale", "")).strip()
    if rationale:
        task["rationale"] = rationale
    targets = normalize_string_list(payload.get("targets_schema_gaps") or payload.get("targets") or [])
    if targets:
        task["targets_schema_gaps"] = targets
    if generation_id:
        task["generation_id"] = generation_id
    if adjusted_for:
        task["adjusted_for"] = adjusted_for
    merged_provenance = dict(payload.get("provenance") or {})
    if source_path:
        merged_provenance["source_path"] = source_path
    if provenance:
        merged_provenance.update(provenance)
    if merged_provenance:
        task["provenance"] = merged_provenance
    return task


def dedupe_query_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for task in tasks:
        normalized = " ".join(str(task.get("query", "")).lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(task)
    return deduped


def write_query_tasks_jsonl(path: Path, tasks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")


def query_source_counts(tasks: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(task.get("source_type", "unknown")) for task in tasks))


def compact_query_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "query": task.get("query"),
        "intent": task.get("intent"),
        "source_type": task.get("source_type"),
        "label_status": task.get("label_status"),
        "generation_id": task.get("generation_id"),
        "targets_schema_gaps": task.get("targets_schema_gaps", []),
    }


def adjusted_for_cu_schema(
    *,
    loop_id: str,
    iteration: int,
    cu_output: Path,
    catalog: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    field_names = sorted(field["name"] for field in catalog.get("fields", []) if isinstance(field, dict) and field.get("name"))
    return {
        "loop_id": loop_id,
        "iteration": iteration,
        "cu_output": str(cu_output),
        "schema_version": catalog.get("schema_version"),
        "schema_hash": stable_hash({"schema_version": catalog.get("schema_version"), "fields": field_names}),
        "schema_field_count": len(field_names),
        "schema_field_names": field_names,
        "record_count": manifest.get("record_count"),
        "chunk_count": manifest.get("chunk_count"),
        "purpose": "bootstrap downstream query distribution for CU-QU feedback",
    }


def compact_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    fields = []
    for field in catalog.get("fields", []):
        if not isinstance(field, dict):
            continue
        fields.append(
            {
                "name": field.get("name"),
                "type": field.get("type"),
                "description": field.get("description", ""),
                "allowed_values": field.get("allowed_values", [])[:12]
                if isinstance(field.get("allowed_values"), list)
                else [],
                "filterable": field.get("search", {}).get("filterable"),
                "aggregatable": field.get("search", {}).get("aggregatable"),
            }
        )
    return {"schema_version": catalog.get("schema_version"), "fields": fields}


def compact_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": manifest.get("dataset"),
        "record_count": manifest.get("record_count"),
        "chunk_count": manifest.get("chunk_count"),
        "source_sql": manifest.get("source_sql"),
        "schema_induction": manifest.get("schema_induction"),
        "feedback_refinement": manifest.get("feedback_refinement"),
    }


def first_text(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ""


def normalize_choice(value: Any, allowed: set[str], fallback: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else fallback


def normalize_string_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    values = raw if isinstance(raw, list) else [raw]
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def default_task_id(source_type: str, query: str, index: int) -> str:
    return f"{source_type}-{index:03d}-{stable_hash(query)[:8]}"


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def first_aggregatable_field(fields: dict[str, dict[str, Any]]) -> str | None:
    for name, field in fields.items():
        if isinstance(field, dict) and field.get("search", {}).get("aggregatable"):
            return str(name)
    return None


def first_filter_value(fields: dict[str, dict[str, Any]]) -> tuple[str | None, str | None]:
    for name, field in fields.items():
        if not isinstance(field, dict) or not field.get("search", {}).get("filterable", True):
            continue
        values = field.get("allowed_values")
        if isinstance(values, list):
            for value in values:
                if isinstance(value, str) and value.strip():
                    return str(name), value
    return None, None
