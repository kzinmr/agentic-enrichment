from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .chunking import build_chunks
from .extraction import FieldExtractor, HeuristicFieldExtractor
from .feedback import (
    FeedbackAwareSchemaInducer,
    FeedbackSummary,
    build_feedback_report,
    empty_feedback_summary,
    load_feedback_jsonl,
    summarize_feedback,
)
from .fields import SCHEMA_VERSION
from .io import load_calls, write_artifact
from .models import (
    CallRecord,
    Chunk,
    ContentUnderstandingArtifact,
    FieldExtraction,
    FieldSpec,
    SilverCallRecord,
    TraceEvent,
)
from .schema import HeuristicSchemaInducer, SchemaInducer


def run_content_understanding(
    input_path: Path,
    output_dir: Path,
    *,
    source_sql: Path | None = None,
    max_chunk_chars: int = 900,
    field_extractor: FieldExtractor | None = None,
    schema_inducer: SchemaInducer | None = None,
    feedback_input: Path | None = None,
) -> ContentUnderstandingArtifact:
    calls = load_calls(input_path)
    feedback = summarize_feedback(load_feedback_jsonl(feedback_input), source_path=feedback_input) if feedback_input else empty_feedback_summary()
    artifact = analyze_calls(
        calls,
        source_sql=source_sql,
        max_chunk_chars=max_chunk_chars,
        field_extractor=field_extractor,
        schema_inducer=schema_inducer,
        feedback=feedback,
    )
    write_artifact(artifact, output_dir)
    return artifact


def analyze_calls(
    calls: list[CallRecord],
    *,
    source_sql: Path | None = None,
    max_chunk_chars: int = 900,
    field_extractor: FieldExtractor | None = None,
    schema_inducer: SchemaInducer | None = None,
    feedback: FeedbackSummary | None = None,
) -> ContentUnderstandingArtifact:
    trace = TraceBuilder()
    # Bare API defaults are for smoke tests; CLI/demo callers inject LLM-first components.
    extractor = field_extractor or HeuristicFieldExtractor()
    feedback = feedback or empty_feedback_summary()
    base_inducer = schema_inducer or HeuristicSchemaInducer()
    inducer = FeedbackAwareSchemaInducer(base_inducer, feedback) if feedback.has_requests else base_inducer
    chunks = build_chunks(calls, max_chars=max_chunk_chars)
    chunks_by_call = group_chunks_by_call(chunks)
    schema_specs = inducer.induce_schema(calls, chunks)
    manifest = build_manifest(
        calls,
        chunks,
        source_sql,
        schema_specs=schema_specs,
        schema_inducer=inducer.name,
        feedback=feedback,
    )
    manifest["prompt_state"] = prompt_state(inducer, extractor)
    trace.add(
        "root",
        "load_manifest",
        {"dataset_id": manifest["dataset_id"]},
        f"{len(calls)} records, {len(chunks)} retrievable chunks",
    )
    if feedback.has_requests:
        trace.add(
            "root",
            "load_feedback",
            {"source_path": feedback.source_path, "request_count": feedback.request_count},
            f"feedback_fields={len(feedback.requests)} search_failures={feedback.search_failure_count}",
        )
    trace.add(
        "root",
        "induce_schema",
        {"inducer": inducer.name, "field_count": len(schema_specs)},
        "induced silver schema from call content and manifest",
    )

    for spec in schema_specs:
        searched = search_chunks(chunks, " ".join([spec.name, spec.description] + spec.allowed_values), limit=5)
        trace.add(
            "root",
            "search_chunks",
            {"query": spec.name, "limit": 5},
            f"candidate_chunks={len(searched)}",
        )

    extractions: list[FieldExtraction] = []
    for call in calls:
        call_extractions = extractor.extract_call_fields(call, chunks_by_call.get(call.call_id, []), schema_specs)
        trace.add(
            "sub-rlm",
            "extract_call_fields",
            {
                "call_id": call.call_id,
                "field_count": len(schema_specs),
                "extractor": extractor.name,
            },
            (
                f"supported_fields={sum(not item.abstained for item in call_extractions)} "
                f"validation_errors={sum(len(item.validation_errors) for item in call_extractions)}"
            ),
        )
        extractions.extend(call_extractions)

    manifest["prompt_state"] = prompt_state(inducer, extractor)
    quality_report = build_quality_report(schema_specs, extractions, total_calls=len(calls))
    promoted_specs = promoted_fields(schema_specs, quality_report)
    silver_calls = materialize_silver_calls(calls, promoted_specs, extractions)
    catalog = build_silver_schema_catalog(promoted_specs, quality_report)
    databricks_contract = build_databricks_contract(source_sql, catalog)
    evaluation_tasks = build_evaluation_tasks(silver_calls)
    feedback_report = build_feedback_report(
        feedback,
        schema_specs=schema_specs,
        promoted_specs=promoted_specs,
        extractions=extractions,
    )
    trace.add(
        "root",
        "evaluate_quality",
        {"fields": len(schema_specs), "records": len(calls)},
        f"promoted_fields={len(promoted_specs)}",
    )
    trace.add(
        "root",
        "publish_silver_contract",
        {"schema_version": SCHEMA_VERSION},
        "wrote manifest, schema catalog, silver_calls, chunks, and trace",
    )
    return ContentUnderstandingArtifact(
        manifest=manifest,
        chunks=chunks,
        field_specs=promoted_specs,
        extractions=extractions,
        quality_report=quality_report,
        feedback_report=feedback_report,
        silver_schema_catalog=catalog,
        silver_calls=silver_calls,
        trace=trace.events,
        evaluation_tasks=evaluation_tasks,
        databricks_contract=databricks_contract,
    )


class TraceBuilder:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def add(self, actor: str, tool: str, arguments: dict[str, Any], result_summary: str) -> None:
        self.events.append(
            TraceEvent(
                step=len(self.events) + 1,
                actor=actor,
                tool=tool,
                arguments=arguments,
                result_summary=result_summary,
            )
        )


def build_manifest(
    calls: list[CallRecord],
    chunks: list[Chunk],
    source_sql: Path | None,
    *,
    schema_specs: list[FieldSpec],
    schema_inducer: str,
    feedback: FeedbackSummary,
) -> dict[str, Any]:
    chunk_ids_by_call: dict[str, list[str]] = {}
    for chunk in chunks:
        chunk_ids_by_call.setdefault(chunk.call_id, []).append(chunk.chunk_id)
    return {
        "dataset_id": dataset_id(calls),
        "schema_version": SCHEMA_VERSION,
        "source": {
            "kind": "databricks_call_records",
            "table": "call_records",
            "record_key": "calllog_id",
            "source_sql": str(source_sql) if source_sql else None,
        },
        "record_count": len(calls),
        "chunk_count": len(chunks),
        "schema_induction": {
            "inducer": schema_inducer,
            "field_count": len(schema_specs),
            "fields": [
                {
                    "name": spec.name,
                    "type": spec.type,
                    "allowed_values": spec.allowed_values,
                }
                for spec in schema_specs
            ],
        },
        "feedback_refinement": feedback.manifest_summary(),
        "records": [
            {
                "call_id": call.call_id,
                "customer_id": call.customer_id,
                "account_name": call.account_name,
                "date": call.date,
                "source_index": call.metadata.get("source_index", "call_records"),
                "source_record_id": call.metadata.get("source_record_id", call.call_id),
                "call_type": call.metadata.get("type"),
                "is_connected": call.metadata.get("is_connected"),
                "is_effective_connected": call.metadata.get("is_effective_connected"),
                "best_texts_length": call.metadata.get("best_texts_length"),
                "chunk_ids": chunk_ids_by_call.get(call.call_id, []),
            }
            for call in calls
        ],
        "retrieval_tools": {
            "search_chunks": "Keyword/BM25-like local retrieval over chunk snippets; production maps to OpenSearch or Databricks vector index.",
            "fetch_chunks": "Resolve evidence refs to approved snippets or full text under access control.",
            "query_silver": "Filter materialized silver fields without loading raw transcripts.",
            "aggregate_silver": "Group and count on fields marked aggregatable in silver_schema_catalog.json.",
            "run_sql": "Production-only Databricks SELECT over call_records or the materialized silver view.",
        },
    }


def prompt_state(schema_inducer: SchemaInducer, extractor: FieldExtractor) -> dict[str, Any]:
    base = getattr(schema_inducer, "base", None)
    return {
        "schema_inducer": getattr(schema_inducer, "last_prompt", None)
        or (getattr(base, "last_prompt", None) if base is not None else None)
        or {},
        "field_extractor": getattr(extractor, "last_prompt", None) or {},
    }


def dataset_id(calls: list[CallRecord]) -> str:
    payload = [(call.call_id, call.date, call.customer_id) for call in calls]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"call_records:{digest}"


def group_chunks_by_call(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.call_id, []).append(chunk)
    return grouped


def search_chunks(chunks: list[Chunk], query: str, limit: int) -> list[Chunk]:
    terms = {term for term in re.findall(r"[a-z0-9_]+", query.lower()) if len(term) > 2}
    scored: list[tuple[int, Chunk]] = []
    for chunk in chunks:
        haystack = chunk.text.lower()
        score = sum(1 for term in terms if term in haystack)
        if score:
            scored.append((score, chunk))
    return [chunk for _, chunk in sorted(scored, key=lambda item: (-item[0], item[1].chunk_id))[:limit]]


def build_quality_report(
    specs: list[FieldSpec],
    extractions: list[FieldExtraction],
    *,
    total_calls: int,
) -> dict[str, Any]:
    by_field: dict[str, list[FieldExtraction]] = {}
    for extraction in extractions:
        by_field.setdefault(extraction.field_name, []).append(extraction)
    fields: list[dict[str, Any]] = []
    for spec in specs:
        items = by_field.get(spec.name, [])
        supported = [item for item in items if not item.abstained]
        evidence_supported = [item for item in supported if item.evidence_refs]
        support_count = len(supported)
        fill_rate = support_count / total_calls if total_calls else 0.0
        evidence_coverage = len(evidence_supported) / support_count if support_count else 0.0
        recommended_status = "promote" if support_count > 0 and evidence_coverage >= 0.75 else "hold"
        reasons = []
        if support_count == 0:
            reasons.append("no_supported_calls")
        if support_count > 0 and evidence_coverage < 0.75:
            reasons.append("low_evidence_coverage")
        fields.append(
            {
                "field_name": spec.name,
                "type": spec.type,
                "support_call_count": support_count,
                "fill_rate": round(fill_rate, 4),
                "evidence_coverage": round(evidence_coverage, 4),
                "recommended_status": recommended_status,
                "reasons": reasons,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "total_calls": total_calls,
        "fields": fields,
        "promotion_policy": {
            "minimum_supported_calls": 1,
            "minimum_evidence_coverage": 0.75,
            "privacy": "silver fields carry refs only; raw transcript access stays behind fetch_chunks/run_sql tools",
        },
    }


def promoted_fields(specs: list[FieldSpec], quality_report: dict[str, Any]) -> list[FieldSpec]:
    quality_by_name = {field["field_name"]: field for field in quality_report["fields"]}
    return [spec for spec in specs if quality_by_name[spec.name]["recommended_status"] == "promote"]


def materialize_silver_calls(
    calls: list[CallRecord],
    promoted_specs: list[FieldSpec],
    extractions: list[FieldExtraction],
) -> list[SilverCallRecord]:
    promoted_names = {spec.name for spec in promoted_specs}
    by_call: dict[str, dict[str, FieldExtraction]] = {}
    for extraction in extractions:
        if extraction.field_name in promoted_names:
            by_call.setdefault(extraction.call_id, {})[extraction.field_name] = extraction

    records: list[SilverCallRecord] = []
    for call in calls:
        fields: dict[str, Any] = {}
        refs: dict[str, list[str]] = {}
        quality_flags: list[str] = []
        for spec in promoted_specs:
            extraction = by_call.get(call.call_id, {}).get(spec.name)
            if extraction is None:
                quality_flags.append(f"{spec.name}:missing_extraction")
                continue
            fields[spec.name] = extraction.value
            refs[spec.name] = extraction.evidence_refs
            quality_flags.extend(f"{spec.name}:{error}" for error in extraction.validation_errors)
            if not extraction.abstained and not extraction.evidence_refs:
                quality_flags.append(f"{spec.name}:missing_evidence")
        records.append(
            SilverCallRecord(
                call_id=call.call_id,
                customer_id=call.customer_id,
                account_name=call.account_name,
                date=call.date,
                source_index=str(call.metadata.get("source_index", "call_records")),
                source_record_id=str(call.metadata.get("source_record_id", call.call_id)),
                schema_version=SCHEMA_VERSION,
                fields=fields,
                evidence_refs=refs,
                quality_flags=quality_flags,
            )
        )
    return records


def build_silver_schema_catalog(specs: list[FieldSpec], quality_report: dict[str, Any]) -> dict[str, Any]:
    quality_by_name = {field["field_name"]: field for field in quality_report["fields"]}
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {"system": "databricks", "table": "call_records", "record_key": "calllog_id"},
        "publication": {
            "local": {
                "manifest": "manifest.json",
                "chunks": "chunks.jsonl",
                "data": "silver_calls.jsonl",
                "trace": "rlm_trace.jsonl",
            },
            "production": {
                "sql_source": "call_records.sql",
                "silver_view": "call_records_silver",
                "document_path": "silver",
            },
        },
        "query_planning": {
            "contract": "Use fields for filters and aggregations first; call fetch_chunks only for evidence.",
            "large_data_policy": "Root agents receive this catalog plus manifest stats, not raw call_records rows.",
            "evidence_ref_prefix": "chunk:",
        },
        "tools": {
            "search_chunks": {"args": ["query", "filters", "limit"], "returns": ["chunk_id", "call_id", "snippet"]},
            "fetch_chunks": {"args": ["chunk_ids"], "returns": ["chunk_id", "call_id", "text"]},
            "query_silver": {"args": ["filters", "limit"], "returns": ["silver_call_records"]},
            "aggregate_silver": {"args": ["group_by", "filters"], "returns": ["counts"]},
            "run_sql": {"args": ["select_sql"], "scope": "Databricks call_records or materialized silver view"},
        },
        "fields": [
            {
                "name": spec.name,
                "type": spec.type,
                "description": spec.description,
                "allowed_values": spec.allowed_values,
                "search": {
                    "filterable": spec.filterable,
                    "facetable": spec.facetable,
                    "aggregatable": spec.aggregatable,
                    "path": f"silver.fields.{spec.name}",
                },
                "evidence": {"required": True, "ref_path": f"silver.evidence_refs.{spec.name}"},
                "usage": {"good_for": spec.downstream_use_cases},
                "quality": quality_by_name.get(spec.name, {}),
            }
            for spec in specs
        ],
    }


def build_databricks_contract(source_sql: Path | None, catalog: dict[str, Any]) -> dict[str, Any]:
    sql_text = source_sql.read_text(encoding="utf-8") if source_sql else ""
    return {
        "source": {
            "system": "databricks",
            "input_table_or_view": "call_records",
            "source_sql_path": str(source_sql) if source_sql else None,
            "observed_columns": observed_columns(sql_text),
        },
        "materialization": {
            "local_artifacts": ["manifest.json", "chunks.jsonl", "silver_schema_catalog.json", "silver_calls.jsonl"],
            "production_target": "call_records_silver",
            "join_key": "calllog_id",
        },
        "allowlisted_tools": catalog["tools"],
        "sql_policy": {
            "allowed": "SELECT-only reads from call_records and writes to controlled silver materialization jobs.",
            "raw_transcript_policy": "Do not pass calllog_result wholesale to root agents; expose snippets through fetch_chunks.",
        },
    }


def observed_columns(sql_text: str) -> list[str]:
    if not sql_text:
        return []
    aliases = re.findall(r"\bAS\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql_text)
    explicit = [
        "calllog_id",
        "call_event_date",
        "finished_at_jst",
        "type",
        "client_id",
        "caller_staff_name",
        "caller_staff_email",
        "receiver_staff_name",
        "receiver_staff_email",
        "receiver_number",
        "caller_number",
        "call_to",
        "contact_detail",
        "calllog_result",
        "best_texts_length",
        "transcription_summary",
        "contact_count",
        "is_connected",
        "is_effective_connected",
    ]
    return sorted(set(explicit + aliases))


def build_evaluation_tasks(silver_calls: list[SilverCallRecord]) -> list[dict[str, Any]]:
    if not silver_calls:
        return []
    field_names = set(silver_calls[0].fields)
    tasks: list[dict[str, Any]] = []
    if "security_review_requested" in field_names:
        tasks.append(
            {
                "task_id": "security_review_queue",
                "query": "Which calls mention security review, SSO, redaction, audit logs, or strict access controls?",
                "expected_operation": "filter",
                "expected_filters": {"security_review_requested": True},
                "expected_call_ids": [
                    call.call_id for call in silver_calls if call.fields.get("security_review_requested") is True
                ],
            }
        )
    if "renewal_risk" in field_names:
        tasks.append(
            {
                "task_id": "pricing_objection_accounts",
                "query": "Which accounts have pricing objections or pricing blocked deals?",
                "expected_operation": "filter",
                "expected_filters": {"renewal_risk": "pricing_pushback"},
                "expected_call_ids": [
                    call.call_id for call in silver_calls if call.fields.get("renewal_risk") == "pricing_pushback"
                ],
            }
        )
    group_by = first_present_field(field_names, ["conversation_topic", "product_area", "customer_need"])
    if group_by:
        tasks.append(
            {
                "task_id": f"{group_by}_distribution",
                "query": f"Count calls by {group_by.replace('_', ' ')}.",
                "expected_operation": "aggregate",
                "expected_group_by": group_by,
            }
        )
    filter_field = first_present_field(field_names, ["risk_or_blocker", "complaint_theme", "mentioned_system_or_tool"])
    if filter_field:
        value = first_supported_value(silver_calls, filter_field)
        if value:
            tasks.append(
                {
                    "task_id": f"{filter_field}_{value}",
                    "query": f"Find calls where {filter_field.replace('_', ' ')} includes {value.replace('_', ' ')} and show evidence.",
                    "expected_operation": "filter",
                    "expected_filters": {filter_field: value},
                    "expected_call_ids": [
                        call.call_id
                        for call in silver_calls
                        if value_matches(call.fields.get(filter_field), value)
                    ],
                }
            )
    return tasks


def first_present_field(field_names: set[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in field_names:
            return candidate
    return None


def first_supported_value(silver_calls: list[SilverCallRecord], field_name: str) -> str | None:
    for call in silver_calls:
        value = call.fields.get(field_name)
        if isinstance(value, list) and value:
            return str(value[0])
        if value not in (None, "", False, "not_mentioned", []):
            return str(value)
    return None


def value_matches(current: Any, expected: str) -> bool:
    if isinstance(current, list):
        return expected in current
    return current == expected


def summarize_counts(values: list[Any]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        if isinstance(value, list):
            counter.update(str(item) for item in value)
        elif value not in (None, "", "not_mentioned", False):
            counter[str(value)] += 1
    return dict(sorted(counter.items()))
