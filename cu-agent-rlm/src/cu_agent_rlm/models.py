from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TranscriptTurn:
    call_id: str
    turn_index: int
    speaker: str
    text: str


@dataclass
class CallRecord:
    call_id: str
    customer_id: str
    account_name: str
    date: str
    transcript: str
    turns: list[TranscriptTurn]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    call_id: str
    turn_start: int
    turn_end: int
    speaker_set: list[str]
    text: str
    snippet: str
    char_count: int


@dataclass
class FieldSpec:
    name: str
    type: str
    description: str
    allowed_values: list[str]
    downstream_use_cases: list[str]
    filterable: bool = True
    facetable: bool = True
    aggregatable: bool = True


@dataclass
class FieldExtraction:
    call_id: str
    field_name: str
    value: Any
    confidence: str
    evidence_refs: list[str]
    abstained: bool
    rationale: str
    validation_errors: list[str] = field(default_factory=list)


@dataclass
class SilverCallRecord:
    call_id: str
    customer_id: str
    account_name: str
    date: str
    source_index: str
    source_record_id: str
    schema_version: str
    fields: dict[str, Any]
    evidence_refs: dict[str, list[str]]
    quality_flags: list[str] = field(default_factory=list)


@dataclass
class TraceEvent:
    step: int
    actor: str
    tool: str
    arguments: dict[str, Any]
    result_summary: str
    latency_ms: float | None = None
    tokens: dict[str, Any] = field(default_factory=dict)
    fallback_reason: str | None = None
    prompt_hash: str | None = None
    validation_result: str | None = None


@dataclass
class ContentUnderstandingArtifact:
    manifest: dict[str, Any]
    chunks: list[Chunk]
    field_specs: list[FieldSpec]
    extractions: list[FieldExtraction]
    quality_report: dict[str, Any]
    feedback_report: dict[str, Any]
    silver_schema_catalog: dict[str, Any]
    silver_calls: list[SilverCallRecord]
    trace: list[TraceEvent]
    evaluation_tasks: list[dict[str, Any]]
    databricks_contract: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
