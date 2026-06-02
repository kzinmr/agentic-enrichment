from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
from typing import Any

from .models import CallRecord, ContentUnderstandingArtifact, TranscriptTurn
from .replay import redact_for_replay


TRANSCRIPT_KEYS = ("transcript", "transcript_text", "text", "content", "transcription_summary")
CALL_ID_KEYS = ("call_id", "calllog_id", "id", "record_id")
CUSTOMER_ID_KEYS = ("customer_id", "client_id", "account_id", "tenant_id")
ACCOUNT_NAME_KEYS = ("account_name", "customer_name", "account", "contact_detail")
DATE_KEYS = ("date", "call_event_date", "call_date", "finished_at_jst", "started_at", "created_at")


def load_calls(path: Path) -> list[CallRecord]:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    if path.suffix.lower() == ".csv":
        return load_csv(path)
    raise ValueError(f"Unsupported input type: {path}. Use JSONL or CSV exports of call_records.")


def load_jsonl(path: Path) -> list[CallRecord]:
    calls: list[CallRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                calls.append(record_from_mapping(json.loads(line), f"{path}:{line_no}"))
    return calls


def load_csv(path: Path) -> list[CallRecord]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [record_from_mapping(row, f"{path}:{row_no}") for row_no, row in enumerate(csv.DictReader(handle), start=2)]


def record_from_mapping(payload: dict[str, Any], source: str) -> CallRecord:
    call_id = str(first_present(payload, CALL_ID_KEYS) or source)
    turns = turns_from_payload(payload, call_id)
    transcript = "\n".join(format_turn(turn) for turn in turns) if turns else str(first_present(payload, TRANSCRIPT_KEYS) or "")
    if not transcript.strip():
        raise ValueError(f"Missing transcript text in {source}")

    metadata = payload.get("metadata", {})
    if isinstance(metadata, str) and metadata.strip():
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {"raw_metadata": metadata}
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": metadata}
    metadata = {
        **metadata,
        "source_index": metadata.get("source_index", "call_records"),
        "source_record_id": metadata.get("source_record_id", call_id),
        "source_path": source,
    }
    for key, value in payload.items():
        if key not in metadata and key not in {"calllog_result", "metadata"}:
            metadata[key] = value

    return CallRecord(
        call_id=call_id,
        customer_id=str(first_present(payload, CUSTOMER_ID_KEYS) or "unknown"),
        account_name=str(first_present(payload, ACCOUNT_NAME_KEYS) or "unknown"),
        date=str(first_present(payload, DATE_KEYS) or "unknown"),
        transcript=transcript.strip(),
        turns=turns,
        metadata=metadata,
    )


def turns_from_payload(payload: dict[str, Any], call_id: str) -> list[TranscriptTurn]:
    result = normalize_calllog_result(payload.get("calllog_result"))
    turns: list[TranscriptTurn] = []
    for index, item in enumerate(result, start=1):
        if isinstance(item, dict):
            text = item.get("best_text") or item.get("text") or item.get("transcript")
            if text:
                turns.append(TranscriptTurn(call_id, index, str(item.get("speaker", "unknown")), str(text).strip()))
        elif item not in (None, ""):
            turns.append(TranscriptTurn(call_id, index, "unknown", str(item).strip()))
    return turns


def normalize_calllog_result(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    return [value]


def first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def format_turn(turn: TranscriptTurn) -> str:
    return f"turn:{turn.turn_index} speaker:{turn.speaker}: {turn.text}"


def write_artifact(artifact: ContentUnderstandingArtifact, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(artifact.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "quality_report.json").write_text(
        json.dumps(artifact.quality_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "feedback_report.json").write_text(
        json.dumps(artifact.feedback_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "field_candidates.json").write_text(
        json.dumps(artifact.field_candidates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "schema_negotiation.json").write_text(
        json.dumps(artifact.schema_negotiation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "silver_schema_catalog.json").write_text(
        json.dumps(artifact.silver_schema_catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "databricks_contract.json").write_text(
        json.dumps(artifact.databricks_contract, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "extraction_contract.json").write_text(
        json.dumps(artifact.extraction_contract, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "evaluation_tasks.json").write_text(
        json.dumps(artifact.evaluation_tasks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "artifact.json").write_text(
        json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(output_dir / "chunks.jsonl", artifact.chunks)
    write_jsonl(output_dir / "field_specs.jsonl", artifact.field_specs)
    write_jsonl(output_dir / "field_extractions.jsonl", artifact.extractions)
    write_jsonl(output_dir / "silver_calls.jsonl", artifact.silver_calls)
    write_jsonl(output_dir / "rlm_trace.jsonl", artifact.trace)
    # Replay-safe trace: raw transcript text (if any future event carries it) stripped by default.
    write_jsonl(
        output_dir / "rlm_trace.replay.jsonl",
        [redact_for_replay(asdict(event)) for event in artifact.trace],
    )


def write_jsonl(path: Path, records: list[object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = asdict(record) if hasattr(record, "__dataclass_fields__") else record
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
