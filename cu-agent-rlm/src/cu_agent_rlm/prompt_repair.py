from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import ContentUnderstandingArtifact


PROMOTION_GATE = "external_hand_labeled_eval_required"


def build_prompt_repair_requests(artifact: ContentUnderstandingArtifact) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    prompt_state = artifact.manifest.get("prompt_state", {})
    schema_prompt = prompt_state.get("schema_inducer", {}) if isinstance(prompt_state, dict) else {}
    extractor_prompt = prompt_state.get("field_extractor", {}) if isinstance(prompt_state, dict) else {}

    held_feedback = [
        item
        for item in artifact.feedback_report.get("requested_fields", [])
        if item.get("accepted_into_schema") and not item.get("promoted_to_silver")
    ]
    if held_feedback:
        requests.append(
            repair_request(
                target="cu.schema_induction",
                prompt=schema_prompt,
                reason="Feedback-derived fields were accepted into schema but held by quality gates.",
                signals={"held_feedback_fields": compact_feedback(held_feedback)},
            )
        )

    low_quality_fields = [
        field
        for field in artifact.quality_report.get("fields", [])
        if field.get("recommended_status") != "promote"
    ]
    if low_quality_fields:
        requests.append(
            repair_request(
                target="cu.field_extraction",
                prompt=extractor_prompt,
                reason="Some induced fields failed support or evidence coverage gates.",
                signals={"low_quality_fields": low_quality_fields[:12]},
            )
        )

    validation_error_counts = artifact.feedback_report.get("validation_error_counts", {})
    if validation_error_counts:
        requests.append(
            repair_request(
                target="cu.field_extraction",
                prompt=extractor_prompt,
                reason="Extractor validation errors were observed.",
                signals={"validation_error_counts": validation_error_counts},
            )
        )

    if artifact.feedback_report.get("feedback", {}).get("answerability_failure_count", 0):
        requests.append(
            repair_request(
                target="cu.schema_induction",
                prompt=schema_prompt,
                reason="Downstream QU answerability judgement reported failures.",
                signals={"feedback": artifact.feedback_report.get("feedback", {})},
            )
        )

    return requests


def write_prompt_repair_requests(path: Path, artifact: ContentUnderstandingArtifact) -> int:
    requests = build_prompt_repair_requests(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(request, ensure_ascii=False, sort_keys=True) + "\n")
    return len(requests)


def repair_request(*, target: str, prompt: Any, reason: str, signals: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "cu-agent-rlm",
        "target": target,
        "prompt": prompt if isinstance(prompt, dict) else {},
        "reason": reason,
        "signals": signals,
        "action": "collect_signal_only",
        "promotion_gate": PROMOTION_GATE,
        "early_stopping_policy": "Do not auto-promote prompt changes without external eval fixtures.",
    }


def compact_feedback(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "field_name": item.get("field_name"),
            "reason": item.get("reason"),
            "priority": item.get("priority"),
            "source_queries": item.get("source_queries", []),
        }
        for item in items[:12]
    ]
