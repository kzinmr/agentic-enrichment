from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROMOTION_GATE = "external_hand_labeled_eval_required"


def build_prompt_repair_requests(result: dict[str, Any]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    query = str(result.get("query", ""))
    prompt_state = result.get("prompt_state", {}) if isinstance(result.get("prompt_state"), dict) else {}
    plan = result.get("plan", {}) if isinstance(result.get("plan"), dict) else {}
    judgement = result.get("judgement", {}) if isinstance(result.get("judgement"), dict) else {}
    diagnostics = result.get("search_diagnostics", {}) if isinstance(result.get("search_diagnostics"), dict) else {}

    planner_prompt = prompt_state.get("planner", {}) if isinstance(prompt_state.get("planner"), dict) else {}
    if "fallback" in str(plan.get("planner", "")) or "LLM planner failed" in str(plan.get("reasoning", "")):
        requests.append(
            repair_request(
                query=query,
                target="qu.query_planner",
                prompt=planner_prompt,
                reason=str(plan.get("reasoning", "planner fallback")),
                signals={"plan": plan},
            )
        )

    failures = diagnostics.get("failures", [])
    if isinstance(failures, list) and failures:
        requests.append(
            repair_request(
                query=query,
                target="qu.retrieval_controller",
                prompt=prompt_state.get("retrieval_subagent", {}),
                reason="Retrieval diagnostics reported failed or insufficient searches.",
                signals={"search_failures": failures[:6], "subagent": diagnostics.get("subagent")},
            )
        )

    if judgement and (not judgement.get("success", True) or judgement.get("needs_cu_feedback")):
        requests.append(
            repair_request(
                query=query,
                target="qu.answer_judge",
                prompt=prompt_state.get("answer_judge", {}),
                reason=str(judgement.get("rationale", "answerability judgement requested improvement")),
                signals={
                    "success": judgement.get("success"),
                    "needs_cu_feedback": judgement.get("needs_cu_feedback"),
                    "confidence": judgement.get("confidence"),
                    "failure_modes": judgement.get("failure_modes", []),
                },
            )
        )

    column_requests = result.get("column_requests", [])
    if isinstance(column_requests, list) and column_requests:
        requests.append(
            repair_request(
                query=query,
                target="cu.schema_induction",
                prompt={},
                reason="QU needed CU schema feedback; keep as prompt/schema improvement signal only until eval gate exists.",
                signals={"column_requests": column_requests[:6]},
            )
        )

    return requests


def append_prompt_repair_requests(path: Path, result: dict[str, Any]) -> int:
    requests = build_prompt_repair_requests(result)
    if not requests:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(request, ensure_ascii=False, sort_keys=True) + "\n")
    return len(requests)


def repair_request(
    *,
    query: str,
    target: str,
    prompt: Any,
    reason: str,
    signals: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source": "qu-agent-rlm",
        "target": target,
        "query": query,
        "prompt": prompt if isinstance(prompt, dict) else {},
        "reason": reason,
        "signals": signals,
        "action": "collect_signal_only",
        "promotion_gate": PROMOTION_GATE,
        "early_stopping_policy": "Do not auto-promote prompt changes without external eval fixtures.",
    }
