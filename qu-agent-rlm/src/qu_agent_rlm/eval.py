from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agent import QueryUnderstandingAgent


def run_eval(agent: QueryUnderstandingAgent, tasks_path: Path) -> dict[str, Any]:
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    results = []
    for task in tasks:
        result = agent.answer(task["query"])
        results.append(score_task(task, result))
    passed = sum(1 for result in results if result["passed"])
    return {
        "task_count": len(results),
        "passed": passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "results": results,
    }


def score_task(task: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    expected_operation = task.get("expected_operation")
    operation_ok = result["plan"]["operation"] == expected_operation
    if expected_operation == "aggregate":
        group_ok = result["plan"].get("group_by") == task.get("expected_group_by")
        return {
            "task_id": task["task_id"],
            "passed": operation_ok and group_ok and bool(result["aggregation"]),
            "operation_ok": operation_ok,
            "group_ok": group_ok,
            "actual_call_ids": [record["call_id"] for record in result["records"]],
        }
    expected_ids = set(task.get("expected_call_ids", []))
    actual_ids = {record["call_id"] for record in result["records"]}
    return {
        "task_id": task["task_id"],
        "passed": operation_ok and expected_ids.issubset(actual_ids),
        "operation_ok": operation_ok,
        "expected_call_ids": sorted(expected_ids),
        "actual_call_ids": sorted(actual_ids),
    }

