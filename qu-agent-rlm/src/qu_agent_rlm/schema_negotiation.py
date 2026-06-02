from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .corpus import SilverCorpus


DECISION_VALUES = {"promote", "defer", "retrieval_only", "merge_with_existing", "needs_human_review"}


def load_field_candidates(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        path = path / "field_candidates.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"field_candidates artifact must be a JSON array: {path}")
    return [candidate for candidate in payload if isinstance(candidate, dict)]


def evaluate_field_candidates(
    corpus: SilverCorpus,
    candidates: list[dict[str, Any]] | None = None,
    *,
    query_tasks: list[dict[str, Any]] | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    candidates = candidates if candidates is not None else corpus.field_candidates
    evaluations = [
        evaluate_field_candidate(corpus, candidate, query_tasks=query_tasks or [], limit=limit)
        for candidate in candidates
    ]
    return {
        "contract": "qu.field_candidate_evaluation@2026-06-02.1",
        "candidate_count": len(evaluations),
        "decision_values": sorted(DECISION_VALUES),
        "evaluations": evaluations,
    }


def evaluate_field_candidate(
    corpus: SilverCorpus,
    candidate: dict[str, Any],
    *,
    query_tasks: list[dict[str, Any]],
    limit: int,
) -> dict[str, Any]:
    field_name = str(candidate.get("field_name", "")).strip()
    candidate_id = str(candidate.get("candidate_id") or f"field:{field_name}")
    field = corpus.fields.get(field_name)
    simulation = {
        "filter": simulate_filter(corpus, field_name, candidate, limit=limit) if field else {},
        "aggregate": simulate_aggregate(corpus, field_name, candidate) if field else {},
        "search": simulate_search(corpus, candidate, limit=limit),
        "observed_queries": matching_query_tasks(candidate, query_tasks),
    }
    decision, rationale = decide_candidate(candidate, field, simulation)
    return {
        "candidate_id": candidate_id,
        "field_name": field_name,
        "decision": decision,
        "rationale": rationale,
        "simulation": simulation,
        "input_refs": [f"field_candidate:{candidate_id}"],
        "output_schema": {
            "type": "object",
            "required": ["decision", "rationale", "simulation"],
            "properties": {
                "decision": {"enum": sorted(DECISION_VALUES)},
                "rationale": {"type": "string"},
                "simulation": {"type": "object"},
            },
        },
        "validator": "qu.field_candidate_evaluation.output@2026-06-02.1",
        "validation_result": "ok",
    }


def simulate_filter(
    corpus: SilverCorpus,
    field_name: str,
    candidate: dict[str, Any],
    *,
    limit: int,
) -> dict[str, Any]:
    filters = first_filter(candidate, field_name)
    if not filters:
        return {"supported": False, "reason": "candidate has no filterable example"}
    records = corpus.query_silver(filters, limit=limit)
    return {
        "supported": bool(records),
        "filters": filters,
        "record_count": len(records),
        "record_ids": [record["call_id"] for record in records],
    }


def simulate_aggregate(corpus: SilverCorpus, field_name: str, candidate: dict[str, Any]) -> dict[str, Any]:
    if not corpus.fields[field_name].get("search", {}).get("aggregatable", False):
        return {"supported": False, "reason": "field is not aggregatable"}
    expression = None
    for example in candidate.get("example_aggregations", []):
        if isinstance(example, dict) and example.get("expression"):
            expression = str(example["expression"])
            break
    try:
        result = corpus.aggregate_silver_result(field_name, {}, expression=expression)
    except ValueError as exc:
        return {"supported": False, "reason": str(exc)}
    return {
        "supported": bool(result.result),
        "group_by": field_name if expression is None else None,
        "expression": expression,
        "aggregation": result.result,
        "used_fields": result.used_fields,
    }


def simulate_search(corpus: SilverCorpus, candidate: dict[str, Any], *, limit: int) -> dict[str, Any]:
    query = search_probe(candidate)
    chunks = corpus.bm25_search_chunks(query, limit=limit)
    return {
        "supported": bool(chunks),
        "query": query,
        "chunk_count": len(chunks),
        "chunk_refs": [f"chunk:{chunk['chunk_id']}" for chunk in chunks],
    }


def decide_candidate(
    candidate: dict[str, Any],
    field: dict[str, Any] | None,
    simulation: dict[str, Any],
) -> tuple[str, str]:
    if candidate.get("merge_with_existing"):
        return "merge_with_existing", "Candidate should merge with an existing field contract before more QU replay."
    if field is None:
        if simulation["search"].get("supported"):
            return "retrieval_only", "Candidate is not in the active silver schema; retrieval can cover it for now."
        return "defer", "Candidate is neither active in silver nor recoverable by the current retrieval probe."
    if candidate.get("uncertainty") == "high":
        return "needs_human_review", "Candidate is active but CU marked uncertainty high."
    if simulation["filter"].get("supported") or simulation["aggregate"].get("supported"):
        return "promote", "Candidate passed QU filter or aggregate simulation against active silver records."
    if simulation["search"].get("supported"):
        return "retrieval_only", "Retrieval finds evidence, but silver filter/aggregate simulation did not pass."
    return "defer", "No QU simulation produced useful records or evidence."


def first_filter(candidate: dict[str, Any], field_name: str) -> dict[str, Any]:
    for example in candidate.get("example_filters", []):
        if isinstance(example, dict) and isinstance(example.get("filters"), dict):
            filters = example["filters"]
            if field_name in filters:
                return {field_name: filters[field_name]}
    return {}


def search_probe(candidate: dict[str, Any]) -> str:
    parts = [
        str(candidate.get("field_name", "")).replace("_", " "),
        str(candidate.get("description", "")),
        " ".join(str(value).replace("_", " ") for value in candidate.get("allowed_values", [])[:5]),
    ]
    return " ".join(part for part in parts if part.strip())


def matching_query_tasks(candidate: dict[str, Any], query_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    field_name = str(candidate.get("field_name", ""))
    intended = " ".join(str(query) for query in candidate.get("intended_queries", []))
    needle_terms = content_terms(" ".join([field_name, intended]))
    matches = []
    for task in query_tasks:
        query = str(task.get("query", ""))
        if needle_terms and any(term in query.lower().replace("_", " ") for term in needle_terms):
            matches.append(
                {
                    "task_id": task.get("task_id"),
                    "query": query,
                    "expected_operation": task.get("expected_operation"),
                }
            )
    return matches[:6]


def content_terms(text: str) -> list[str]:
    stopwords = {"calls", "count", "field", "find", "where", "with"}
    return [term for term in text.lower().replace("_", " ").split() if len(term) > 2 and term not in stopwords]
