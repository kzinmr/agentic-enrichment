from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .corpus import ToolEvent
from .llm import JSONChatClient, LLMError
from .planner import QueryPlan, QueryToolStep
from .prompt_registry import RERANKER_PROMPT, RETRIEVAL_CONTROLLER_PROMPT
from .retrieval import english_tokenize


@dataclass
class SearchExecutionPolicy:
    min_calls: int = 1
    max_iterations: int = 0
    query_diversity_threshold: float = 0.8


class SearchStepExecutor(Protocol):
    def __call__(
        self,
        step: QueryToolStep,
        *,
        query: str,
        plan: QueryPlan,
        state: dict[str, Any],
        limit: int,
        trace: list[ToolEvent],
    ) -> None:
        raise NotImplementedError


class RetrievalSubAgent(Protocol):
    name: str

    def after_search_step(
        self,
        *,
        query: str,
        plan: QueryPlan,
        state: dict[str, Any],
        limit: int,
        trace: list[ToolEvent],
        execute_step: SearchStepExecutor,
        available_tools: list[str],
        trace_tool_name: Any,
    ) -> None:
        raise NotImplementedError

    def finalize(
        self,
        *,
        query: str,
        plan: QueryPlan,
        state: dict[str, Any],
        limit: int,
        trace: list[ToolEvent],
    ) -> None:
        raise NotImplementedError


class AgenticRetrievalSubAgent:
    def __init__(
        self,
        *,
        controller: JSONChatClient | None = None,
        reranker: JSONChatClient | None = None,
        policy: SearchExecutionPolicy | None = None,
        name: str | None = None,
    ) -> None:
        self.controller = controller
        self.reranker = reranker
        self.policy = policy or SearchExecutionPolicy()
        controller_name = controller.provider_name if controller is not None else "executor"
        self.name = name or f"agentic_retrieval:{controller_name}"
        self.last_controller_prompt: dict[str, Any] | None = None
        self.last_reranker_prompt: dict[str, Any] | None = None

    def after_search_step(
        self,
        *,
        query: str,
        plan: QueryPlan,
        state: dict[str, Any],
        limit: int,
        trace: list[ToolEvent],
        execute_step: SearchStepExecutor,
        available_tools: list[str],
        trace_tool_name: Any,
    ) -> None:
        existing_calls = len(state["search_calls"])
        required_extra_calls = max(0, self.policy.min_calls - existing_calls)
        iteration_budget = max(self.policy.max_iterations, required_extra_calls)
        if iteration_budget <= 0:
            return

        iterations = 0
        while iterations < iteration_budget:
            if len(state["search_calls"]) >= self.policy.min_calls and self.controller is None:
                return
            next_step: QueryToolStep | None = None
            if self.controller is not None:
                try:
                    next_step = self.propose_search_iteration(
                        query=query,
                        plan=plan,
                        state=state,
                        limit=limit,
                        available_tools=available_tools,
                    )
                except (LLMError, ValueError, TypeError) as exc:
                    trace.append(
                        ToolEvent(
                            step=len(trace) + 1,
                            tool="search_iteration:error",
                            arguments={"subagent": self.name, "provider": self.controller.provider_name},
                            result_summary=str(exc)[:240],
                            fallback_reason=str(exc)[:240],
                            validation_result="error",
                        )
                    )
                    if len(state["search_calls"]) >= self.policy.min_calls:
                        return
            if next_step is None:
                if len(state["search_calls"]) >= self.policy.min_calls:
                    return
                next_step = self.forced_search_iteration(
                    query=query,
                    plan=plan,
                    state=state,
                    limit=limit,
                    available_tools=available_tools,
                )
            if next_step is None:
                return
            diversity_error = self.validate_search_diversity(
                next_step,
                state["search_calls"],
                trace_tool_name=trace_tool_name,
            )
            if diversity_error:
                trace.append(
                    ToolEvent(
                        step=len(trace) + 1,
                        tool="search_iteration:rejected",
                        arguments={
                            "subagent": self.name,
                            "tool": next_step.tool,
                            "query": next_step.arguments.get("query"),
                        },
                        result_summary=diversity_error,
                    )
                )
                if len(state["search_calls"]) >= self.policy.min_calls:
                    return
                next_step = self.forced_search_iteration(
                    query=query,
                    plan=plan,
                    state=state,
                    limit=limit,
                    available_tools=available_tools,
                )
                if next_step is None:
                    return
                forced_error = self.validate_search_diversity(
                    next_step,
                    state["search_calls"],
                    trace_tool_name=trace_tool_name,
                )
                if forced_error:
                    trace.append(
                        ToolEvent(
                            step=len(trace) + 1,
                            tool="search_iteration:rejected",
                            arguments={
                                "subagent": self.name,
                                "tool": next_step.tool,
                                "query": next_step.arguments.get("query"),
                            },
                            result_summary=forced_error,
                        )
                    )
                    return
            execute_step(next_step, query=query, plan=plan, state=state, limit=limit, trace=trace)
            iterations += 1

    def propose_search_iteration(
        self,
        *,
        query: str,
        plan: QueryPlan,
        state: dict[str, Any],
        limit: int,
        available_tools: list[str],
    ) -> QueryToolStep | None:
        if self.controller is None:
            return None
        prompt = RETRIEVAL_CONTROLLER_PROMPT.render(
            {
                "query": query,
                "operation": plan.operation,
                "filters": plan.filters,
                "group_by": plan.group_by,
                "ranking_query": plan.ranking_query,
                "available_tools": available_tools,
                "minimum_search_calls": self.policy.min_calls,
                "search_calls_so_far": len(state["search_calls"]),
                "previous_searches": summarize_search_calls_for_llm(state["search_calls"]),
                "candidate_chunks": chunks_for_llm(state["chunks"][:12]),
            }
        )
        self.last_controller_prompt = prompt.metadata()
        state["retrieval_prompt"] = self.last_controller_prompt
        payload = self.controller.complete_json(system=prompt.system, user=prompt.user)
        action = str(payload.get("action", "")).strip().lower()
        failure_reason = str(payload.get("failure_reason", "")).strip()
        if failure_reason:
            state["search_failures"] = dedupe_dicts(
                [
                    *state["search_failures"],
                    {
                        "tool": self.name,
                        "query": query,
                        "failure_reason": failure_reason,
                        "query_terms": english_tokenize(query),
                        "matched_terms": [],
                    },
                ],
                key="failure_reason",
            )
        if action == "stop":
            return None
        if action != "search":
            raise ValueError(f"Invalid retrieval subagent action: {action!r}")
        tool = str(payload.get("tool", "")).strip()
        if tool not in available_tools:
            raise ValueError(f"Retrieval subagent selected unavailable tool: {tool}")
        step_limit = bounded_limit(payload.get("limit"), default=min(limit, 8))
        purpose = str(payload.get("reason", "")).strip() or f"{self.name} follow-up retrieval."
        fanout_queries = normalize_subquery_payload(payload.get("queries"))
        if fanout_queries:
            # The controller can split one decision into several parallel subqueries.
            return QueryToolStep(
                tool=tool,
                arguments={"queries": fanout_queries, "filters": plan.filters, "limit": step_limit, "promote_records": True},
                purpose=purpose,
            )
        next_query = str(payload.get("query", "")).strip()
        if not next_query:
            raise ValueError("Retrieval subagent selected an empty query")
        return QueryToolStep(
            tool=tool,
            arguments={"query": next_query, "filters": plan.filters, "limit": step_limit, "promote_records": True},
            purpose=purpose,
        )

    def forced_search_iteration(
        self,
        *,
        query: str,
        plan: QueryPlan,
        state: dict[str, Any],
        limit: int,
        available_tools: list[str],
    ) -> QueryToolStep | None:
        used_tools = {call["tool"] for call in state["search_calls"]}
        if "embedding_search_chunks" in available_tools and "embedding_search_chunks" not in used_tools:
            tool = "embedding_search_chunks"
            next_query = plan.ranking_query or query
        elif "bm25_search_chunks" in available_tools and "bm25_search_chunks" not in used_tools:
            tool = "bm25_search_chunks"
            next_query = plan.ranking_query or query
        else:
            tool = available_tools[0] if available_tools else ""
            next_query = diversified_query(query, state["search_calls"])
        if not tool or not next_query:
            return None
        return QueryToolStep(
            tool=tool,
            arguments={"query": next_query, "filters": plan.filters, "limit": min(limit, 8), "promote_records": True},
            purpose=f"{self.name} enforced minimum diverse retrieval call.",
        )

    def validate_search_diversity(
        self,
        step: QueryToolStep,
        previous_calls: list[dict[str, Any]],
        *,
        trace_tool_name: Any,
    ) -> str:
        # A fan-out step carries several subqueries; each is checked against prior calls and
        # against the earlier subqueries in the same step so the fan-out stays diverse too.
        subqueries = step_subqueries(step)
        if not subqueries:
            return "Rejected empty search query."
        tool_name = trace_tool_name(step.tool)
        prior_terms: list[set[str]] = [
            set(call.get("query_terms") or english_tokenize(str(call.get("query", ""))))
            for call in previous_calls
            if call.get("tool") == tool_name
        ]
        prior_normalized = {
            normalize_query(str(call.get("query", "")))
            for call in previous_calls
            if call.get("tool") == tool_name
        }
        seen_terms: list[set[str]] = []
        seen_normalized: set[str] = set()
        for query in subqueries:
            candidate_terms = set(english_tokenize(query))
            normalized = normalize_query(query)
            if normalized in prior_normalized or normalized in seen_normalized:
                return "Rejected duplicate query/tool pair."
            for previous_terms in (*prior_terms, *seen_terms):
                if not candidate_terms or not previous_terms:
                    continue
                overlap = len(candidate_terms & previous_terms) / len(candidate_terms | previous_terms)
                if overlap > self.policy.query_diversity_threshold:
                    return f"Rejected low-diversity query for {step.tool}: term overlap={overlap:.2f}."
            seen_terms.append(candidate_terms)
            seen_normalized.add(normalized)
        return ""

    def finalize(
        self,
        *,
        query: str,
        plan: QueryPlan,
        state: dict[str, Any],
        limit: int,
        trace: list[ToolEvent],
    ) -> None:
        if self.reranker is None or not state["chunks"]:
            return
        prompt = RERANKER_PROMPT.render(
            {
                "query": query,
                "operation": plan.operation,
                "filters": plan.filters,
                "candidate_chunks": chunks_for_llm(state["chunks"][:20]),
            }
        )
        self.last_reranker_prompt = prompt.metadata()
        state["reranker_prompt"] = self.last_reranker_prompt
        try:
            payload = self.reranker.complete_json(system=prompt.system, user=prompt.user)
            reranked_chunks, details = apply_rerank_payload(state["chunks"], payload, limit=max(limit, 10))
        except (LLMError, ValueError, TypeError) as exc:
            trace.append(
                ToolEvent(
                    step=len(trace) + 1,
                    tool="llm_rerank:error",
                    arguments={"subagent": self.name, "provider": self.reranker.provider_name},
                    result_summary=str(exc)[:240],
                    fallback_reason=str(exc)[:240],
                    validation_result="error",
                )
            )
            return
        state["chunks"] = reranked_chunks
        state["rerank"] = details
        state["rerank"]["prompt"] = self.last_reranker_prompt
        if state["evidence"]:
            state["evidence"] = chunks_to_evidence(reranked_chunks[: len(state["evidence"])])
        trace.append(
            ToolEvent(
                step=len(trace) + 1,
                tool=f"llm_rerank:{self.reranker.provider_name}",
                arguments={"subagent": self.name, "candidate_count": len(details.get("ranked_chunks", []))},
                result_summary=details.get("reasoning", "reranked candidate chunks")[:240],
            )
        )


def summarize_search_call(
    *,
    tool: str,
    query: str,
    filters: dict[str, Any],
    chunks: list[dict[str, Any]],
    purpose: str,
) -> dict[str, Any]:
    query_terms = dedupe(english_tokenize(query))
    matched_terms = dedupe(
        [
            str(term)
            for chunk in chunks
            for term in chunk.get("matched_terms", [])
            if str(term).strip()
        ]
    )
    if not matched_terms:
        matched_terms = dedupe(
            [
                term
                for term in query_terms
                if any(term in str(chunk.get("text", "")).lower().replace("_", " ") for chunk in chunks)
            ]
        )
    missing_terms = [term for term in query_terms if term not in matched_terms]
    failure_reason = ""
    if not chunks:
        failure_reason = f"{tool} returned no chunks for query terms: {', '.join(query_terms) or query}."
    elif query_terms and not matched_terms and tool.startswith("bm25"):
        failure_reason = f"{tool} returned chunks without direct lexical matches for query terms: {', '.join(query_terms)}."
    top_chunks = [
        {
            "chunk_id": chunk["chunk_id"],
            "call_id": chunk["call_id"],
            "bm25_score": chunk.get("bm25_score"),
            "embedding_score": chunk.get("embedding_score"),
            "matched_terms": chunk.get("matched_terms", []),
            "snippet": chunk.get("snippet") or str(chunk.get("text", ""))[:180],
        }
        for chunk in chunks[:5]
    ]
    return {
        "tool": tool,
        "query": query,
        "filters": filters,
        "purpose": purpose,
        "result_count": len(chunks),
        "query_terms": query_terms,
        "matched_terms": matched_terms,
        "missing_terms": missing_terms,
        "top_chunks": top_chunks,
        "failure_reason": failure_reason,
        "summary": f"{len(chunks)} chunks; matched_terms={matched_terms[:8]}; missing_terms={missing_terms[:8]}",
    }


def summarize_search_calls_for_llm(search_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "tool": call.get("tool"),
            "query": call.get("query"),
            "result_count": call.get("result_count"),
            "matched_terms": call.get("matched_terms", []),
            "missing_terms": call.get("missing_terms", []),
            "failure_reason": call.get("failure_reason", ""),
            "top_chunks": call.get("top_chunks", [])[:3],
        }
        for call in search_calls[-6:]
    ]


def chunks_for_llm(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk["chunk_id"],
            "call_id": chunk["call_id"],
            "snippet": chunk.get("snippet") or str(chunk.get("text", ""))[:320],
            "bm25_score": chunk.get("bm25_score"),
            "embedding_score": chunk.get("embedding_score"),
            "matched_terms": chunk.get("matched_terms", []),
            "query_terms": chunk.get("query_terms", []),
            "score_details": compact_score_details(chunk.get("score_details")),
        }
        for chunk in chunks
    ]


def compact_score_details(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    return {
        "backend": raw.get("backend"),
        "backend_score": raw.get("backend_score"),
        "missing_terms": raw.get("missing_terms", []),
        "terms": raw.get("terms", [])[:8],
    }


def apply_rerank_payload(
    chunks: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_ranked = payload.get("ranked_chunks", [])
    if not isinstance(raw_ranked, list):
        raise ValueError("reranker response ranked_chunks must be a list")
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    ranked: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_ranked:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id", "")).strip()
        if chunk_id not in chunks_by_id or chunk_id in seen:
            continue
        relevance = safe_float(item.get("relevance"), default=0.0)
        reason = str(item.get("reason", "")).strip()
        chunk = dict(chunks_by_id[chunk_id])
        chunk["rerank_score"] = round(max(0.0, min(relevance, 1.0)), 6)
        if reason:
            chunk["rerank_reason"] = reason
        ranked.append(chunk)
        details.append({"chunk_id": chunk_id, "relevance": chunk["rerank_score"], "reason": reason})
        seen.add(chunk_id)
    if not ranked:
        raise ValueError("reranker returned no valid candidate chunk ids")
    return ranked[:limit], {
        "reasoning": str(payload.get("reasoning", "")).strip(),
        "ranked_chunks": details,
    }


def chunks_to_evidence(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence = []
    for chunk in chunks:
        item = {
            "chunk_id": chunk["chunk_id"],
            "call_id": chunk["call_id"],
            "snippet": chunk.get("snippet") or chunk["text"][:220],
        }
        for key in ("bm25_score", "embedding_score", "rerank_score", "matched_terms", "query_terms", "score_details"):
            if key in chunk:
                item[key] = chunk[key]
        evidence.append(item)
    return evidence


def bounded_limit(value: Any, *, default: int) -> int:
    try:
        return max(1, min(int(value), 100))
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def dedupe_dicts(values: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for value in values:
        marker = str(value.get(key, "")).strip()
        if not marker or marker in seen:
            continue
        result.append(value)
        seen.add(marker)
    return result


def normalize_query(query: str) -> str:
    return " ".join(english_tokenize(query))


def normalize_subquery_payload(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    queries: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in queries:
            queries.append(text)
    return queries[:20]


def step_subqueries(step: QueryToolStep) -> list[str]:
    queries = normalize_subquery_payload(step.arguments.get("queries"))
    if queries:
        return queries
    single = str(step.arguments.get("query", "")).strip()
    return [single] if single else []


def diversified_query(query: str, search_calls: list[dict[str, Any]]) -> str:
    missing_terms = dedupe(
        [
            str(term)
            for call in search_calls
            for term in call.get("missing_terms", [])
            if str(term).strip()
        ]
    )
    if missing_terms:
        return " ".join(missing_terms[:8])
    tokens = english_tokenize(query)
    return " ".join(tokens[1:] or tokens)
