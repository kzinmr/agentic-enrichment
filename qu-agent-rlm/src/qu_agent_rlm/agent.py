from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any

from .corpus import SilverCorpus, ToolEvent
from .judge import AnswerJudge, HeuristicAnswerJudge
from .llm import JSONChatClient
from .planner import (
    ColumnRequest,
    HeuristicQueryPlanner,
    QueryPlan,
    QueryPlanner,
    QueryToolStep,
    default_steps_for_plan,
)
from .retrieval import english_tokenize
from .retrieval_agent import AgenticRetrievalSubAgent, RetrievalSubAgent, SearchExecutionPolicy
from .usage import usage_delta, usage_summary_from_components


SEARCH_TOOL_NAMES = {"search_chunks", "bm25_search_chunks", "embedding_search_chunks"}


class QueryUnderstandingAgent:
    def __init__(
        self,
        corpus: SilverCorpus,
        planner: QueryPlanner | None = None,
        *,
        retrieval_mode: str = "bm25",
        search_controller: JSONChatClient | None = None,
        reranker: JSONChatClient | None = None,
        retrieval_subagent: RetrievalSubAgent | None = None,
        answer_judge: AnswerJudge | None = None,
        search_policy: SearchExecutionPolicy | None = None,
        max_errors: int = 3,
        max_budget_usd: float | None = None,
        max_timeout_seconds: float | None = None,
    ) -> None:
        self.corpus = corpus
        # Bare constructor defaults are for smoke tests; CLI/demo callers inject LLM-first components.
        self.planner = planner or HeuristicQueryPlanner()
        self.retrieval_mode = retrieval_mode
        self.search_controller = search_controller
        self.reranker = reranker
        self.answer_judge = answer_judge or HeuristicAnswerJudge()
        self.search_policy = search_policy or SearchExecutionPolicy()
        self.max_errors = max_errors
        self.max_budget_usd = max_budget_usd
        self.max_timeout_seconds = max_timeout_seconds
        self.retrieval_subagent = retrieval_subagent or AgenticRetrievalSubAgent(
            controller=search_controller,
            reranker=reranker,
            policy=self.search_policy,
        )

    def answer(self, query: str, *, limit: int = 10) -> dict[str, Any]:
        trace: list[ToolEvent] = []
        guardrails = GuardrailState(
            max_errors=self.max_errors,
            max_budget_usd=self.max_budget_usd,
            max_timeout_seconds=self.max_timeout_seconds,
        )
        before_usage = self.usage_summary()
        plan_started = time.perf_counter()
        plan = self.planner.plan(query, self.corpus.catalog)
        plan_tokens = usage_delta(self.usage_summary(), before_usage)
        if not plan.steps:
            plan.steps = default_steps_for_plan(plan)
        trace.append(
            ToolEvent(
                step=1,
                tool="load_schema",
                arguments={"schema_version": self.corpus.catalog.get("schema_version")},
                result_summary=f"{len(self.corpus.fields)} fields available",
            )
        )
        trace.append(
            ToolEvent(
                step=2,
                tool=f"plan_query:{plan.planner}",
                arguments={"query": query},
                result_summary=(
                    f"operation={plan.operation} filters={plan.filters} "
                    f"group_by={plan.group_by} steps={len(plan.steps)} reasoning={plan.reasoning}"
                ),
                latency_ms=elapsed_ms(plan_started),
                tokens=plan_tokens,
                prompt_hash=prompt_hash_from_component(self.planner),
                validation_result="ok",
            )
        )

        state: dict[str, Any] = {
            "records": [],
            "chunks": [],
            "aggregation": {},
            "evidence": [],
            "column_requests": list(plan.column_requests),
            "search_calls": [],
            "search_failures": [],
            "rerank": {},
            "retrieval_prompt": {},
            "reranker_prompt": {},
        }
        for step in plan.steps:
            stop_reason = guardrails.stop_reason(self.usage_summary())
            if stop_reason:
                append_guardrail_trace(trace, stop_reason)
                break
            try:
                step_started = time.perf_counter()
                trace_len_before = len(trace)
                self.execute_step(step, query=query, plan=plan, state=state, limit=limit, trace=trace)
                annotate_new_events(trace, trace_len_before, latency_ms=elapsed_ms(step_started), tokens={})
                guardrails.record_success()
            except (ValueError, TypeError) as exc:
                guardrails.record_error()
                trace.append(
                    ToolEvent(
                        step=len(trace) + 1,
                        tool="execute_step:error",
                        arguments={"tool": step.tool, "purpose": step.purpose},
                        result_summary=str(exc)[:240],
                        fallback_reason=str(exc)[:240],
                        validation_result="error",
                    )
                )
                if guardrails.stop_reason(self.usage_summary()):
                    append_guardrail_trace(trace, guardrails.stop_reason_value or "max_errors_exceeded")
                    break
            if step.tool in SEARCH_TOOL_NAMES:
                before_usage = self.usage_summary()
                retrieval_started = time.perf_counter()
                trace_len_before = len(trace)
                error_count_before = trace_error_count(trace)
                self.retrieval_subagent.after_search_step(
                    query=query,
                    plan=plan,
                    state=state,
                    limit=limit,
                    trace=trace,
                    execute_step=self.execute_step,
                    available_tools=self.available_search_tools(),
                    trace_tool_name=self.trace_tool_name,
                )
                retrieval_tokens = usage_delta(self.usage_summary(), before_usage)
                annotate_new_events(
                    trace,
                    trace_len_before,
                    latency_ms=elapsed_ms(retrieval_started),
                    tokens=retrieval_tokens,
                    prompt_hash=prompt_hash_from_component(self.retrieval_subagent),
                )
                if trace_error_count(trace) > error_count_before:
                    guardrails.record_error()
                else:
                    guardrails.record_success()
                stop_reason = guardrails.stop_reason(self.usage_summary())
                if stop_reason:
                    append_guardrail_trace(trace, stop_reason)
                    break

        if not guardrails.stopped:
            before_usage = self.usage_summary()
            finalize_started = time.perf_counter()
            trace_len_before = len(trace)
            error_count_before = trace_error_count(trace)
            self.retrieval_subagent.finalize(query=query, plan=plan, state=state, limit=limit, trace=trace)
            finalize_tokens = usage_delta(self.usage_summary(), before_usage)
            annotate_new_events(
                trace,
                trace_len_before,
                latency_ms=elapsed_ms(finalize_started),
                tokens=finalize_tokens,
                prompt_hash=prompt_hash_from_component(self.retrieval_subagent),
            )
            if trace_error_count(trace) > error_count_before:
                guardrails.record_error()
            stop_reason = guardrails.stop_reason(self.usage_summary())
            if stop_reason:
                append_guardrail_trace(trace, stop_reason)

        if not state["records"] and state["chunks"]:
            state["records"] = records_for_chunks(self.corpus, state["chunks"])
        elif plan.operation == "search" and state["chunks"]:
            state["records"] = records_for_chunks(self.corpus, state["chunks"])
        if plan.retrieve_evidence and not state["evidence"] and not guardrails.stopped:
            self.execute_step(
                QueryToolStep(tool="fetch_chunks", arguments={"limit": 8}, purpose="Fetch fallback evidence."),
                query=query,
                plan=plan,
                state=state,
                limit=limit,
                trace=trace,
            )

        records = state["records"][:limit]
        aggregation = state["aggregation"]
        evidence = state["evidence"]
        column_requests = merge_column_requests(state["column_requests"])
        answer = aggregate_answer(plan, aggregation) if plan.operation == "aggregate" else search_answer(records, plan)
        before_usage = self.usage_summary()
        judge_started = time.perf_counter()
        judgement = self.answer_judge.judge(
            query=query,
            plan=plan,
            answer=answer,
            records=records,
            aggregation=aggregation,
            evidence=evidence,
            search_calls=state["search_calls"],
            search_failures=state["search_failures"],
            rerank=state["rerank"],
            column_requests=column_requests,
            catalog=self.corpus.catalog,
        )
        judge_tokens = usage_delta(self.usage_summary(), before_usage)
        if guardrails.stopped:
            judgement = judgement_with_guardrail(judgement, guardrails.stop_reason_value or "guardrail_stop")
        trace.append(
            ToolEvent(
                step=len(trace) + 1,
                tool=f"answer_judge:{judgement.get('judge', self.answer_judge.name)}",
                arguments={
                    "success": judgement.get("success"),
                    "answerable": judgement.get("answerable"),
                    "evidence_sufficient": judgement.get("evidence_sufficient"),
                    "needs_cu_feedback": judgement.get("needs_cu_feedback"),
                },
                result_summary=str(judgement.get("rationale", ""))[:240],
                latency_ms=elapsed_ms(judge_started),
                tokens=judge_tokens,
                prompt_hash=prompt_hash_from_component(self.answer_judge),
                validation_result="ok",
            )
        )
        return result_payload(
            query,
            plan,
            answer,
            records,
            aggregation,
            evidence,
            trace,
            column_requests,
            search_calls=state["search_calls"],
            search_failures=state["search_failures"],
            rerank=state["rerank"],
            judgement=judgement,
            retrieval_subagent=self.retrieval_subagent.name,
            prompt_state=self.prompt_state(judgement, state),
            usage_summary=self.usage_summary(),
            guardrails=guardrails.to_dict(),
            best_partial_answer=best_partial_answer(records, aggregation, evidence),
        )

    def usage_summary(self) -> dict[str, Any]:
        return usage_summary_from_components(
            self.planner,
            self.search_controller,
            self.reranker,
            self.retrieval_subagent,
            self.answer_judge,
        )

    def prompt_state(self, judgement: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        return {
            "planner": getattr(self.planner, "last_prompt", None) or {},
            "retrieval_subagent": state.get("retrieval_prompt", {}),
            "reranker": state.get("reranker_prompt", {}),
            "answer_judge": judgement.get("prompt", {}),
        }

    def execute_step(
        self,
        step: QueryToolStep,
        *,
        query: str,
        plan: QueryPlan,
        state: dict[str, Any],
        limit: int,
        trace: list[ToolEvent],
    ) -> None:
        arguments = dict(step.arguments)
        if step.tool in {"search_chunks", "bm25_search_chunks", "embedding_search_chunks"}:
            step_query = str(arguments.get("query") or plan.ranking_query or query)
            filters = filters_from_arguments(arguments, default=plan.filters)
            step_limit = bounded_limit(arguments.get("limit"), default=limit)
            chunks = self.search_with_tool(step.tool, step_query, filters=filters, limit=step_limit)
            trace_tool = self.trace_tool_name(step.tool)
            state["chunks"] = merge_chunks(state["chunks"], chunks)
            if plan.operation == "search" or bool(arguments.get("promote_records")):
                state["records"] = merge_records(state["records"], records_for_chunks(self.corpus, chunks))
            search_call = summarize_search_call(
                tool=trace_tool,
                query=step_query,
                filters=filters,
                chunks=chunks,
                purpose=step.purpose,
            )
            state["search_calls"].append(search_call)
            if search_call.get("failure_reason"):
                state["search_failures"] = dedupe_dicts(
                    [*state["search_failures"], search_call],
                    key="failure_reason",
                )
            trace.append(
                ToolEvent(
                    step=len(trace) + 1,
                    tool=trace_tool,
                    arguments={
                        "query": step_query,
                        "filters": filters,
                        "limit": step_limit,
                        "purpose": step.purpose,
                        "query_terms": search_call["query_terms"],
                    },
                    result_summary=search_call["summary"],
                )
            )
            return

        if step.tool == "query_silver":
            filters = filters_from_arguments(arguments, default=plan.filters)
            step_limit = bounded_limit(arguments.get("limit"), default=limit)
            records = self.corpus.query_silver(filters, limit=step_limit)
            state["records"] = merge_records(state["records"], records)
            trace.append(
                ToolEvent(
                    step=len(trace) + 1,
                    tool="query_silver",
                    arguments={"filters": filters, "limit": step_limit, "purpose": step.purpose},
                    result_summary=f"{len(records)} records",
                )
            )
            return

        if step.tool == "aggregate_silver":
            filters = filters_from_arguments(arguments, default=plan.filters)
            group_by = str(arguments.get("group_by") or plan.group_by or "")
            aggregation = self.corpus.aggregate_silver(group_by, filters) if group_by else {}
            state["aggregation"] = aggregation
            trace.append(
                ToolEvent(
                    step=len(trace) + 1,
                    tool="aggregate_silver",
                    arguments={"group_by": group_by, "filters": filters, "purpose": step.purpose},
                    result_summary=f"{len(aggregation)} groups",
                )
            )
            return

        if step.tool == "fetch_chunks":
            refs = normalize_ref_list(arguments.get("refs"))
            step_limit = bounded_limit(arguments.get("limit"), default=8)
            if not refs:
                refs = collect_evidence_refs(state["records"][:limit], plan)
            if not refs and state["chunks"]:
                refs = [f"chunk:{chunk['chunk_id']}" for chunk in state["chunks"][:step_limit]]
            refs = dedupe(refs)[:step_limit]
            chunks = self.fetch_evidence_chunks(refs, state["chunks"], step_limit=step_limit)
            state["evidence"] = chunks_to_evidence(chunks)
            trace.append(
                ToolEvent(
                    step=len(trace) + 1,
                    tool="fetch_chunks",
                    arguments={"refs": refs, "purpose": step.purpose},
                    result_summary=f"{len(chunks)} chunks",
                )
            )
            return

        if step.tool == "review_schema_gaps":
            requests = infer_column_requests(
                query=query,
                catalog=self.corpus.catalog,
                plan=plan,
                records=state["records"],
                chunks=state["chunks"],
                aggregation=state["aggregation"],
                search_failures=state["search_failures"],
            )
            state["column_requests"] = merge_column_requests([*state["column_requests"], *requests])
            trace.append(
                ToolEvent(
                    step=len(trace) + 1,
                    tool="review_schema_gaps",
                    arguments={"query": query, "purpose": step.purpose},
                    result_summary=f"{len(requests)} column request(s)",
                )
            )
            return

        raise ValueError(f"Unsupported query tool step: {step.tool}")

    def search_with_tool(
        self,
        tool: str,
        query: str,
        *,
        filters: dict[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        active_filters = filters or None
        if tool == "bm25_search_chunks":
            if self.retrieval_mode == "embedding":
                return self.corpus.embedding_search_chunks(query, filters=active_filters, limit=limit)
            if self.retrieval_mode == "hybrid":
                return self.corpus.hybrid_search_chunks(query, filters=active_filters, limit=limit)
            return self.corpus.bm25_search_chunks(query, filters=active_filters, limit=limit)
        if tool == "embedding_search_chunks":
            return self.corpus.embedding_search_chunks(query, filters=active_filters, limit=limit)
        if self.retrieval_mode == "lexical":
            return self.corpus.lexical_overlap_search_chunks(query, filters=active_filters, limit=limit)
        if self.retrieval_mode == "embedding":
            return self.corpus.embedding_search_chunks(query, filters=active_filters, limit=limit)
        if self.retrieval_mode == "hybrid":
            return self.corpus.hybrid_search_chunks(query, filters=active_filters, limit=limit)
        return self.corpus.bm25_search_chunks(query, filters=active_filters, limit=limit)

    def trace_tool_name(self, tool: str) -> str:
        if tool == "bm25_search_chunks" and self.retrieval_mode == "embedding":
            return "embedding_search_chunks"
        if tool == "bm25_search_chunks" and self.retrieval_mode == "hybrid":
            return "hybrid_search_chunks"
        if tool != "search_chunks":
            return tool
        if self.retrieval_mode == "lexical":
            return "search_chunks:lexical_overlap"
        if self.retrieval_mode == "embedding":
            return "search_chunks:embedding"
        if self.retrieval_mode == "hybrid":
            return "search_chunks:hybrid_agentic_union"
        return f"search_chunks:bm25:{self.corpus.bm25_backend_name()}"

    def available_search_tools(self) -> list[str]:
        tools = ["bm25_search_chunks", "search_chunks"]
        if self.corpus.embedding_client is not None:
            tools.insert(1, "embedding_search_chunks")
        return tools

    def fetch_evidence_chunks(
        self,
        refs: list[str],
        enriched_chunks: list[dict[str, Any]],
        *,
        step_limit: int,
    ) -> list[dict[str, Any]]:
        enriched_by_id = {chunk["chunk_id"]: chunk for chunk in enriched_chunks}
        if not refs and enriched_chunks:
            return enriched_chunks[:step_limit]
        fetched = self.corpus.fetch_chunks(refs)
        chunks = [enriched_by_id.get(chunk["chunk_id"], chunk) for chunk in fetched]
        if not chunks and enriched_chunks:
            return enriched_chunks[:step_limit]
        return chunks


def result_payload(
    query: str,
    plan: QueryPlan,
    answer: str,
    records: list[dict[str, Any]],
    aggregation: dict[str, int],
    evidence: list[dict[str, Any]],
    trace: list[ToolEvent],
    column_requests: list[ColumnRequest],
    *,
    search_calls: list[dict[str, Any]],
    search_failures: list[dict[str, Any]],
    rerank: dict[str, Any],
    judgement: dict[str, Any],
    retrieval_subagent: str,
    prompt_state: dict[str, Any],
    usage_summary: dict[str, Any],
    guardrails: dict[str, Any],
    best_partial_answer: dict[str, Any],
) -> dict[str, Any]:
    return {
        "query": query,
        "answer": answer,
        "plan": asdict(plan),
        "records": [compact_record(record) for record in records],
        "aggregation": aggregation,
        "evidence": evidence,
        "column_requests": [asdict(request) for request in column_requests],
        "search_diagnostics": {"subagent": retrieval_subagent, "calls": search_calls, "failures": search_failures},
        "rerank": rerank,
        "judgement": judgement,
        "prompt_state": prompt_state,
        "usage_summary": usage_summary,
        "guardrails": guardrails,
        "best_partial_answer": best_partial_answer,
        "trace": [asdict(event) for event in trace],
    }


class GuardrailState:
    def __init__(
        self,
        *,
        max_errors: int,
        max_budget_usd: float | None,
        max_timeout_seconds: float | None,
    ) -> None:
        self.max_errors = max_errors
        self.max_budget_usd = max_budget_usd
        self.max_timeout_seconds = max_timeout_seconds
        self.started_at = time.perf_counter()
        self.consecutive_errors = 0
        self.stop_reason_value: str | None = None

    @property
    def stopped(self) -> bool:
        return self.stop_reason_value is not None

    def record_success(self) -> None:
        self.consecutive_errors = 0

    def record_error(self) -> None:
        self.consecutive_errors += 1

    def stop_reason(self, usage_summary: dict[str, Any]) -> str | None:
        if self.stop_reason_value is not None:
            return self.stop_reason_value
        if self.max_errors > 0 and self.consecutive_errors >= self.max_errors:
            self.stop_reason_value = "max_errors_exceeded"
        elif self.max_budget_usd is not None and float(usage_summary.get("total_cost_usd", 0.0) or 0.0) >= self.max_budget_usd:
            self.stop_reason_value = "budget_exceeded"
        elif self.max_timeout_seconds is not None and time.perf_counter() - self.started_at >= self.max_timeout_seconds:
            self.stop_reason_value = "timeout"
        return self.stop_reason_value

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_errors": self.max_errors,
            "max_budget_usd": self.max_budget_usd,
            "max_timeout_seconds": self.max_timeout_seconds,
            "consecutive_errors": self.consecutive_errors,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason_value,
        }


def append_guardrail_trace(trace: list[ToolEvent], reason: str) -> None:
    trace.append(
        ToolEvent(
            step=len(trace) + 1,
            tool="guardrail_stop",
            arguments={"reason": reason},
            result_summary=f"Stopped execution because {reason}. Returning best partial answer.",
            fallback_reason=reason,
            validation_result="stopped",
        )
    )


def annotate_new_events(
    trace: list[ToolEvent],
    start_index: int,
    *,
    latency_ms: float,
    tokens: dict[str, Any],
    prompt_hash: str | None = None,
) -> None:
    for event in trace[start_index:]:
        if event.latency_ms is None:
            event.latency_ms = latency_ms
        if event.tokens is None:
            event.tokens = tokens
        elif not event.tokens and tokens:
            event.tokens = tokens
        if event.prompt_hash is None:
            event.prompt_hash = prompt_hash
        if event.validation_result is None:
            event.validation_result = "ok"


def trace_error_count(trace: list[ToolEvent]) -> int:
    return sum(1 for event in trace if event.tool.endswith(":error") or event.validation_result == "error")


def judgement_with_guardrail(judgement: dict[str, Any], reason: str) -> dict[str, Any]:
    updated = dict(judgement)
    modes = list(updated.get("failure_modes", [])) if isinstance(updated.get("failure_modes"), list) else []
    if reason not in modes:
        modes.append(reason)
    updated["failure_modes"] = modes
    updated["success"] = False
    updated["needs_cu_feedback"] = True
    rationale = str(updated.get("rationale", "")).strip()
    updated["rationale"] = f"{rationale} Guardrail stopped execution: {reason}.".strip()
    return updated


def best_partial_answer(
    records: list[dict[str, Any]],
    aggregation: dict[str, int],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "record_count": len(records),
        "aggregation_group_count": len(aggregation),
        "evidence_count": len(evidence),
        "record_ids": [record.get("call_id") for record in records[:10]],
    }


def prompt_hash_from_component(component: object) -> str | None:
    prompt = getattr(component, "last_prompt", None)
    if prompt is None:
        prompt = getattr(component, "last_controller_prompt", None) or getattr(component, "last_reranker_prompt", None)
    if isinstance(prompt, dict):
        value = prompt.get("prompt_hash")
        return str(value) if value else None
    nested = getattr(component, "llm", None)
    if nested is not None and nested is not component:
        return prompt_hash_from_component(nested)
    return None


def elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "call_id": record["call_id"],
        "account_name": record.get("account_name"),
        "date": record.get("date"),
        "fields": record.get("fields", {}),
        "quality_flags": record.get("quality_flags", []),
    }


def aggregate_answer(plan: QueryPlan, aggregation: dict[str, int]) -> str:
    if not aggregation:
        return "No matching records were found for the requested aggregation."
    pairs = ", ".join(f"{key}: {value}" for key, value in aggregation.items())
    return f"Grouped by {plan.group_by}: {pairs}."


def search_answer(records: list[dict[str, Any]], plan: QueryPlan) -> str:
    if not records:
        return "No matching calls were found."
    names = ", ".join(f"{record['call_id']} ({record.get('account_name')})" for record in records[:6])
    return f"Found {len(records)} matching call(s): {names}."


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def filters_from_arguments(arguments: dict[str, Any], *, default: dict[str, Any]) -> dict[str, Any]:
    if "filters" not in arguments:
        return default
    filters = arguments.get("filters")
    return filters if isinstance(filters, dict) else {}


def bounded_limit(value: Any, *, default: int) -> int:
    try:
        return max(1, min(int(value), 100))
    except (TypeError, ValueError):
        return default


def merge_records(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {record["call_id"] for record in existing}
    merged = list(existing)
    for record in incoming:
        call_id = record.get("call_id")
        if call_id and call_id not in seen:
            merged.append(record)
            seen.add(call_id)
    return merged


def merge_chunks(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {chunk["chunk_id"] for chunk in existing}
    merged = list(existing)
    for chunk in incoming:
        chunk_id = chunk.get("chunk_id")
        if chunk_id and chunk_id not in seen:
            merged.append(chunk)
            seen.add(chunk_id)
    return merged


def records_for_chunks(corpus: SilverCorpus, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in chunks:
        call_id = chunk.get("call_id")
        if call_id in corpus.records_by_id and call_id not in seen:
            records.append(corpus.records_by_id[call_id])
            seen.add(call_id)
    return records


def collect_evidence_refs(records: list[dict[str, Any]], plan: QueryPlan) -> list[str]:
    refs: list[str] = []
    for record in records:
        record_refs = record.get("evidence_refs", {})
        if plan.filters:
            for field_name in plan.filters:
                refs.extend(record_refs.get(field_name, []))
        elif plan.group_by:
            refs.extend(record_refs.get(plan.group_by, []))
        if not plan.filters and not plan.group_by:
            for values in record_refs.values():
                refs.extend(values)
    return refs


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


def safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
    values = suggested_values_from_query(query)
    return " ".join(values[1:] or values)


def normalize_ref_list(raw: Any) -> list[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item).strip()]


def infer_column_requests(
    *,
    query: str,
    catalog: dict[str, Any],
    plan: QueryPlan,
    records: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    aggregation: dict[str, int],
    search_failures: list[dict[str, Any]],
) -> list[ColumnRequest]:
    fields = {field["name"] for field in catalog.get("fields", [])}
    evidence_refs = [f"chunk:{chunk['chunk_id']}" for chunk in chunks[:3]]
    should_request_search_column = plan.operation == "search" and bool(chunks)
    should_request_filter_improvement = bool(plan.filters) and not records and bool(chunks)
    should_request_aggregation_improvement = plan.operation == "aggregate" and not aggregation
    should_request_failure_column = bool(search_failures) and (plan.operation == "search" or not records)
    if not (
        should_request_search_column
        or should_request_filter_improvement
        or should_request_aggregation_improvement
        or should_request_failure_column
    ):
        return []

    field_name = candidate_field_name_from_failures(query, search_failures)
    action = "improve_extraction" if field_name in fields or should_request_filter_improvement else "add_field"
    reason = "The query required semantic retrieval because no current silver field represented the requested concept."
    if should_request_filter_improvement:
        reason = "Silver filters produced no records, but semantic retrieval found candidate evidence."
    if should_request_aggregation_improvement:
        reason = "The requested aggregation could not produce groups from current silver coverage."
    if search_failures:
        failure_text = "; ".join(str(item.get("failure_reason", "")) for item in search_failures[:3] if item.get("failure_reason"))
        reason = f"{reason} Search diagnostics: {failure_text}"
    suggested_values = suggested_values_from_failures(query, search_failures)
    return [
        ColumnRequest(
            action=action,
            field_name=field_name,
            field_type="list",
            description=f"Reusable silver signal that makes downstream queries expressible without repeated retrieval: {query}",
            reason=reason,
            priority="high" if chunks else "medium",
            suggested_allowed_values=suggested_values,
            example_queries=[query],
            evidence_refs=evidence_refs,
        )
    ]


def merge_column_requests(requests: list[ColumnRequest]) -> list[ColumnRequest]:
    merged: list[ColumnRequest] = []
    seen: set[tuple[str, str]] = set()
    for request in requests:
        key = (request.action, request.field_name)
        if key in seen:
            continue
        merged.append(request)
        seen.add(key)
    return merged


def candidate_field_name(query: str) -> str:
    values = suggested_values_from_query(query)
    if not values:
        return "query_signal"
    return "_".join(values[:3])


def candidate_field_name_from_failures(query: str, search_failures: list[dict[str, Any]]) -> str:
    values = suggested_values_from_failures(query, search_failures)
    if not values:
        return candidate_field_name(query)
    return "_".join(values[:3])


def suggested_values_from_failures(query: str, search_failures: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for call in search_failures:
        for term in [*call.get("missing_terms", []), *call.get("query_terms", [])]:
            normalized = "".join(char for char in str(term).lower() if char.isalnum() or char == "_")
            if len(normalized) <= 2 or normalized in QUERY_STOPWORDS or normalized in seen:
                continue
            values.append(normalized)
            seen.add(normalized)
    for value in suggested_values_from_query(query):
        if value not in seen:
            values.append(value)
            seen.add(value)
    return values[:8]


def suggested_values_from_query(query: str) -> list[str]:
    tokens = [
        token
        for token in query.lower().replace("-", " ").replace("/", " ").split()
        if token.strip(".,?!:;()[]{}\"'")
    ]
    values: list[str] = []
    seen: set[str] = set()
    for raw in tokens:
        token = "".join(char for char in raw.strip(".,?!:;()[]{}\"'") if char.isalnum() or char == "_")
        if len(token) <= 2 or token in QUERY_STOPWORDS or token in seen:
            continue
        values.append(token)
        seen.add(token)
    return values[:8]


QUERY_STOPWORDS = {
    "and",
    "are",
    "break",
    "breakdown",
    "calls",
    "count",
    "down",
    "find",
    "for",
    "have",
    "how",
    "many",
    "mention",
    "mentions",
    "show",
    "the",
    "which",
    "with",
}
