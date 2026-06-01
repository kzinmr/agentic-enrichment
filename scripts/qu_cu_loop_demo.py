#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
from uuid import uuid4

from demo_tui import DemoTUI

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "cu-agent-rlm" / "src"))
sys.path.insert(0, str(REPO_ROOT / "qu-agent-rlm" / "src"))

from cu_agent_rlm.extraction import HeuristicFieldExtractor, LLMFieldExtractor
from cu_agent_rlm.llm import (
    OpenAICompatibleChatClient as CUOpenAICompatibleChatClient,
    OpenAIResponsesClient as CUOpenAIResponsesClient,
)
from cu_agent_rlm.pipeline import run_content_understanding
from cu_agent_rlm.schema import HeuristicSchemaInducer, LLMSchemaInducer
from qu_agent_rlm.agent import QueryUnderstandingAgent
from qu_agent_rlm.cli import append_feedback
from qu_agent_rlm.corpus import SilverCorpus
from qu_agent_rlm.env import default_env_file, load_env_file
from qu_agent_rlm.judge import HeuristicAnswerJudge, LLMAnswerJudge
from qu_agent_rlm.llm import (
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    JSONChatClient,
    OpenAICompatibleChatClient,
    OpenAIResponsesClient,
)
from qu_agent_rlm.planner import HeuristicQueryPlanner, LLMQueryPlanner
from qu_agent_rlm.query_tasks import (
    LLMDownstreamQueryGenerator,
    QueryBootstrapReport,
    adjusted_for_cu_schema,
    compact_query_task,
    dedupe_query_tasks,
    load_query_tasks,
    normalize_query_task,
    query_source_counts,
    write_query_tasks_jsonl,
)
from qu_agent_rlm.replay import redact_for_replay
from qu_agent_rlm.retrieval import DEFAULT_EMBEDDING_MODEL, OpenAIEmbeddingClient
from qu_agent_rlm.usage import budget_unpriced_warning
from qu_agent_rlm.retrieval_agent import AgenticRetrievalSubAgent, SearchExecutionPolicy


DEFAULT_QUERY = "Which calls mention founder-led sales calls?"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a QU-CU feedback loop demo.")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "cu-agent" / "data" / "sample_calls.jsonl")
    parser.add_argument("--source-sql", type=Path, default=REPO_ROOT / "call_records.sql")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "output" / "qu_cu_loop_demo")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--skip-default-query", action="store_true")
    parser.add_argument(
        "--query-tasks",
        type=Path,
        action="append",
        default=[],
        help="JSON/JSONL query task fixtures. These may be curated, hand-labeled, or weak-labeled.",
    )
    parser.add_argument(
        "--production-query-log",
        type=Path,
        action="append",
        default=[],
        help="JSON/JSONL production query logs. Records are normalized into query tasks with source_type=production_log.",
    )
    parser.add_argument(
        "--bootstrap-queries",
        choices=("openai", "openai-compatible"),
        default="openai",
        help="Generate extra downstream primary queries from the baseline CU schema with an LLM.",
    )
    parser.add_argument("--bootstrap-query-count", type=int, default=6)
    parser.add_argument("--bootstrap-focus", default="")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--env-file", type=Path, default=default_env_file())
    parser.add_argument("--cu-backend", choices=("openai", "openai-compatible"), default="openai")
    parser.add_argument("--qu-backend", choices=("openai", "openai-compatible"), default="openai")
    parser.add_argument(
        "--retrieval-mode",
        choices=("lexical", "bm25", "embedding", "hybrid", "agentic"),
        default="agentic",
        help="QU retrieval mode. agentic uses an LLM retrieval subagent plus deterministic search tools.",
    )
    parser.add_argument("--min-search-calls", type=int, default=2)
    parser.add_argument("--max-search-iterations", type=int, default=2)
    parser.add_argument("--query-diversity-threshold", type=float, default=0.8)
    parser.add_argument("--max-errors", type=int, default=3)
    parser.add_argument("--max-budget-usd", type=float, default=None)
    parser.add_argument("--max-timeout-seconds", type=float, default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-cache", type=Path, default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-timeout-seconds", type=int, default=60)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-timeout-seconds", type=int, default=None)
    parser.add_argument("--no-tui", action="store_true", help="Disable the terminal UI and print JSON only.")
    parser.add_argument("--print-json", action="store_true", help="Print the raw summary JSON after the TUI.")
    parser.add_argument(
        "--no-replay",
        action="store_true",
        help="Do not write redacted *.replay.json artifacts (raw transcript snippets stripped) next to QU answers.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    configure_llm_args(args)
    budget_warning = budget_unpriced_warning(args.max_budget_usd)
    if budget_warning:
        print(f"WARNING: {budget_warning}", file=sys.stderr)
    args.output_root.mkdir(parents=True, exist_ok=True)
    loop_id = f"qu-cu-loop-{uuid4().hex[:12]}"
    tui = DemoTUI(title="QU-CU Improvement Loop", enabled=not args.no_tui)
    tui.header(
        subtitle="LLM-first bootstrap, replay, feedback, and schema refinement.",
        config={
            "loop_id": loop_id,
            "output_root": args.output_root,
            "cu_backend": args.cu_backend,
            "qu_backend": args.qu_backend,
            "model": args.llm_model,
            "retrieval_mode": args.retrieval_mode,
            "bootstrap_queries": args.bootstrap_query_count,
        },
    )
    orchestration_events: list[dict[str, Any]] = []
    baseline_dir = args.output_root / "01_cu_baseline"
    query_tasks_path = args.output_root / "02_qu_feedback" / "query_tasks.jsonl"
    query_bootstrap_path = args.output_root / "02_qu_feedback" / "query_bootstrap.json"
    feedback_path = args.output_root / "02_qu_feedback" / "column_requests.jsonl"
    baseline_answer_path = args.output_root / "02_qu_feedback" / "baseline_answer.json"
    baseline_answers_dir = args.output_root / "02_qu_feedback" / "baseline_answers"
    refined_dir = args.output_root / "03_cu_refined"
    refined_answer_path = args.output_root / "04_qu_refined" / "refined_answer.json"
    refined_answers_dir = args.output_root / "04_qu_refined" / "refined_answers"
    summary_path = args.output_root / "loop_summary.json"
    orchestration_trace_path = args.output_root / "orchestration_trace.jsonl"

    with tui.phase("Baseline CU", f"Induce and extract silver fields into {baseline_dir}"):
        baseline_artifact = run_cu(args, baseline_dir)
    tui.show_cu_artifact("Baseline CU", baseline_artifact, baseline_dir)
    orchestration_events.append(
        orchestration_event(
            loop_id=loop_id,
            iteration=0,
            phase="baseline_cu",
            agent="cu",
            event_type="cu.run_completed",
            output_artifacts={"cu_output": str(baseline_dir)},
            metrics=cu_metrics(baseline_artifact),
            summary="CU published the initial silver contract.",
        )
    )

    with tui.phase("Build Query Tasks", "Load external fixtures and generate synthetic LLM bootstrap probes."):
        query_tasks, bootstrap_report = build_query_tasks(args, baseline_dir=baseline_dir, loop_id=loop_id)
    write_query_tasks_jsonl(query_tasks_path, query_tasks)
    write_json(query_bootstrap_path, bootstrap_report_to_dict(bootstrap_report, query_tasks))
    tui.show_query_tasks(query_tasks, bootstrap_report, query_tasks_path)
    orchestration_events.append(
        orchestration_event(
            loop_id=loop_id,
            iteration=0,
            phase="query_bootstrap",
            agent="orchestrator",
            event_type="query_tasks.created",
            input_artifacts={"cu_output": str(baseline_dir)},
            output_artifacts={"query_tasks": str(query_tasks_path), "query_bootstrap": str(query_bootstrap_path)},
            metrics={"task_count": len(query_tasks), "source_counts": query_source_counts(query_tasks)},
            decision={
                "bootstrap": bootstrap_report_to_dict(bootstrap_report, query_tasks),
                "generated_task_ids": [task["task_id"] for task in query_tasks if task.get("generation_id")],
                "adjusted_for": first_adjusted_for(query_tasks),
            },
            summary="The orchestrator prepared downstream primary queries for QU replay.",
        )
    )

    with tui.phase("Baseline QU Replay", f"Answer {len(query_tasks)} query tasks against the baseline corpus."):
        baseline_answers = answer_tasks_with_qu(
            args,
            baseline_dir,
            query_tasks,
            limit=args.limit,
            output_dir=baseline_answers_dir,
        )
    if baseline_answers:
        write_answer_artifact(baseline_answer_path, baseline_answers[0], write_replay=not args.no_replay)
    tui.show_answers("Baseline QU", baseline_answers, baseline_answers_dir)
    orchestration_events.extend(
        query_answer_events(
            loop_id=loop_id,
            iteration=0,
            phase="baseline_qu",
            corpus_dir=baseline_dir,
            answers_dir=baseline_answers_dir,
            answers=baseline_answers,
            summary="QU answered a bootstrap query against the baseline silver contract.",
        )
    )

    with tui.phase("Emit QU Feedback", f"Write QU-to-CU column requests to {feedback_path}"):
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        feedback_path.write_text("", encoding="utf-8")
        for answer in baseline_answers:
            append_feedback(feedback_path, answer, corpus_path=baseline_dir)
    tui.show_feedback(feedback_path, baseline_answers)
    orchestration_events.append(
        orchestration_event(
            loop_id=loop_id,
            iteration=0,
            phase="feedback",
            agent="qu",
            event_type="qu.feedback_emitted",
            input_artifacts={
                "query_tasks": str(query_tasks_path),
                "answers": str(baseline_answers_dir),
                "cu_output": str(baseline_dir),
            },
            output_artifacts={"feedback_jsonl": str(feedback_path)},
            metrics={
                "feedback_record_count": jsonl_count(feedback_path),
                "column_request_count": sum(len(requested_field_names(answer)) for answer in baseline_answers),
                "query_task_count": len(query_tasks),
            },
            decision={"requested_fields": sorted(set(field for answer in baseline_answers for field in requested_field_names(answer)))},
            summary="QU emitted CU feedback from schema gaps and answerability judgement across query tasks.",
        )
    )

    with tui.phase("Refined CU", f"Re-induce schema with QU feedback into {refined_dir}"):
        refined_artifact = run_cu(args, refined_dir, feedback_input=feedback_path)
    tui.show_cu_artifact("Refined CU", refined_artifact, refined_dir)
    orchestration_events.append(
        orchestration_event(
            loop_id=loop_id,
            iteration=1,
            phase="refined_cu",
            agent="cu",
            event_type="cu.run_completed",
            input_artifacts={"feedback_jsonl": str(feedback_path), "query_tasks": str(query_tasks_path)},
            output_artifacts={"cu_output": str(refined_dir)},
            metrics=cu_metrics(refined_artifact),
            decision=feedback_decision(refined_dir),
            summary="CU re-induced the silver contract with QU feedback from the query distribution.",
        )
    )
    with tui.phase("Refined QU Replay", f"Re-run {len(query_tasks)} query tasks against the refined corpus."):
        refined_answers = answer_tasks_with_qu(
            args,
            refined_dir,
            query_tasks,
            limit=args.limit,
            output_dir=refined_answers_dir,
        )
    if refined_answers:
        write_answer_artifact(refined_answer_path, refined_answers[0], write_replay=not args.no_replay)
    tui.show_answers("Refined QU", refined_answers, refined_answers_dir)
    orchestration_events.extend(
        query_answer_events(
            loop_id=loop_id,
            iteration=1,
            phase="refined_qu",
            corpus_dir=refined_dir,
            answers_dir=refined_answers_dir,
            answers=refined_answers,
            summary="QU re-ran a bootstrap query against the refined silver contract.",
        )
    )

    summary = build_summary(
        query_tasks=query_tasks,
        baseline_dir=baseline_dir,
        refined_dir=refined_dir,
        feedback_path=feedback_path,
        query_tasks_path=query_tasks_path,
        query_bootstrap_path=query_bootstrap_path,
        baseline_answers=baseline_answers,
        refined_answers=refined_answers,
    )
    summary["artifacts"]["orchestration_trace"] = str(orchestration_trace_path)
    orchestration_events.append(
        orchestration_event(
            loop_id=loop_id,
            iteration=1,
            phase="loop_evaluation",
            agent="orchestrator",
            event_type="loop.evaluated",
            input_artifacts={
                "query_tasks": str(query_tasks_path),
                "baseline_answers": str(baseline_answers_dir),
                "refined_answers": str(refined_answers_dir),
                "feedback_jsonl": str(feedback_path),
            },
            output_artifacts={"summary": str(summary_path), "orchestration_trace": str(orchestration_trace_path)},
            metrics=summary["loop_result"],
            decision={
                "schema_delta": summary["schema_delta"],
                "query_tasks": summary["query_tasks"],
                "before": summary["before"],
                "after": summary["after"],
            },
            summary="The loop evaluated schema refinement against the same downstream query distribution.",
        )
    )
    write_jsonl(orchestration_trace_path, orchestration_events)
    write_json(summary_path, summary)
    tui.show_loop_summary(summary)
    tui.finish()
    if args.no_tui or args.print_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def configure_llm_args(args: argparse.Namespace) -> None:
    load_env_file(args.env_file)
    args.llm_base_url = args.llm_base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
    args.llm_model = args.llm_model or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    args.llm_api_key = args.llm_api_key or os.environ.get("OPENAI_API_KEY")
    args.embedding_model = args.embedding_model or os.environ.get("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    args.judge_base_url = args.judge_base_url or os.environ.get("JUDGE_OPENAI_BASE_URL") or args.llm_base_url
    args.judge_model = args.judge_model or os.environ.get("JUDGE_OPENAI_MODEL")
    args.judge_api_key = args.judge_api_key or os.environ.get("JUDGE_OPENAI_API_KEY") or args.llm_api_key
    args.judge_timeout_seconds = args.judge_timeout_seconds or args.llm_timeout_seconds


def run_cu(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    feedback_input: Path | None = None,
) -> Any:
    return run_content_understanding(
        args.input,
        output_dir,
        source_sql=args.source_sql,
        schema_inducer=build_cu_schema_inducer(args),
        field_extractor=build_cu_field_extractor(args),
        feedback_input=feedback_input,
        max_errors=args.max_errors,
        max_budget_usd=args.max_budget_usd,
        max_timeout_seconds=args.max_timeout_seconds,
    )


def build_cu_schema_inducer(args: argparse.Namespace) -> LLMSchemaInducer:
    fallback = HeuristicSchemaInducer()
    if args.cu_backend == "openai":
        return LLMSchemaInducer(build_cu_openai_client(args), fallback=fallback)
    if args.cu_backend == "openai-compatible":
        return LLMSchemaInducer(build_cu_openai_compatible_client(args), fallback=fallback)
    raise ValueError(f"Unsupported CU backend: {args.cu_backend}")


def build_cu_field_extractor(args: argparse.Namespace) -> LLMFieldExtractor:
    fallback = HeuristicFieldExtractor()
    if args.cu_backend == "openai":
        return LLMFieldExtractor(build_cu_openai_client(args), fallback=fallback)
    if args.cu_backend == "openai-compatible":
        return LLMFieldExtractor(build_cu_openai_compatible_client(args), fallback=fallback)
    raise ValueError(f"Unsupported CU backend: {args.cu_backend}")


def build_cu_openai_client(args: argparse.Namespace) -> CUOpenAIResponsesClient:
    return CUOpenAIResponsesClient(
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        timeout_seconds=args.llm_timeout_seconds,
    )


def build_cu_openai_compatible_client(args: argparse.Namespace) -> CUOpenAICompatibleChatClient:
    return CUOpenAICompatibleChatClient(
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        timeout_seconds=args.llm_timeout_seconds,
    )


def build_query_tasks(
    args: argparse.Namespace,
    *,
    baseline_dir: Path,
    loop_id: str,
) -> tuple[list[dict[str, Any]], QueryBootstrapReport]:
    catalog = read_json(baseline_dir / "silver_schema_catalog.json")
    manifest = read_json(baseline_dir / "manifest.json")
    adjusted_for = adjusted_for_cu_schema(
        loop_id=loop_id,
        iteration=0,
        cu_output=baseline_dir,
        catalog=catalog,
        manifest=manifest,
    )
    tasks: list[dict[str, Any]] = []
    for path in args.query_tasks:
        tasks.extend(load_query_tasks(path, default_source_type="curated"))
    for path in args.production_query_log:
        tasks.extend(load_query_tasks(path, default_source_type="production_log"))
    if not args.skip_default_query and args.query and args.query.strip():
        task = normalize_query_task(
            {"query": args.query, "intent": "schema_gap", "source_type": "manual"},
            index=1,
            default_source_type="manual",
            default_label_status="unlabeled",
        )
        if task:
            tasks.append(task)

    bootstrap_count = max(0, args.bootstrap_query_count)
    bootstrap_report = QueryBootstrapReport(
        generator=args.bootstrap_queries,
        model=args.llm_model,
        requested_count=bootstrap_count,
    )
    if bootstrap_count > 0:
        generator = LLMDownstreamQueryGenerator(build_query_bootstrap_client(args))
        generated, bootstrap_report = generator.generate(
            catalog=catalog,
            manifest=manifest,
            existing_tasks=tasks,
            adjusted_for=adjusted_for,
            max_tasks=bootstrap_count,
            focus=args.bootstrap_focus,
        )
        tasks.extend(generated)

    tasks = dedupe_query_tasks(tasks)
    if not tasks:
        raise ValueError("No query tasks were provided or generated.")
    return tasks, bootstrap_report


def build_query_bootstrap_client(args: argparse.Namespace) -> JSONChatClient:
    if args.bootstrap_queries == "openai":
        return OpenAIResponsesClient(
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            timeout_seconds=args.llm_timeout_seconds,
        )
    if args.bootstrap_queries == "openai-compatible":
        return OpenAICompatibleChatClient(
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            timeout_seconds=args.llm_timeout_seconds,
        )
    raise ValueError(f"Unsupported query bootstrap mode: {args.bootstrap_queries}")


def answer_tasks_with_qu(
    args: argparse.Namespace,
    corpus_dir: Path,
    query_tasks: list[dict[str, Any]],
    *,
    limit: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    agent = build_qu_agent(args, corpus_dir=corpus_dir)
    answers: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for task in query_tasks:
        answer = agent.answer(str(task["query"]), limit=limit)
        answer["query_task"] = task
        answers.append(answer)
        write_answer_artifact(
            output_dir / f"{safe_file_stem(str(task['task_id']))}.json",
            answer,
            write_replay=not args.no_replay,
        )
    return answers


def build_qu_agent(args: argparse.Namespace, *, corpus_dir: Path) -> QueryUnderstandingAgent:
    embedding_client = build_embedding_client(args)
    embedding_cache = args.embedding_cache
    if embedding_cache is None and embedding_client is not None:
        embedding_cache = corpus_dir / ".qu_agent_rlm" / f"{args.embedding_model}.json"
    corpus = SilverCorpus.from_dir(
        corpus_dir,
        embedding_client=embedding_client,
        embedding_cache_path=embedding_cache,
    )
    qu_llm = build_qu_llm_client(args)
    retrieval_subagent = AgenticRetrievalSubAgent(
        controller=build_qu_llm_client(args) if args.retrieval_mode == "agentic" else None,
        reranker=build_qu_llm_client(args),
        policy=SearchExecutionPolicy(
            min_calls=max(1, args.min_search_calls),
            max_iterations=max(0, args.max_search_iterations),
            query_diversity_threshold=max(0.0, min(float(args.query_diversity_threshold), 1.0)),
        ),
    )
    return QueryUnderstandingAgent(
        corpus,
        planner=LLMQueryPlanner(qu_llm, fallback=HeuristicQueryPlanner()),
        retrieval_mode=args.retrieval_mode,
        retrieval_subagent=retrieval_subagent,
        answer_judge=LLMAnswerJudge(build_qu_judge_client(args), fallback=HeuristicAnswerJudge()),
        max_errors=args.max_errors,
        max_budget_usd=args.max_budget_usd,
        max_timeout_seconds=args.max_timeout_seconds,
    )


def build_embedding_client(args: argparse.Namespace) -> OpenAIEmbeddingClient | None:
    if args.retrieval_mode not in {"embedding", "hybrid", "agentic"}:
        return None
    return OpenAIEmbeddingClient(
        model=args.embedding_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        timeout_seconds=args.llm_timeout_seconds,
    )


def build_qu_llm_client(args: argparse.Namespace) -> JSONChatClient:
    if args.qu_backend == "openai":
        return OpenAIResponsesClient(
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            timeout_seconds=args.llm_timeout_seconds,
        )
    if args.qu_backend == "openai-compatible":
        return OpenAICompatibleChatClient(
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            timeout_seconds=args.llm_timeout_seconds,
        )
    raise ValueError(f"Unsupported QU backend: {args.qu_backend}")


def build_qu_judge_client(args: argparse.Namespace) -> JSONChatClient:
    if args.qu_backend == "openai":
        return OpenAIResponsesClient(
            model=args.judge_model or args.llm_model,
            base_url=args.judge_base_url,
            api_key=args.judge_api_key,
            timeout_seconds=args.judge_timeout_seconds,
        )
    if args.qu_backend == "openai-compatible":
        return OpenAICompatibleChatClient(
            model=args.judge_model or args.llm_model,
            base_url=args.judge_base_url,
            api_key=args.judge_api_key,
            timeout_seconds=args.judge_timeout_seconds,
        )
    raise ValueError(f"Unsupported QU backend: {args.qu_backend}")


def build_summary(
    *,
    query_tasks: list[dict[str, Any]],
    baseline_dir: Path,
    refined_dir: Path,
    feedback_path: Path,
    query_tasks_path: Path,
    query_bootstrap_path: Path,
    baseline_answers: list[dict[str, Any]],
    refined_answers: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_catalog = read_json(baseline_dir / "silver_schema_catalog.json")
    refined_catalog = read_json(refined_dir / "silver_schema_catalog.json")
    feedback_report = read_json(refined_dir / "feedback_report.json")
    baseline_fields = field_names(baseline_catalog)
    refined_fields = field_names(refined_catalog)
    before = [compact_answer(answer) for answer in baseline_answers]
    after = [compact_answer(answer) for answer in refined_answers]
    before_success = sum(1 for item in before if item.get("judge_success"))
    after_success = sum(1 for item in after if item.get("judge_success"))
    before_filters = sum(1 for item in before if item.get("filters"))
    after_filters = sum(1 for item in after if item.get("filters"))
    return {
        "queries": [task["query"] for task in query_tasks],
        "query_tasks": {
            "count": len(query_tasks),
            "by_source_type": query_source_counts(query_tasks),
            "tasks": [compact_query_task(task) for task in query_tasks],
        },
        "artifacts": {
            "baseline_cu": str(baseline_dir),
            "query_tasks": str(query_tasks_path),
            "query_bootstrap": str(query_bootstrap_path),
            "feedback_jsonl": str(feedback_path),
            "refined_cu": str(refined_dir),
        },
        "before": before,
        "feedback": feedback_report["feedback"],
        "schema_delta": {
            "baseline_field_count": len(baseline_fields),
            "refined_field_count": len(refined_fields),
            "added_fields": sorted(refined_fields - baseline_fields),
            "requested_fields": feedback_report["requested_fields"],
        },
        "after": after,
        "loop_result": {
            "task_count": len(query_tasks),
            "feedback_record_count": jsonl_count(feedback_path),
            "feedback_promoted": any(item.get("promoted_to_silver") for item in feedback_report["requested_fields"]),
            "baseline_filter_count": before_filters,
            "refined_filter_count": after_filters,
            "query_uses_refined_filter_count": after_filters,
            "judge_success_delta": after_success - before_success,
            "column_request_count_delta": sum(len(item["column_requests"]) for item in after)
            - sum(len(item["column_requests"]) for item in before),
        },
    }


def compact_answer(answer: dict[str, Any]) -> dict[str, Any]:
    task = answer.get("query_task") if isinstance(answer.get("query_task"), dict) else {}
    judgement = answer.get("judgement", {})
    return {
        "task_id": task.get("task_id"),
        "query": answer.get("query"),
        "source_type": task.get("source_type"),
        "generation_id": task.get("generation_id"),
        "operation": answer.get("plan", {}).get("operation"),
        "filters": answer.get("plan", {}).get("filters", {}),
        "record_ids": call_ids(answer),
        "column_requests": requested_field_names(answer),
        "judge_success": judgement.get("success"),
        "needs_cu_feedback": judgement.get("needs_cu_feedback"),
        "failure_modes": judgement.get("failure_modes", []),
    }


def query_answer_events(
    *,
    loop_id: str,
    iteration: int,
    phase: str,
    corpus_dir: Path,
    answers_dir: Path,
    answers: list[dict[str, Any]],
    summary: str,
) -> list[dict[str, Any]]:
    events = []
    for answer in answers:
        task = answer.get("query_task", {})
        task_id = str(task.get("task_id"))
        events.append(
            orchestration_event(
                loop_id=loop_id,
                iteration=iteration,
                phase=phase,
                agent="qu",
                event_type="qu.query_task_completed",
                input_artifacts={"cu_output": str(corpus_dir)},
                output_artifacts={"answer": str(answers_dir / f"{safe_file_stem(task_id)}.json")},
                metrics=qu_metrics(answer),
                decision={"query_task": compact_query_task(task), **qu_decision(answer)},
                summary=summary,
            )
        )
    return events


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_answer_artifact(path: Path, answer: dict[str, Any], *, write_replay: bool) -> None:
    write_json(path, answer)
    if write_replay:
        write_json(path.with_suffix(".replay.json"), redact_for_replay(answer))


def field_names(catalog: dict[str, Any]) -> set[str]:
    return {field["name"] for field in catalog.get("fields", [])}


def call_ids(answer: dict[str, Any]) -> list[str]:
    return [record["call_id"] for record in answer.get("records", [])]


def requested_field_names(answer: dict[str, Any]) -> list[str]:
    return [request["field_name"] for request in answer.get("column_requests", [])]


def orchestration_event(
    *,
    loop_id: str,
    iteration: int,
    phase: str,
    agent: str,
    event_type: str,
    summary: str,
    input_artifacts: dict[str, str] | None = None,
    output_artifacts: dict[str, str] | None = None,
    metrics: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "event_id": f"evt-{uuid4().hex[:12]}",
        "loop_id": loop_id,
        "iteration": iteration,
        "phase": phase,
        "agent": agent,
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_artifacts": input_artifacts or {},
        "output_artifacts": output_artifacts or {},
        "metrics": metrics or {},
        "decision": decision or {},
        "summary": summary,
    }


def cu_metrics(artifact: Any) -> dict[str, Any]:
    usage = artifact.manifest.get("usage_summary", {})
    return {
        "record_count": artifact.manifest.get("record_count"),
        "chunk_count": artifact.manifest.get("chunk_count"),
        "schema_field_count": len(artifact.silver_schema_catalog.get("fields", [])),
        "trace_event_count": len(artifact.trace),
        "feedback_request_count": artifact.feedback_report.get("feedback", {}).get("request_count", 0),
        "validation_error_count": sum(artifact.feedback_report.get("validation_error_counts", {}).values()),
        "llm_call_count": usage.get("total_calls", 0),
        "llm_input_tokens": usage.get("input_tokens", 0),
        "llm_output_tokens": usage.get("output_tokens", 0),
        "llm_total_tokens": usage.get("total_tokens", 0),
        "llm_total_cost_usd": usage.get("total_cost_usd", 0.0),
        "llm_cost_basis": usage.get("pricing", {}).get("source", "unpriced"),
    }


def qu_metrics(answer: dict[str, Any]) -> dict[str, Any]:
    judgement = answer.get("judgement", {})
    diagnostics = answer.get("search_diagnostics", {})
    usage = answer.get("usage_summary", {})
    return {
        "operation": answer.get("plan", {}).get("operation"),
        "record_count": len(answer.get("records", [])),
        "evidence_count": len(answer.get("evidence", [])),
        "column_request_count": len(answer.get("column_requests", [])),
        "search_call_count": len(diagnostics.get("calls", [])),
        "search_failure_count": len(diagnostics.get("failures", [])),
        "judge_success": judgement.get("success"),
        "judge_confidence": judgement.get("confidence"),
        "needs_cu_feedback": judgement.get("needs_cu_feedback"),
        "llm_call_count": usage.get("total_calls", 0),
        "llm_input_tokens": usage.get("input_tokens", 0),
        "llm_output_tokens": usage.get("output_tokens", 0),
        "llm_total_tokens": usage.get("total_tokens", 0),
        "llm_total_cost_usd": usage.get("total_cost_usd", 0.0),
        "llm_cost_basis": usage.get("pricing", {}).get("source", "unpriced"),
    }


def qu_decision(answer: dict[str, Any]) -> dict[str, Any]:
    plan = answer.get("plan", {})
    judgement = answer.get("judgement", {})
    return {
        "operation": plan.get("operation"),
        "filters": plan.get("filters", {}),
        "requested_fields": requested_field_names(answer),
        "failure_modes": judgement.get("failure_modes", []),
        "rationale": judgement.get("rationale", ""),
    }


def feedback_decision(refined_dir: Path) -> dict[str, Any]:
    report = read_json(refined_dir / "feedback_report.json")
    return {
        "requested_fields": [
            {
                "field_name": item.get("field_name"),
                "accepted_into_schema": item.get("accepted_into_schema"),
                "promoted_to_silver": item.get("promoted_to_silver"),
            }
            for item in report.get("requested_fields", [])
        ]
    }


def bootstrap_report_to_dict(report: QueryBootstrapReport, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "generation_id": report.generation_id,
        "generator": report.generator,
        "model": report.model,
        "prompt": report.prompt,
        "coverage_notes": report.coverage_notes,
        "requested_count": report.requested_count,
        "generated_count": report.generated_count,
        "fallback_reason": report.fallback_reason,
        "source_counts": query_source_counts(tasks),
        "tasks": [compact_query_task(task) for task in tasks],
        "promotion_policy": "synthetic queries are bootstrap probes only; external hand-labeled fixtures are required for promotion gates.",
    }


def first_adjusted_for(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    for task in tasks:
        value = task.get("adjusted_for")
        if isinstance(value, dict):
            return value
    return {}


def write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def safe_file_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())[:120]
    return stem or "query_task"


if __name__ == "__main__":
    raise SystemExit(main())
