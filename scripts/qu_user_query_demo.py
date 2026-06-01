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
sys.path.insert(0, str(REPO_ROOT / "qu-agent-rlm" / "src"))

from qu_agent_rlm.agent import QueryUnderstandingAgent
from qu_agent_rlm.cli import (
    agent_mode,
    append_feedback,
    build_answer_judge,
    build_embedding_client,
    build_optional_agent_llm_client,
    build_planner,
    build_retrieval_subagent,
)
from qu_agent_rlm.corpus import SilverCorpus
from qu_agent_rlm.env import default_env_file, load_env_file
from qu_agent_rlm.llm import DEFAULT_OPENAI_BASE_URL, DEFAULT_OPENAI_MODEL
from qu_agent_rlm.prompt_repair import append_prompt_repair_requests
from qu_agent_rlm.query_tasks import load_query_tasks, normalize_query_task, write_query_tasks_jsonl
from qu_agent_rlm.replay import redact_for_replay
from qu_agent_rlm.retrieval import DEFAULT_EMBEDDING_MODEL
from qu_agent_rlm.usage import budget_unpriced_warning


DEFAULT_USER_QUERIES = [
    "Break down calls by conversation topic.",
    "Which calls mention founder-led sales conversations and show evidence?",
    "Which calls mention pricing blockers or pricing objections?",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query a QU agent built from scripts/qu_cu_loop_demo.py artifacts.",
    )
    parser.add_argument("--loop-output-root", type=Path, default=REPO_ROOT / "output" / "qu_cu_loop_demo")
    parser.add_argument(
        "--stage",
        choices=("refined", "baseline"),
        default="refined",
        help="Which CU artifact from the loop to query. refined uses 03_cu_refined; baseline uses 01_cu_baseline.",
    )
    parser.add_argument("--corpus", type=Path, default=None, help="Override the CU artifact directory.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--query", action="append", default=[], help="User analysis query. Can be provided multiple times.")
    parser.add_argument(
        "--query-file",
        type=Path,
        action="append",
        default=[],
        help="Text, JSON, or JSONL file of user queries. Text files use one query per non-empty line.",
    )
    parser.add_argument("--interactive", action="store_true", help="Read user queries from stdin until an empty line.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--retrieval-mode",
        choices=("lexical", "bm25", "embedding", "hybrid", "agentic"),
        default="agentic",
    )
    parser.add_argument("--env-file", type=Path, default=default_env_file())
    parser.add_argument(
        "--planner",
        choices=("openai", "openai-compatible"),
        default="openai",
    )
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-timeout-seconds", type=int, default=60)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-timeout-seconds", type=int, default=None)
    parser.add_argument(
        "--search-controller",
        choices=("none", "openai", "openai-compatible"),
        default="none",
    )
    parser.add_argument(
        "--retrieval-subagent",
        choices=("default", "none", "openai", "openai-compatible"),
        default="default",
    )
    parser.add_argument(
        "--reranker",
        choices=("none", "openai", "openai-compatible"),
        default="openai",
    )
    parser.add_argument(
        "--judge",
        choices=("openai", "openai-compatible"),
        default="openai",
    )
    parser.add_argument("--min-search-calls", type=int, default=1)
    parser.add_argument("--max-search-iterations", type=int, default=0)
    parser.add_argument("--query-diversity-threshold", type=float, default=0.8)
    parser.add_argument("--max-errors", type=int, default=3)
    parser.add_argument("--max-budget-usd", type=float, default=None)
    parser.add_argument("--max-timeout-seconds", type=float, default=None)
    parser.add_argument("--max-plan-iterations", type=int, default=2)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-cache", type=Path, default=None)
    parser.add_argument(
        "--feedback-output",
        type=Path,
        default=None,
        help="Optional QU-to-CU feedback JSONL. Defaults to <output-dir>/user_feedback.jsonl.",
    )
    parser.add_argument(
        "--prompt-repair-output",
        type=Path,
        default=None,
        help="Optional prompt repair JSONL. Defaults to <output-dir>/prompt_repair_request.jsonl.",
    )
    parser.add_argument("--no-feedback-output", action="store_true")
    parser.add_argument("--no-prompt-repair-output", action="store_true")
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
    corpus_dir = resolve_corpus_dir(args)
    output_dir = args.output_dir or args.loop_output_root / "05_user_queries" / args.stage
    tui = DemoTUI(title="QU User Query Demo", enabled=not args.no_tui)
    tui.header(
        subtitle="Run user-facing analysis queries against a CU-QU loop artifact.",
        config={
            "loop_output_root": args.loop_output_root,
            "stage": args.stage,
            "corpus": corpus_dir,
            "output_dir": output_dir,
            "model": args.llm_model,
            "retrieval_mode": args.retrieval_mode,
            "reranker": args.reranker,
            "judge": args.judge,
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    answers_dir = output_dir / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)
    feedback_output = None if args.no_feedback_output else (args.feedback_output or output_dir / "user_feedback.jsonl")
    prompt_repair_output = (
        None if args.no_prompt_repair_output else (args.prompt_repair_output or output_dir / "prompt_repair_request.jsonl")
    )
    clear_optional_jsonl(feedback_output)
    clear_optional_jsonl(prompt_repair_output)

    with tui.phase("Prepare User Queries", "Collect CLI, file, or interactive query inputs."):
        queries = collect_queries(args)
        if not queries:
            queries = list(DEFAULT_USER_QUERIES)
        query_tasks = build_user_query_tasks(queries)
        write_query_tasks_jsonl(output_dir / "user_query_tasks.jsonl", query_tasks)
    tui.show_user_queries(query_tasks, output_dir / "user_query_tasks.jsonl")

    with tui.phase("Build QU Agent", "Load silver corpus, indexes, planner, retrieval subagent, reranker, and judge."):
        agent_context = build_agent_context(args, corpus_dir=corpus_dir, output_dir=output_dir)
        agent = build_agent(args, corpus_dir=corpus_dir)
    tui.show_user_context(agent_context)
    session_id = f"user-query-session-{uuid4().hex[:12]}"
    trace_events = [
        trace_event(
            session_id=session_id,
            event_type="user_query_session.started",
            agent="orchestrator",
            corpus_dir=corpus_dir,
            output_artifacts={"user_query_tasks": str(output_dir / "user_query_tasks.jsonl")},
            metrics={"query_count": len(query_tasks)},
            decision={"agent_context": agent_context},
            summary="Built a QU agent from CU-QU loop artifacts for user analysis queries.",
        )
    ]
    answers: list[dict[str, Any]] = []
    with tui.phase("Answer User Queries", f"Run {len(query_tasks)} queries through the LLM-first QU agent."):
        for task in query_tasks:
            answer = agent.answer(str(task["query"]), limit=args.limit)
            answer["query_task"] = task
            answer["agent_context"] = agent_context
            answer_path = answers_dir / f"{safe_file_stem(str(task['task_id']))}.json"
            write_json(answer_path, answer)
            if not args.no_replay:
                write_json(answer_path.with_suffix(".replay.json"), redact_for_replay(answer))
            answers.append(answer)
            if feedback_output:
                append_feedback(feedback_output, answer, corpus_path=corpus_dir)
            if prompt_repair_output:
                append_prompt_repair_requests(prompt_repair_output, answer)
            trace_events.append(
                trace_event(
                    session_id=session_id,
                    event_type="user_query.completed",
                    agent="qu",
                    corpus_dir=corpus_dir,
                    output_artifacts={"answer": str(answer_path)},
                    metrics=answer_metrics(answer),
                    decision={"query_task": compact_query_task(task), "plan": answer.get("plan", {})},
                    summary="QU answered a user analysis query against the selected loop artifact.",
                )
            )

    summary = build_summary(
        session_id=session_id,
        agent_context=agent_context,
        answers=answers,
        output_dir=output_dir,
        feedback_output=feedback_output,
        prompt_repair_output=prompt_repair_output,
    )
    write_json(output_dir / "user_query_summary.json", summary)
    write_jsonl(output_dir / "user_query_trace.jsonl", trace_events)
    tui.show_user_summary(summary)
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


def resolve_corpus_dir(args: argparse.Namespace) -> Path:
    corpus_dir = args.corpus
    if corpus_dir is None:
        corpus_dir = args.loop_output_root / ("03_cu_refined" if args.stage == "refined" else "01_cu_baseline")
    if not (corpus_dir / "silver_schema_catalog.json").exists():
        raise FileNotFoundError(
            f"Missing CU artifacts at {corpus_dir}. Run scripts/qu_cu_loop_demo.py first or pass --corpus."
        )
    return corpus_dir


def build_agent(args: argparse.Namespace, *, corpus_dir: Path) -> QueryUnderstandingAgent:
    embedding_client = build_embedding_client(args)
    embedding_cache = args.embedding_cache
    if embedding_cache is None and embedding_client is not None:
        embedding_cache = corpus_dir / ".qu_agent_rlm" / f"{args.embedding_model}.json"
    corpus = SilverCorpus.from_dir(
        corpus_dir,
        embedding_client=embedding_client,
        embedding_cache_path=embedding_cache,
    )
    planner = build_planner(args)
    reranker = build_optional_agent_llm_client(args, mode=agent_mode(args, args.reranker))
    retrieval_subagent = build_retrieval_subagent(args, reranker=reranker)
    answer_judge = build_answer_judge(args)
    return QueryUnderstandingAgent(
        corpus,
        planner=planner,
        retrieval_mode=args.retrieval_mode,
        retrieval_subagent=retrieval_subagent,
        answer_judge=answer_judge,
        max_errors=args.max_errors,
        max_budget_usd=args.max_budget_usd,
        max_timeout_seconds=args.max_timeout_seconds,
        max_plan_iterations=args.max_plan_iterations,
    )


def collect_queries(args: argparse.Namespace) -> list[str]:
    queries = [query.strip() for query in args.query if query and query.strip()]
    for path in args.query_file:
        queries.extend(load_queries_from_file(path))
    if args.interactive:
        print("Enter user analysis queries. Submit an empty line to finish.", file=sys.stderr)
        while True:
            try:
                value = input("> ").strip()
            except EOFError:
                break
            if not value:
                break
            queries.append(value)
    return dedupe_texts(queries)


def load_queries_from_file(path: Path) -> list[str]:
    if path.suffix.lower() in {".json", ".jsonl"}:
        return [str(task["query"]) for task in load_query_tasks(path, default_source_type="user_demo")]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_user_query_tasks(queries: list[str]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for index, query in enumerate(queries, start=1):
        task = normalize_query_task(
            {
                "query": query,
                "source_type": "user_demo",
                "label_status": "unlabeled",
                "split": "production_demo",
            },
            index=index,
            default_source_type="user_demo",
            default_label_status="unlabeled",
        )
        if task:
            tasks.append(task)
    return tasks


def build_agent_context(args: argparse.Namespace, *, corpus_dir: Path, output_dir: Path) -> dict[str, Any]:
    manifest = read_json_if_exists(corpus_dir / "manifest.json")
    catalog = read_json(corpus_dir / "silver_schema_catalog.json")
    loop_summary = read_json_if_exists(args.loop_output_root / "loop_summary.json")
    query_bootstrap = read_json_if_exists(args.loop_output_root / "02_qu_feedback" / "query_bootstrap.json")
    return {
        "loop_output_root": str(args.loop_output_root),
        "stage": args.stage,
        "corpus_dir": str(corpus_dir),
        "output_dir": str(output_dir),
        "schema_version": catalog.get("schema_version"),
        "schema_field_names": [field.get("name") for field in catalog.get("fields", [])],
        "record_count": manifest.get("record_count"),
        "chunk_count": manifest.get("chunk_count"),
        "loop_artifacts": {
            "summary": str(args.loop_output_root / "loop_summary.json"),
            "orchestration_trace": str(args.loop_output_root / "orchestration_trace.jsonl"),
            "query_tasks": str(args.loop_output_root / "02_qu_feedback" / "query_tasks.jsonl"),
            "query_bootstrap": str(args.loop_output_root / "02_qu_feedback" / "query_bootstrap.json"),
        },
        "loop_result": loop_summary.get("loop_result", {}),
        "query_bootstrap": {
            "generation_id": query_bootstrap.get("generation_id"),
            "generator": query_bootstrap.get("generator"),
            "model": query_bootstrap.get("model"),
            "prompt": query_bootstrap.get("prompt", {}),
            "source_counts": query_bootstrap.get("source_counts", {}),
        },
        "agent_config": {
            "planner": args.planner,
            "retrieval_mode": args.retrieval_mode,
            "retrieval_subagent": args.retrieval_subagent,
            "reranker": args.reranker,
            "judge": args.judge,
            "llm_model": args.llm_model,
            "judge_model": args.judge_model or args.llm_model,
            "embedding_model": args.embedding_model,
            "max_plan_iterations": args.max_plan_iterations,
        },
    }


def build_summary(
    *,
    session_id: str,
    agent_context: dict[str, Any],
    answers: list[dict[str, Any]],
    output_dir: Path,
    feedback_output: Path | None,
    prompt_repair_output: Path | None,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "agent_context": agent_context,
        "artifacts": {
            "output_dir": str(output_dir),
            "answers": str(output_dir / "answers"),
            "user_query_tasks": str(output_dir / "user_query_tasks.jsonl"),
            "user_query_trace": str(output_dir / "user_query_trace.jsonl"),
            "feedback_output": str(feedback_output) if feedback_output else None,
            "prompt_repair_output": str(prompt_repair_output) if prompt_repair_output else None,
        },
        "results": [compact_answer(answer) for answer in answers],
        "metrics": {
            "query_count": len(answers),
            "success_count": sum(1 for answer in answers if answer.get("judgement", {}).get("success")),
            "needs_cu_feedback_count": sum(
                1 for answer in answers if answer.get("judgement", {}).get("needs_cu_feedback")
            ),
            "column_request_count": sum(len(answer.get("column_requests", [])) for answer in answers),
        },
    }


def compact_answer(answer: dict[str, Any]) -> dict[str, Any]:
    judgement = answer.get("judgement", {})
    task = answer.get("query_task", {})
    return {
        "task_id": task.get("task_id"),
        "query": answer.get("query"),
        "operation": answer.get("plan", {}).get("operation"),
        "filters": answer.get("plan", {}).get("filters", {}),
        "group_by": answer.get("plan", {}).get("group_by"),
        "record_ids": [record.get("call_id") for record in answer.get("records", [])],
        "aggregation": answer.get("aggregation", {}),
        "evidence_count": len(answer.get("evidence", [])),
        "column_requests": [request.get("field_name") for request in answer.get("column_requests", [])],
        "judge_success": judgement.get("success"),
        "needs_cu_feedback": judgement.get("needs_cu_feedback"),
        "failure_modes": judgement.get("failure_modes", []),
    }


def compact_query_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "query": task.get("query"),
        "source_type": task.get("source_type"),
        "label_status": task.get("label_status"),
    }


def answer_metrics(answer: dict[str, Any]) -> dict[str, Any]:
    diagnostics = answer.get("search_diagnostics", {})
    judgement = answer.get("judgement", {})
    usage = answer.get("usage_summary", {})
    return {
        "operation": answer.get("plan", {}).get("operation"),
        "record_count": len(answer.get("records", [])),
        "evidence_count": len(answer.get("evidence", [])),
        "aggregation_group_count": len(answer.get("aggregation", {})),
        "column_request_count": len(answer.get("column_requests", [])),
        "search_call_count": len(diagnostics.get("calls", [])),
        "search_failure_count": len(diagnostics.get("failures", [])),
        "judge_success": judgement.get("success"),
        "needs_cu_feedback": judgement.get("needs_cu_feedback"),
        "llm_call_count": usage.get("total_calls", 0),
        "llm_input_tokens": usage.get("input_tokens", 0),
        "llm_output_tokens": usage.get("output_tokens", 0),
        "llm_total_tokens": usage.get("total_tokens", 0),
        "llm_total_cost_usd": usage.get("total_cost_usd", 0.0),
        "llm_cost_basis": usage.get("pricing", {}).get("source", "unpriced"),
    }


def trace_event(
    *,
    session_id: str,
    event_type: str,
    agent: str,
    corpus_dir: Path,
    summary: str,
    output_artifacts: dict[str, str] | None = None,
    metrics: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "event_id": f"evt-{uuid4().hex[:12]}",
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "event_type": event_type,
        "input_artifacts": {"corpus_dir": str(corpus_dir)},
        "output_artifacts": output_artifacts or {},
        "metrics": metrics or {},
        "decision": decision or {},
        "summary": summary,
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def clear_optional_jsonl(path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def safe_file_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())[:120]
    return stem or "user_query"


def dedupe_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = " ".join(value.lower().split())
        if key and key not in seen:
            seen.add(key)
            deduped.append(value)
    return deduped


if __name__ == "__main__":
    raise SystemExit(main())
