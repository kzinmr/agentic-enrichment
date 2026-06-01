from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .agent import QueryUnderstandingAgent
from .corpus import SilverCorpus
from .env import default_env_file, load_env_file
from .eval import run_eval
from .judge import AnswerJudge, HeuristicAnswerJudge, LLMAnswerJudge
from .llm import (
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    JSONChatClient,
    OpenAICompatibleChatClient,
    OpenAIResponsesClient,
)
import sys

from .planner import HeuristicQueryPlanner, LLMQueryPlanner, QueryPlanner
from .prompt_repair import append_prompt_repair_requests
from .retrieval import DEFAULT_EMBEDDING_MODEL, OpenAIEmbeddingClient
from .retrieval_agent import AgenticRetrievalSubAgent, RetrievalSubAgent, SearchExecutionPolicy
from .usage import budget_unpriced_warning


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query CU silver artifacts with an RLM-style local agent.")
    parser.add_argument("--corpus", type=Path, default=Path("../cu-agent-rlm/output/demo"))
    parser.add_argument("--query", default=None)
    parser.add_argument("--eval-tasks", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--feedback-output",
        type=Path,
        default=None,
        help="Optional JSONL path for QU-to-CU column requests discovered while answering a query.",
    )
    parser.add_argument(
        "--prompt-repair-output",
        type=Path,
        default=None,
        help="Optional JSONL path for prompt repair signals. This only records signals; it never changes prompts.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=("lexical", "bm25", "embedding", "hybrid", "agentic"),
        default="bm25",
        help="Default implementation for abstract search_chunks steps. Concrete BM25/embedding tools remain agent-callable.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=default_env_file(),
        help="Optional .env file for OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL.",
    )
    parser.add_argument(
        "--planner",
        choices=("openai", "openai-compatible"),
        default="openai",
        help="LLM query planner backend. Heuristic planning is retained only as a reliability fallback.",
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
        help="Deprecated low-level retrieval controller selector. Prefer --retrieval-subagent.",
    )
    parser.add_argument(
        "--retrieval-subagent",
        choices=("default", "none", "openai", "openai-compatible"),
        default="default",
        help=(
            "Specialized retrieval subagent backend. default uses OpenAI for --retrieval-mode agentic; "
            "openai-compatible can point to SID-1-style search models via --llm-base-url and --llm-model."
        ),
    )
    parser.add_argument(
        "--reranker",
        choices=("none", "openai", "openai-compatible"),
        default="none",
        help="Optional LLM reranker for final BM25/embedding candidate ordering.",
    )
    parser.add_argument(
        "--judge",
        choices=("openai", "openai-compatible"),
        default="openai",
        help="LLM answerability/evidence judge backend. Heuristic judging is retained only as a reliability fallback.",
    )
    parser.add_argument("--min-search-calls", type=int, default=1)
    parser.add_argument("--max-search-iterations", type=int, default=0)
    parser.add_argument("--query-diversity-threshold", type=float, default=0.8)
    parser.add_argument("--max-errors", type=int, default=3)
    parser.add_argument("--max-budget-usd", type=float, default=None)
    parser.add_argument("--max-timeout-seconds", type=float, default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-cache", type=Path, default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    load_env_file(args.env_file)
    args.llm_base_url = args.llm_base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
    args.llm_model = args.llm_model or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    args.llm_api_key = args.llm_api_key or os.environ.get("OPENAI_API_KEY")
    args.embedding_model = args.embedding_model or os.environ.get("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    args.judge_base_url = args.judge_base_url or os.environ.get("JUDGE_OPENAI_BASE_URL") or args.llm_base_url
    args.judge_model = args.judge_model or os.environ.get("JUDGE_OPENAI_MODEL")
    args.judge_api_key = args.judge_api_key or os.environ.get("JUDGE_OPENAI_API_KEY") or args.llm_api_key
    args.judge_timeout_seconds = args.judge_timeout_seconds or args.llm_timeout_seconds
    return args


def main() -> int:
    args = parse_args()
    budget_warning = budget_unpriced_warning(args.max_budget_usd)
    if budget_warning:
        print(f"WARNING: {budget_warning}", file=sys.stderr)
    embedding_client = build_embedding_client(args)
    embedding_cache = args.embedding_cache
    if embedding_cache is None and embedding_client is not None:
        embedding_cache = args.corpus / ".qu_agent_rlm" / f"{args.embedding_model}.json"
    corpus = SilverCorpus.from_dir(
        args.corpus,
        embedding_client=embedding_client,
        embedding_cache_path=embedding_cache,
    )
    planner = build_planner(args)
    reranker = build_optional_agent_llm_client(args, mode=agent_mode(args, args.reranker))
    retrieval_subagent = build_retrieval_subagent(args, reranker=reranker)
    answer_judge = build_answer_judge(args)
    agent = QueryUnderstandingAgent(
        corpus,
        planner=planner,
        retrieval_mode=args.retrieval_mode,
        retrieval_subagent=retrieval_subagent,
        answer_judge=answer_judge,
        max_errors=args.max_errors,
        max_budget_usd=args.max_budget_usd,
        max_timeout_seconds=args.max_timeout_seconds,
    )
    if args.eval_tasks:
        report = run_eval(agent, args.eval_tasks)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    query = args.query or "Which calls mention security review or access controls?"
    result = agent.answer(query, limit=args.limit)
    if args.feedback_output:
        append_feedback(args.feedback_output, result, corpus_path=args.corpus)
    if args.prompt_repair_output:
        append_prompt_repair_requests(args.prompt_repair_output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_planner(args: argparse.Namespace) -> QueryPlanner:
    heuristic = HeuristicQueryPlanner()
    if args.planner == "openai":
        return LLMQueryPlanner(build_openai_client(args), fallback=heuristic)
    if args.planner == "openai-compatible":
        return LLMQueryPlanner(build_openai_compatible_client(args), fallback=heuristic)
    raise ValueError(f"Unsupported planner: {args.planner}")


def agent_mode(args: argparse.Namespace, requested: str) -> str:
    return requested


def retrieval_subagent_mode(args: argparse.Namespace) -> str:
    if args.retrieval_subagent == "default":
        if args.search_controller != "none":
            return args.search_controller
        if args.retrieval_mode == "agentic":
            return "openai"
        return "none"
    return args.retrieval_subagent


def build_retrieval_subagent(args: argparse.Namespace, *, reranker: JSONChatClient | None) -> RetrievalSubAgent:
    controller = build_optional_agent_llm_client(args, mode=retrieval_subagent_mode(args))
    return AgenticRetrievalSubAgent(
        controller=controller,
        reranker=reranker,
        policy=build_search_policy(args),
    )


def build_search_policy(args: argparse.Namespace) -> SearchExecutionPolicy:
    min_calls = max(1, args.min_search_calls)
    max_iterations = max(0, args.max_search_iterations)
    if args.retrieval_mode == "agentic":
        min_calls = max(min_calls, 2)
        max_iterations = max(max_iterations, 2)
    return SearchExecutionPolicy(
        min_calls=min_calls,
        max_iterations=max_iterations,
        query_diversity_threshold=max(0.0, min(float(args.query_diversity_threshold), 1.0)),
    )


def build_optional_agent_llm_client(args: argparse.Namespace, *, mode: str) -> JSONChatClient | None:
    if mode == "none":
        return None
    if mode == "openai":
        return build_openai_client(args)
    if mode == "openai-compatible":
        return build_openai_compatible_client(args)
    raise ValueError(f"Unsupported agent LLM mode: {mode}")


def build_answer_judge(args: argparse.Namespace) -> AnswerJudge | None:
    heuristic = HeuristicAnswerJudge()
    if args.judge == "openai":
        return LLMAnswerJudge(build_openai_judge_client(args), fallback=heuristic)
    if args.judge == "openai-compatible":
        return LLMAnswerJudge(build_openai_compatible_judge_client(args), fallback=heuristic)
    raise ValueError(f"Unsupported answer judge mode: {args.judge}")


def build_embedding_client(args: argparse.Namespace) -> OpenAIEmbeddingClient | None:
    if args.retrieval_mode in {"embedding", "hybrid", "agentic"} and not args.llm_api_key:
        raise ValueError("OPENAI_API_KEY is required for embedding, hybrid, or agentic retrieval mode.")
    if not args.llm_api_key:
        return None
    return OpenAIEmbeddingClient(
        model=args.embedding_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        timeout_seconds=args.llm_timeout_seconds,
    )


def build_openai_client(args: argparse.Namespace) -> OpenAIResponsesClient:
    return OpenAIResponsesClient(
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        timeout_seconds=args.llm_timeout_seconds,
    )


def build_openai_judge_client(args: argparse.Namespace) -> OpenAIResponsesClient:
    return OpenAIResponsesClient(
        model=args.judge_model or args.llm_model,
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
        timeout_seconds=args.judge_timeout_seconds,
    )


def build_openai_compatible_client(args: argparse.Namespace) -> OpenAICompatibleChatClient:
    return OpenAICompatibleChatClient(
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        timeout_seconds=args.llm_timeout_seconds,
    )


def build_openai_compatible_judge_client(args: argparse.Namespace) -> OpenAICompatibleChatClient:
    return OpenAICompatibleChatClient(
        model=args.judge_model or args.llm_model,
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
        timeout_seconds=args.judge_timeout_seconds,
    )


def append_feedback(path: Path, result: dict[str, object], *, corpus_path: Path) -> None:
    requests = result.get("column_requests") or []
    judgement = result.get("judgement")
    needs_feedback = bool(requests)
    if isinstance(judgement, dict):
        needs_feedback = needs_feedback or bool(judgement.get("needs_cu_feedback")) or not bool(
            judgement.get("success", True)
        )
    if not needs_feedback:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": result.get("query"),
        "query_task": result.get("query_task"),
        "corpus": str(corpus_path),
        "plan": result.get("plan"),
        "column_requests": requests,
        "search_diagnostics": result.get("search_diagnostics"),
        "rerank": result.get("rerank"),
        "judgement": judgement,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
