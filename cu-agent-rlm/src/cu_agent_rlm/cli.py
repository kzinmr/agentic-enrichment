from __future__ import annotations

import argparse
import os
from pathlib import Path

from .env import default_env_file, load_env_file
from .extraction import FieldExtractor, HeuristicFieldExtractor, LLMFieldExtractor
from .llm import (
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    OpenAICompatibleChatClient,
    OpenAIResponsesClient,
)
import sys

from .pipeline import run_content_understanding
from .prompt_repair import write_prompt_repair_requests
from .schema import HeuristicSchemaInducer, LLMSchemaInducer, SchemaInducer
from .usage import budget_unpriced_warning


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an RLM-style silver contract from Databricks call_records exports."
    )
    parser.add_argument("--input", type=Path, default=Path("../cu-agent/data/sample_calls.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("output/demo"))
    parser.add_argument("--source-sql", type=Path, default=Path("../call_records.sql"))
    parser.add_argument(
        "--feedback-input",
        type=Path,
        default=None,
        help="Optional QU feedback JSONL from qu-agent-rlm --feedback-output.",
    )
    parser.add_argument(
        "--prompt-repair-output",
        type=Path,
        default=None,
        help="Optional JSONL path for prompt repair signals. This only records signals; it never changes prompts.",
    )
    parser.add_argument("--max-chunk-chars", type=int, default=900)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=default_env_file(),
        help="Optional .env file for OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL.",
    )
    parser.add_argument(
        "--schema-inducer",
        choices=("openai", "openai-compatible"),
        default="openai",
        help="LLM schema induction backend. Heuristic induction is retained only as a reliability fallback.",
    )
    parser.add_argument("--max-schema-fields", type=int, default=10)
    parser.add_argument("--max-schema-values", type=int, default=10)
    parser.add_argument(
        "--extractor",
        choices=("openai", "openai-compatible"),
        default="openai",
        help="LLM field extraction backend. Heuristic extraction is retained only as a reliability fallback.",
    )
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-timeout-seconds", type=int, default=120)
    parser.add_argument("--max-errors", type=int, default=3)
    parser.add_argument("--max-budget-usd", type=float, default=None)
    parser.add_argument("--max-timeout-seconds", type=float, default=None)
    parser.add_argument(
        "--batch-max-concurrent",
        type=int,
        default=8,
        help=(
            "Maximum number of per-call field extractions to run concurrently. "
            "1 forces sequential extraction; results are identical regardless of this value."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    load_env_file(args.env_file)
    args.llm_base_url = args.llm_base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
    args.llm_model = args.llm_model or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    args.llm_api_key = args.llm_api_key or os.environ.get("OPENAI_API_KEY")
    return args


def main() -> int:
    args = parse_args()
    budget_warning = budget_unpriced_warning(args.max_budget_usd)
    if budget_warning:
        print(f"WARNING: {budget_warning}", file=sys.stderr)
    source_sql = args.source_sql
    if source_sql and not source_sql.exists():
        raise FileNotFoundError(f"Source SQL file not found: {source_sql}")
    artifact = run_content_understanding(
        args.input,
        args.output,
        source_sql=source_sql,
        max_chunk_chars=args.max_chunk_chars,
        schema_inducer=build_schema_inducer(args),
        field_extractor=build_extractor(args),
        feedback_input=args.feedback_input,
        max_errors=args.max_errors,
        max_budget_usd=args.max_budget_usd,
        max_timeout_seconds=args.max_timeout_seconds,
        batch_max_concurrent=args.batch_max_concurrent,
    )
    promoted_count = len(artifact.silver_schema_catalog["fields"])
    print(f"Wrote CU RLM artifacts to {args.output}")
    print(
        f"Records: {artifact.manifest['record_count']} | Chunks: {artifact.manifest['chunk_count']} | "
        f"Promoted fields: {promoted_count}"
    )
    print(
        "Key artifacts: manifest.json, chunks.jsonl, silver_schema_catalog.json, "
        "silver_calls.jsonl, feedback_report.json, rlm_trace.jsonl"
    )
    if args.prompt_repair_output:
        write_prompt_repair_requests(args.prompt_repair_output, artifact)
    return 0


def build_schema_inducer(args: argparse.Namespace) -> SchemaInducer:
    heuristic = HeuristicSchemaInducer(
        max_fields=args.max_schema_fields,
        max_values_per_field=args.max_schema_values,
    )
    if args.schema_inducer == "openai":
        return LLMSchemaInducer(
            build_openai_client(args),
            fallback=heuristic,
            max_fields=args.max_schema_fields,
            max_values_per_field=args.max_schema_values,
        )
    if args.schema_inducer == "openai-compatible":
        return LLMSchemaInducer(
            build_openai_compatible_client(args),
            fallback=heuristic,
            max_fields=args.max_schema_fields,
            max_values_per_field=args.max_schema_values,
        )
    raise ValueError(f"Unsupported schema inducer: {args.schema_inducer}")


def build_extractor(args: argparse.Namespace) -> FieldExtractor:
    heuristic = HeuristicFieldExtractor()
    if args.extractor == "openai":
        return LLMFieldExtractor(
            build_openai_client(args),
            fallback=heuristic,
        )
    if args.extractor == "openai-compatible":
        return LLMFieldExtractor(build_openai_compatible_client(args), fallback=heuristic)
    raise ValueError(f"Unsupported extractor: {args.extractor}")


def build_openai_client(args: argparse.Namespace) -> OpenAIResponsesClient:
    return OpenAIResponsesClient(
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        timeout_seconds=args.llm_timeout_seconds,
    )


def build_openai_compatible_client(args: argparse.Namespace) -> OpenAICompatibleChatClient:
    return OpenAICompatibleChatClient(
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        timeout_seconds=args.llm_timeout_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
