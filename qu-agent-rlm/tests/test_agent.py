from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

workspace = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(workspace / "cu-agent-rlm" / "src"))

from cu_agent_rlm.pipeline import run_content_understanding
from cu_agent_rlm.schema import StaticSchemaInducer
from qu_agent_rlm.agent import QueryUnderstandingAgent, SearchExecutionPolicy
from qu_agent_rlm.cli import append_feedback, parse_args
from qu_agent_rlm.corpus import SilverCorpus, ToolEvent
from qu_agent_rlm.eval import run_eval
from qu_agent_rlm.judge import LLMAnswerJudge
from qu_agent_rlm.llm import DEFAULT_OPENAI_MODEL
from qu_agent_rlm.planner import LLMQueryPlanner, QueryPlan, QueryToolStep
from qu_agent_rlm.query_tasks import (
    LLMDownstreamQueryGenerator,
    adjusted_for_cu_schema,
    build_heuristic_bootstrap_tasks,
    dedupe_query_tasks,
)
from qu_agent_rlm.replay import REDACTED, redact_for_replay
from qu_agent_rlm.retrieval import BM25Index
from qu_agent_rlm.schema_negotiation import evaluate_field_candidates
from qu_agent_rlm.usage import UsageSummary, budget_unpriced_warning


class QueryUnderstandingRLMTest(unittest.TestCase):
    def test_filters_aggregate_and_evaluates_cu_artifacts(self) -> None:
        sample = workspace / "data" / "sample_calls.jsonl"
        source_sql = next((p for p in [workspace / "call_records.sql"] if p.exists()), None)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            run_content_understanding(sample, output, source_sql=source_sql, schema_inducer=StaticSchemaInducer())
            agent = QueryUnderstandingAgent(SilverCorpus.from_dir(output))

            security = agent.answer("Which calls mention security review or access controls?")
            self.assertEqual(security["plan"]["filters"]["security_review_requested"], True)
            self.assertGreaterEqual(len(security["plan"]["steps"]), 4)
            self.assertIn("query_silver", [event["tool"] for event in security["trace"]])
            self.assertIn("bm25_search_chunks", [event["tool"] for event in security["trace"]])
            self.assertIn("review_schema_gaps", [event["tool"] for event in security["trace"]])
            self.assertTrue(security["records"])
            self.assertTrue(security["evidence"])

            pricing = agent.answer("Which accounts have pricing objections or pricing blocked deals?")
            self.assertEqual(pricing["plan"]["filters"]["renewal_risk"], "pricing_pushback")
            self.assertTrue({record["call_id"] for record in pricing["records"]}.issuperset({"call-002", "call-005", "call-006"}))

            aggregate = agent.answer("Count calls by product area.")
            self.assertEqual(aggregate["plan"]["operation"], "aggregate")
            self.assertEqual(aggregate["plan"]["group_by"], "product_area")
            self.assertIn("crm", aggregate["aggregation"])

            eval_report = run_eval(agent, output / "evaluation_tasks.json")
            self.assertEqual(eval_report["passed"], eval_report["task_count"])

    def test_agent_accepts_injected_planner(self) -> None:
        sample = workspace / "data" / "sample_calls.jsonl"
        source_sql = next((p for p in [workspace / "call_records.sql"] if p.exists()), None)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            run_content_understanding(sample, output, source_sql=source_sql, schema_inducer=StaticSchemaInducer())
            agent = QueryUnderstandingAgent(SilverCorpus.from_dir(output), planner=StaticPlanner())

            result = agent.answer("Find anything relevant.")

            self.assertEqual(result["plan"]["planner"], "static")
            self.assertEqual(result["plan"]["operation"], "filter")
            self.assertTrue(result["records"])

    def test_llm_planner_maps_json_to_valid_plan(self) -> None:
        sample = workspace / "data" / "sample_calls.jsonl"
        source_sql = next((p for p in [workspace / "call_records.sql"] if p.exists()), None)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            run_content_understanding(sample, output, source_sql=source_sql, schema_inducer=StaticSchemaInducer())
            corpus = SilverCorpus.from_dir(output)
            planner = LLMQueryPlanner(
                FakeJSONClient(
                    {
                        "operation": "aggregate",
                        "filters": {"security_review_requested": True},
                        "group_by": "product_area",
                        "retrieve_evidence": True,
                        "ranking_query": "security review product areas",
                        "reasoning": "The user asked for a grouped count.",
                    }
                )
            )
            agent = QueryUnderstandingAgent(corpus, planner=planner)

            result = agent.answer("Break down security review calls by product area.")

            self.assertEqual(result["plan"]["planner"], "fake-llm")
            self.assertEqual(result["plan"]["operation"], "aggregate")
            self.assertEqual(result["plan"]["filters"], {"security_review_requested": True})
            self.assertEqual(result["plan"]["group_by"], "product_area")
            self.assertTrue(result["aggregation"])
            self.assertEqual(result["prompt_state"]["planner"]["prompt_id"], "qu.query_planner")
            self.assertIn("prompt_hash", result["prompt_state"]["planner"])

    def test_aggregate_silver_accepts_bounded_expression(self) -> None:
        corpus = SilverCorpus(
            manifest={"schema_version": "test"},
            catalog={
                "schema_version": "test",
                "fields": [
                    {
                        "name": "product_area",
                        "type": "enum",
                        "allowed_values": ["crm", "analytics"],
                        "search": {"filterable": True, "aggregatable": True},
                    },
                    {
                        "name": "security_review_requested",
                        "type": "boolean",
                        "allowed_values": [],
                        "search": {"filterable": True, "aggregatable": True},
                    },
                    {
                        "name": "deal_size",
                        "type": "string",
                        "allowed_values": [],
                        "search": {"filterable": True, "aggregatable": True},
                    },
                ],
            },
            records=[
                {
                    "call_id": "call-001",
                    "account_name": "Acme",
                    "date": "2026-01-01",
                    "fields": {"product_area": "crm", "security_review_requested": True, "deal_size": 2500},
                    "evidence_refs": {"product_area": ["chunk:call-001:chunk-001"]},
                    "quality_flags": [],
                },
                {
                    "call_id": "call-002",
                    "account_name": "Beta",
                    "date": "2026-01-02",
                    "fields": {"product_area": "crm", "security_review_requested": False, "deal_size": 800},
                    "evidence_refs": {"product_area": ["chunk:call-002:chunk-001"]},
                    "quality_flags": [],
                },
                {
                    "call_id": "call-003",
                    "account_name": "Core",
                    "date": "2026-02-01",
                    "fields": {"product_area": "analytics", "security_review_requested": True, "deal_size": 9000},
                    "evidence_refs": {"product_area": ["chunk:call-003:chunk-001"]},
                    "quality_flags": [],
                },
            ],
            chunks=[],
        )
        agent = QueryUnderstandingAgent(
            corpus,
            planner=AggregationExpressionPlanner(
                'ratio(count_if(records, "security_review_requested", True), count(records))'
            ),
        )

        result = agent.answer("What ratio of calls ask for security review?", limit=5)

        self.assertEqual(result["aggregation"]["numerator"], 2)
        self.assertEqual(result["aggregation"]["denominator"], 3)
        self.assertEqual(result["aggregation"]["ratio"], 0.6667)
        aggregate_events = [event for event in result["trace"] if event["tool"] == "aggregate_silver"]
        self.assertEqual(
            aggregate_events[0]["arguments"]["expression"],
            'ratio(count_if(records, "security_review_requested", True), count(records))',
        )

        top_k = corpus.aggregate_silver_result("", {}, expression='top_k(records, "product_area", k=1)')
        self.assertEqual(top_k.result, {"crm": 2})
        numeric = corpus.aggregate_silver_result(
            "",
            {},
            expression='numeric_range_count(records, "deal_size", min_value=1000, max_value=5000)',
        )
        self.assertEqual(numeric.result["count"], 1)
        with self.assertRaises(ValueError):
            corpus.aggregate_silver_result("", {}, expression='__import__("os").system("whoami")')

    def test_usage_summary_accumulates_query_llm_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = write_sample_calls(Path(tmp) / "sample_calls.jsonl")
            output = Path(tmp) / "corpus"
            run_content_understanding(sample, output, schema_inducer=StaticSchemaInducer())
            planner_client = FakeJSONClient(
                {
                    "operation": "search",
                    "filters": {},
                    "retrieve_evidence": True,
                    "ranking_query": "security review",
                    "steps": [
                        {
                            "tool": "bm25_search_chunks",
                            "arguments": {"query": "security review", "limit": 2},
                            "purpose": "Find matching security chunks.",
                        }
                    ],
                    "reasoning": "Search for evidence.",
                },
                usage={"model": "fake-planner", "input_tokens": 90, "output_tokens": 10},
            )
            judge_client = FakeJSONClient(
                {
                    "answerable": True,
                    "evidence_sufficient": True,
                    "success": True,
                    "needs_cu_feedback": False,
                    "confidence": "high",
                    "failure_modes": [],
                    "missing_field_requests": [],
                    "rationale": "Supported.",
                },
                usage={"model": "fake-judge", "input_tokens": 80, "output_tokens": 12},
            )
            agent = QueryUnderstandingAgent(
                SilverCorpus.from_dir(output),
                planner=LLMQueryPlanner(planner_client),
                answer_judge=LLMAnswerJudge(judge_client),
            )

            result = agent.answer("Which calls mention security review?", limit=2)

            usage = result["usage_summary"]
            self.assertEqual(usage["total_calls"], 2)
            self.assertEqual(usage["input_tokens"], 170)
            self.assertEqual(usage["output_tokens"], 22)
            self.assertEqual(usage["by_model"]["fake-planner"]["total_calls"], 1)
            self.assertEqual(usage["by_model"]["fake-judge"]["total_calls"], 1)

    def test_guardrail_timeout_returns_best_partial_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = write_sample_calls(Path(tmp) / "sample_calls.jsonl")
            output = Path(tmp) / "corpus"
            run_content_understanding(sample, output, schema_inducer=StaticSchemaInducer())
            agent = QueryUnderstandingAgent(
                SilverCorpus.from_dir(output),
                max_timeout_seconds=0,
            )

            result = agent.answer("Which calls mention security review?", limit=2)

            self.assertTrue(result["guardrails"]["stopped"])
            self.assertEqual(result["guardrails"]["stop_reason"], "timeout")
            self.assertIn("timeout", result["judgement"]["failure_modes"])
            self.assertIn("guardrail_stop", [event["tool"] for event in result["trace"]])
            self.assertEqual(result["best_partial_answer"]["record_count"], 0)

    def test_redact_for_replay_strips_answer_snippets(self) -> None:
        answer = {
            "query": "Which calls mention security review?",
            "records": [{"call_id": "c1", "fields": {"security_review_requested": True}}],
            "evidence": [{"chunk_id": "c1:0", "call_id": "c1", "snippet": "raw transcript text", "bm25_score": 1.2}],
            "search_diagnostics": {"calls": [{"tool": "bm25", "top_chunks": [{"chunk_id": "c1:0", "snippet": "more raw text"}]}]},
        }
        redacted = redact_for_replay(answer)
        self.assertEqual(redacted["query"], answer["query"])  # the query itself is preserved
        self.assertEqual(redacted["records"][0]["fields"]["security_review_requested"], True)
        self.assertEqual(redacted["evidence"][0]["snippet"], REDACTED)
        self.assertEqual(redacted["evidence"][0]["bm25_score"], 1.2)
        self.assertEqual(redacted["search_diagnostics"]["calls"][0]["top_chunks"][0]["snippet"], REDACTED)
        self.assertEqual(answer["evidence"][0]["snippet"], "raw transcript text")  # original untouched

    def test_budget_unpriced_warning(self) -> None:
        keys = ("OPENAI_INPUT_USD_PER_MTOK", "OPENAI_OUTPUT_USD_PER_MTOK")
        previous = {key: os.environ.get(key) for key in keys}
        try:
            for key in keys:
                os.environ.pop(key, None)
            self.assertIsNone(budget_unpriced_warning(None))
            self.assertIsNotNone(budget_unpriced_warning(2.5))
            os.environ["OPENAI_INPUT_USD_PER_MTOK"] = "0.15"
            os.environ["OPENAI_OUTPUT_USD_PER_MTOK"] = "0.60"
            self.assertIsNone(budget_unpriced_warning(2.5))
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_search_only_query_emits_column_request_for_cu_feedback(self) -> None:
        sample = workspace / "data" / "sample_calls.jsonl"
        source_sql = next((p for p in [workspace / "call_records.sql"] if p.exists()), None)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "corpus"
            run_content_understanding(sample, output, source_sql=source_sql, schema_inducer=StaticSchemaInducer())
            agent = QueryUnderstandingAgent(SilverCorpus.from_dir(output))

            result = agent.answer("Which calls mention founder-led sales calls?")

            self.assertEqual(result["plan"]["operation"], "search")
            self.assertTrue(result["records"])
            self.assertTrue(result["column_requests"])
            self.assertEqual(result["column_requests"][0]["action"], "add_field")
            self.assertIn("founder", result["column_requests"][0]["field_name"])

            feedback_path = Path(tmp) / "feedback" / "column_requests.jsonl"
            append_feedback(feedback_path, result, corpus_path=output)
            self.assertTrue(feedback_path.exists())
            feedback_payload = json.loads(feedback_path.read_text(encoding="utf-8"))
            self.assertIn("column_requests", feedback_payload)
            self.assertIn("search_diagnostics", feedback_payload)
            self.assertIn("judgement", feedback_payload)

    def test_embedding_search_tool_is_agent_callable(self) -> None:
        sample = workspace / "data" / "sample_calls.jsonl"
        source_sql = next((p for p in [workspace / "call_records.sql"] if p.exists()), None)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "corpus"
            run_content_understanding(sample, output, source_sql=source_sql, schema_inducer=StaticSchemaInducer())
            corpus = SilverCorpus.from_dir(output, embedding_client=FakeEmbeddingClient())
            agent = QueryUnderstandingAgent(corpus, planner=EmbeddingSearchPlanner())

            result = agent.answer("Find founder led conversations.", limit=1)

            self.assertEqual(result["records"][0]["call_id"], "call-005")
            self.assertIn("embedding_search_chunks", [event["tool"] for event in result["trace"]])

    def test_search_iteration_rejects_duplicate_and_reranks_candidates(self) -> None:
        sample = workspace / "data" / "sample_calls.jsonl"
        source_sql = next((p for p in [workspace / "call_records.sql"] if p.exists()), None)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "corpus"
            run_content_understanding(sample, output, source_sql=source_sql, schema_inducer=StaticSchemaInducer())
            corpus = SilverCorpus.from_dir(output, embedding_client=FakeEmbeddingClient())
            agent = QueryUnderstandingAgent(
                corpus,
                search_controller=FakeSequenceJSONClient(
                    [
                        {
                            "action": "search",
                            "tool": "bm25_search_chunks",
                            "query": "Which calls mention founder-led sales calls?",
                            "limit": 2,
                            "reason": "This intentionally repeats the initial retrieval.",
                        }
                    ]
                ),
                reranker=FakeSequenceJSONClient(
                    [
                        {
                            "ranked_chunks": [
                                {"chunk_id": "call-001:chunk-001", "relevance": 0.9, "reason": "Test rerank order."},
                                {"chunk_id": "call-005:chunk-001", "relevance": 0.8, "reason": "Test rerank order."},
                            ],
                            "reasoning": "Reranked by fixture.",
                        }
                    ],
                    provider_name="fake-reranker",
                ),
                search_policy=SearchExecutionPolicy(min_calls=2, max_iterations=1),
            )

            result = agent.answer("Which calls mention founder-led sales calls?", limit=2)

            trace_tools = [event["tool"] for event in result["trace"]]
            self.assertIn("search_iteration:rejected", trace_tools)
            self.assertIn("embedding_search_chunks", trace_tools)
            self.assertIn("llm_rerank:fake-reranker", trace_tools)
            self.assertEqual(result["records"][0]["call_id"], "call-001")
            self.assertGreaterEqual(len(result["search_diagnostics"]["calls"]), 2)

    def test_search_step_fans_out_subqueries_in_parallel_and_merges(self) -> None:
        sample = workspace / "data" / "sample_calls.jsonl"
        source_sql = next((p for p in [workspace / "call_records.sql"] if p.exists()), None)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "corpus"
            run_content_understanding(sample, output, source_sql=source_sql, schema_inducer=StaticSchemaInducer())
            agent = ConcurrencyProbeAgent(
                SilverCorpus.from_dir(output),
                planner=FanoutSearchPlanner(
                    [
                        "security review access controls",
                        "founder led sales calls",
                        "founder led sales calls",  # duplicate collapses before execution
                    ]
                ),
            )

            result = agent.answer("Find security and founder-led calls.", limit=10)

            self.assertEqual(result["plan"]["operation"], "search")
            queries = [call["query"] for call in result["search_diagnostics"]["calls"]]
            self.assertIn("security review access controls", queries)
            self.assertIn("founder led sales calls", queries)
            # The duplicate subquery is dropped, so exactly two fan-out searches run.
            self.assertEqual(queries.count("founder led sales calls"), 1)
            fanout_events = [event for event in result["trace"] if event["arguments"].get("fanout")]
            self.assertTrue(fanout_events)
            self.assertEqual(fanout_events[0]["arguments"]["fanout"], 2)
            self.assertIn("subagent_join", [event["tool"] for event in result["trace"]])
            branch_calls = result["subagent_diagnostics"]["calls"]
            self.assertEqual(len(branch_calls), 2)
            self.assertEqual(branch_calls[0]["validation_result"], "ok")
            self.assertEqual(branch_calls[0]["call"]["capability"], "retrieval_branch")
            self.assertTrue(branch_calls[0]["call"]["input_refs"])
            self.assertIn("output_schema", branch_calls[0]["call"])
            self.assertIn("max_results", branch_calls[0]["call"]["budget"])
            record_ids = [record["call_id"] for record in result["records"]]
            self.assertEqual(len(record_ids), len(set(record_ids)))  # merged + deduped
            self.assertIn("call-001", record_ids)
            self.assertIn("call-005", record_ids)
            self.assertGreaterEqual(agent.max_active, 2)  # subqueries ran concurrently

    def test_qu_evaluates_cu_field_candidates(self) -> None:
        sample = workspace / "data" / "sample_calls.jsonl"
        source_sql = next((p for p in [workspace / "call_records.sql"] if p.exists()), None)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "corpus"
            run_content_understanding(sample, output, source_sql=source_sql, schema_inducer=StaticSchemaInducer())
            corpus = SilverCorpus.from_dir(output)

            report = evaluate_field_candidates(
                corpus,
                query_tasks=[
                    {
                        "task_id": "security",
                        "query": "Which calls mention security review?",
                        "expected_operation": "filter",
                    }
                ],
            )

            self.assertEqual(report["contract"], "qu.field_candidate_evaluation@2026-06-02.1")
            self.assertTrue(report["evaluations"])
            security = next(item for item in report["evaluations"] if item["field_name"] == "security_review_requested")
            self.assertIn(security["decision"], report["decision_values"])
            self.assertEqual(security["validation_result"], "ok")
            self.assertIn("filter", security["simulation"])

    def test_agent_accepts_specialized_retrieval_subagent(self) -> None:
        sample = workspace / "data" / "sample_calls.jsonl"
        source_sql = next((p for p in [workspace / "call_records.sql"] if p.exists()), None)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "corpus"
            run_content_understanding(sample, output, source_sql=source_sql, schema_inducer=StaticSchemaInducer())
            corpus = SilverCorpus.from_dir(output, embedding_client=FakeEmbeddingClient())
            agent = QueryUnderstandingAgent(
                corpus,
                retrieval_subagent=Sid1StyleRetrievalSubAgent(),
            )

            result = agent.answer("Which calls mention founder-led sales calls?", limit=2)

            trace_tools = [event["tool"] for event in result["trace"]]
            self.assertIn("embedding_search_chunks", trace_tools)
            self.assertIn("retrieval_subagent:sid1-fixture", trace_tools)
            self.assertEqual(result["search_diagnostics"]["subagent"], "sid1-fixture")

    def test_search_failure_generates_schema_gap_request(self) -> None:
        sample = workspace / "data" / "sample_calls.jsonl"
        source_sql = next((p for p in [workspace / "call_records.sql"] if p.exists()), None)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "corpus"
            run_content_understanding(sample, output, source_sql=source_sql, schema_inducer=StaticSchemaInducer())
            agent = QueryUnderstandingAgent(SilverCorpus.from_dir(output))

            result = agent.answer("Which calls mention zzzqv impossible terminology?")

            self.assertFalse(result["records"])
            self.assertTrue(result["search_diagnostics"]["failures"])
            self.assertFalse(result["judgement"]["success"])
            self.assertIn("no_records", result["judgement"]["failure_modes"])
            self.assertTrue(result["column_requests"])
            self.assertIn("Search diagnostics", result["column_requests"][0]["reason"])

    def test_failed_plan_observation_can_replan_without_dropping_cu_field_proposal(self) -> None:
        sample = workspace / "data" / "sample_calls.jsonl"
        source_sql = next((p for p in [workspace / "call_records.sql"] if p.exists()), None)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "corpus"
            run_content_understanding(sample, output, source_sql=source_sql, schema_inducer=StaticSchemaInducer())
            planner = ObservedReplanPlanner()
            agent = QueryUnderstandingAgent(
                SilverCorpus.from_dir(output),
                planner=planner,
                max_plan_iterations=2,
            )

            result = agent.answer("Which calls mention security review or access controls?", limit=2)

            self.assertEqual(result["plan"]["operation"], "filter")
            self.assertEqual(result["plan"]["filters"], {"security_review_requested": True})
            self.assertEqual(len(result["plan_iterations"]), 2)
            self.assertTrue(result["records"])
            self.assertTrue(result["column_requests"])
            trace_tools = [event["tool"] for event in result["trace"]]
            self.assertIn("plan_observation", trace_tools)
            self.assertIn("replan_query:observed-static", trace_tools)
            self.assertEqual(len(planner.observations), 1)
            self.assertIn("no_records", planner.observations[0]["judgement"]["failure_modes"])
            self.assertTrue(planner.observations[0]["column_requests"])

    def test_llm_answer_judge_can_emit_feedback_without_planner_column_request(self) -> None:
        sample = workspace / "data" / "sample_calls.jsonl"
        source_sql = next((p for p in [workspace / "call_records.sql"] if p.exists()), None)
        judge_payload = {
            "answerable": True,
            "evidence_sufficient": True,
            "success": True,
            "needs_cu_feedback": True,
            "confidence": "medium",
            "failure_modes": ["schema_gap"],
            "missing_field_requests": [
                {
                    "action": "add_field",
                    "field_name": "security_evidence_type",
                    "field_type": "list",
                    "description": "Reusable security evidence category.",
                    "reason": "Judge found that answers would be easier to verify with a reusable evidence type.",
                    "priority": "medium",
                    "suggested_allowed_values": ["sso", "audit_logs"],
                    "example_queries": ["Which calls mention security review or access controls?"],
                    "evidence_refs": ["chunk:call-001:chunk-001"],
                }
            ],
            "rationale": "The answer is supported, but schema can improve reusable security evidence.",
        }

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "corpus"
            run_content_understanding(sample, output, source_sql=source_sql, schema_inducer=StaticSchemaInducer())
            agent = QueryUnderstandingAgent(
                SilverCorpus.from_dir(output),
                answer_judge=LLMAnswerJudge(FakeJSONClient(judge_payload)),
            )

            result = agent.answer("Which calls mention security review or access controls?")

            self.assertEqual(result["judgement"]["judge"], "fake-llm")
            self.assertTrue(result["judgement"]["needs_cu_feedback"])
            self.assertEqual(result["prompt_state"]["answer_judge"]["prompt_id"], "qu.answer_judge")
            self.assertEqual(result["column_requests"], [])
            self.assertIn("answer_judge:fake-llm", [event["tool"] for event in result["trace"]])

            feedback_path = Path(tmp) / "feedback" / "column_requests.jsonl"
            append_feedback(feedback_path, result, corpus_path=output)
            feedback_payload = json.loads(feedback_path.read_text(encoding="utf-8"))
            self.assertEqual(feedback_payload["column_requests"], [])
            self.assertEqual(
                feedback_payload["judgement"]["missing_field_requests"][0]["field_name"],
                "security_evidence_type",
            )

    def test_heuristic_planner_uses_feedback_refined_fields(self) -> None:
        sample = workspace / "data" / "sample_calls.jsonl"
        source_sql = next((p for p in [workspace / "call_records.sql"] if p.exists()), None)
        feedback_payload = {
            "query": "Which calls mention founder-led sales calls?",
            "column_requests": [
                {
                    "action": "add_field",
                    "field_name": "founder_led_sales",
                    "field_type": "list",
                    "description": "Reusable silver signal for founder-led sales calls.",
                    "reason": "No current silver field represented this concept.",
                    "priority": "high",
                    "suggested_allowed_values": ["founder", "led", "sales"],
                    "example_queries": ["Which calls mention founder-led sales calls?"],
                    "evidence_refs": ["chunk:call-005:chunk-001"],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            feedback_path = Path(tmp) / "feedback.jsonl"
            feedback_path.write_text(json.dumps(feedback_payload) + "\n", encoding="utf-8")
            output = Path(tmp) / "corpus"
            run_content_understanding(
                sample,
                output,
                source_sql=source_sql,
                schema_inducer=StaticSchemaInducer(),
                feedback_input=feedback_path,
            )
            agent = QueryUnderstandingAgent(SilverCorpus.from_dir(output))

            result = agent.answer("Which calls mention founder-led sales calls?")

            self.assertEqual(result["plan"]["operation"], "filter")
            self.assertEqual(result["plan"]["filters"], {"founder_led_sales": "founder"})
            self.assertEqual(result["records"][0]["call_id"], "call-005")

    def test_bm25_index_handles_corpus_with_no_search_tokens(self) -> None:
        index = BM25Index(
            [
                {
                    "chunk_id": "chunk-001",
                    "call_id": "call-001",
                    "bm25_text": "the and call",
                    "embedding_text": "",
                }
            ]
        )

        self.assertEqual(index.search("the call", allowed_call_ids=None, limit=1), [])

    def test_bm25_index_returns_match_details(self) -> None:
        index = BM25Index(
            [
                {
                    "chunk_id": "chunk-001",
                    "call_id": "call-001",
                    "bm25_text": "founder led sales calls",
                    "embedding_text": "",
                }
            ]
        )

        results = index.search("founder sales", allowed_call_ids=None, limit=1)

        self.assertEqual(results[0]["matched_terms"], ["founder", "sales"])
        self.assertEqual(results[0]["query_terms"], ["founder", "sales"])
        self.assertIn("score_details", results[0])


    def test_cli_loads_openai_env_file_without_printing_secret_defaults(self) -> None:
        keys = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")
        previous = {key: os.environ.get(key) for key in keys}
        for key in keys:
            os.environ.pop(key, None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                env_file = Path(tmp) / ".env"
                env_file.write_text(
                    "OPENAI_API_KEY=sk-test-secret\nOPENAI_BASE_URL=https://example.test/v1\n",
                    encoding="utf-8",
                )

                args = parse_args(["--env-file", str(env_file)])

                self.assertEqual(args.llm_api_key, "sk-test-secret")
                self.assertEqual(args.llm_base_url, "https://example.test/v1")
                self.assertEqual(args.llm_model, DEFAULT_OPENAI_MODEL)
                self.assertEqual(args.judge_api_key, "sk-test-secret")
                self.assertEqual(args.judge_base_url, "https://example.test/v1")
                self.assertIsNone(args.judge_model)

                routed = parse_args(
                    [
                        "--env-file",
                        str(env_file),
                        "--judge-model",
                        "gpt-5.4",
                        "--judge-base-url",
                        "https://judge.example.test/v1",
                    ]
                )
                self.assertEqual(routed.judge_model, "gpt-5.4")
                self.assertEqual(routed.judge_base_url, "https://judge.example.test/v1")
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_cli_rejects_removed_non_llm_modes(self) -> None:
        for args in (
            ["--planner", "heuristic"],
            ["--planner", "auto"],
            ["--judge", "heuristic"],
            ["--judge", "auto"],
            ["--judge", "none"],
            ["--retrieval-subagent", "auto"],
            ["--reranker", "auto"],
            ["--search-controller", "auto"],
        ):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                parse_args(args)

    def test_llm_downstream_query_generator_records_provenance(self) -> None:
        catalog = {
            "schema_version": "test",
            "fields": [
                {
                    "name": "product_area",
                    "type": "enum",
                    "description": "Product area discussed in the call.",
                    "allowed_values": ["crm"],
                    "search": {"filterable": True, "aggregatable": True},
                }
            ],
        }
        adjusted_for = adjusted_for_cu_schema(
            loop_id="loop-test",
            iteration=0,
            cu_output=Path("/tmp/cu"),
            catalog=catalog,
            manifest={"record_count": 2, "chunk_count": 3},
        )
        generator = LLMDownstreamQueryGenerator(
            FakeJSONClient(
                {
                    "tasks": [
                        {
                            "query": "Count calls by product area.",
                            "intent": "aggregate",
                            "expected_operation": "aggregate",
                            "targets_schema_gaps": ["product_area"],
                            "rationale": "Aggregation probe.",
                        }
                    ],
                    "coverage_notes": "One aggregate probe.",
                }
            )
        )

        tasks, report = generator.generate(
            catalog=catalog,
            manifest={"record_count": 2, "chunk_count": 3},
            existing_tasks=[],
            adjusted_for=adjusted_for,
            max_tasks=3,
        )

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["source_type"], "synthetic_llm")
        self.assertEqual(tasks[0]["label_status"], "unlabeled")
        self.assertEqual(tasks[0]["adjusted_for"]["schema_hash"], adjusted_for["schema_hash"])
        self.assertEqual(tasks[0]["provenance"]["prompt"]["prompt_id"], "qu.downstream_query_bootstrap")
        self.assertEqual(report.generated_count, 1)

    def test_heuristic_bootstrap_tasks_are_dedupable(self) -> None:
        catalog = {
            "schema_version": "test",
            "fields": [
                {
                    "name": "product_area",
                    "type": "enum",
                    "allowed_values": ["crm"],
                    "search": {"filterable": True, "aggregatable": True},
                }
            ],
        }
        adjusted_for = adjusted_for_cu_schema(
            loop_id="loop-test",
            iteration=0,
            cu_output=Path("/tmp/cu"),
            catalog=catalog,
            manifest={"record_count": 2, "chunk_count": 3},
        )

        tasks, report = build_heuristic_bootstrap_tasks(
            catalog=catalog,
            adjusted_for=adjusted_for,
            max_tasks=4,
        )

        self.assertTrue(tasks)
        self.assertEqual(report.generator, "heuristic")
        self.assertEqual(len(dedupe_query_tasks([tasks[0], tasks[0]])), 1)
        self.assertEqual(tasks[0]["adjusted_for"]["purpose"], "bootstrap downstream query distribution for CU-QU feedback")


class StaticPlanner:
    name = "static"

    def plan(self, query, catalog):
        return QueryPlan(
            operation="filter",
            filters={"security_review_requested": True},
            ranking_query=query,
            planner=self.name,
            reasoning="Test planner injection.",
        )


class EmbeddingSearchPlanner:
    name = "embedding-static"

    def plan(self, query, catalog):
        return QueryPlan(
            operation="search",
            ranking_query="founder led conversations",
            planner=self.name,
            reasoning="Test embedding retrieval tool.",
            steps=[
                QueryToolStep(
                    tool="embedding_search_chunks",
                    arguments={"query": "founder led conversations", "limit": 1, "promote_records": True},
                    purpose="Exercise embedding retrieval.",
                ),
                QueryToolStep(tool="fetch_chunks", arguments={"limit": 1}, purpose="Fetch evidence."),
            ],
        )


class FanoutSearchPlanner:
    name = "fanout-static"

    def __init__(self, queries):
        self.queries = queries

    def plan(self, query, catalog):
        del catalog
        return QueryPlan(
            operation="search",
            ranking_query=query,
            planner=self.name,
            reasoning="Fan-out test planner.",
            steps=[
                QueryToolStep(
                    tool="bm25_search_chunks",
                    arguments={"queries": self.queries, "limit": 5, "promote_records": True},
                    purpose="Parallel subquery fan-out.",
                ),
                QueryToolStep(tool="fetch_chunks", arguments={"limit": 5}, purpose="Fetch evidence."),
            ],
        )


class AggregationExpressionPlanner:
    name = "aggregate-expression-static"

    def __init__(self, expression):
        self.expression = expression

    def plan(self, query, catalog):
        del catalog
        return QueryPlan(
            operation="aggregate",
            aggregation_expression=self.expression,
            retrieve_evidence=False,
            ranking_query=query,
            planner=self.name,
            reasoning="Fixture aggregate expression plan.",
            steps=[
                QueryToolStep(
                    tool="query_silver",
                    arguments={"filters": {}, "limit": 50},
                    purpose="Load records for expression aggregation.",
                ),
                QueryToolStep(
                    tool="aggregate_silver",
                    arguments={"expression": self.expression, "filters": {}},
                    purpose="Run bounded aggregate expression.",
                ),
            ],
        )


class ObservedReplanPlanner:
    name = "observed-static"

    def __init__(self):
        self.observations = []

    def plan(self, query, catalog):
        del catalog
        return QueryPlan(
            operation="search",
            ranking_query="nonexistent terminology",
            planner=self.name,
            reasoning="First attempt intentionally misses so observation can drive replanning.",
            steps=[
                QueryToolStep(
                    tool="bm25_search_chunks",
                    arguments={"query": "nonexistent terminology", "limit": 3, "promote_records": True},
                    purpose="Initial brittle retrieval attempt.",
                ),
                QueryToolStep(tool="fetch_chunks", arguments={"limit": 3}, purpose="Fetch evidence if any exists."),
                QueryToolStep(tool="review_schema_gaps", arguments={}, purpose="Capture CU feedback from failure."),
            ],
        )

    def replan(self, query, catalog, observation):
        del query, catalog
        self.observations.append(observation)
        return QueryPlan(
            operation="filter",
            filters={"security_review_requested": True},
            ranking_query="security review access controls",
            planner=self.name,
            reasoning="Observation showed no records, so use the available security silver field.",
        )


class ConcurrencyProbeAgent(QueryUnderstandingAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._probe_lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def search_with_tool(self, tool, query, *, filters, limit):
        with self._probe_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.02)
            return super().search_with_tool(tool, query, filters=filters, limit=limit)
        finally:
            with self._probe_lock:
                self.active -= 1


class FakeEmbeddingClient:
    model = "fake-embedding"

    def embed_texts(self, texts):
        embeddings = []
        for text in texts:
            normalized = text.lower().replace("-", " ")
            embeddings.append([1.0, 0.0] if "founder led" in normalized else [0.0, 1.0])
        return embeddings


class Sid1StyleRetrievalSubAgent:
    name = "sid1-fixture"

    def __init__(self):
        self.called = False

    def after_search_step(
        self,
        *,
        query,
        plan,
        state,
        limit,
        trace,
        execute_step,
        available_tools,
        trace_tool_name,
    ):
        del available_tools, trace_tool_name
        if self.called:
            return
        self.called = True
        execute_step(
            QueryToolStep(
                tool="embedding_search_chunks",
                arguments={"query": "founder led conversations", "limit": 1, "promote_records": True},
                purpose="SID-1 fixture semantic recall branch.",
            ),
            query=query,
            plan=plan,
            state=state,
            limit=limit,
            trace=trace,
        )

    def finalize(self, *, query, plan, state, limit, trace):
        del query, plan, state, limit
        trace.append(
            ToolEvent(
                step=len(trace) + 1,
                tool="retrieval_subagent:sid1-fixture",
                arguments={"mode": "fixture"},
                result_summary="Executed specialized retrieval subagent fixture.",
            )
        )


class FakeJSONClient:
    provider_name = "fake-llm"

    def __init__(self, payload, usage=None):
        self.payload = payload
        self.usage = usage
        self.usage_summary = UsageSummary()

    def complete_json(self, *, system, user):
        self.system = system
        self.user = user
        if self.usage:
            self.usage_summary.add_call(
                model=self.usage["model"],
                input_tokens=self.usage["input_tokens"],
                output_tokens=self.usage["output_tokens"],
            )
        return self.payload


class FakeSequenceJSONClient:
    def __init__(self, payloads, provider_name="fake-llm"):
        self.payloads = list(payloads)
        self.provider_name = provider_name
        self.calls = []
        self.usage_summary = UsageSummary()

    def complete_json(self, *, system, user):
        self.calls.append({"system": system, "user": user})
        if not self.payloads:
            return {"action": "stop"}
        return self.payloads.pop(0)


def write_sample_calls(path: Path) -> Path:
    rows = [
        {
            "call_id": "call-001",
            "customer_id": "cust-001",
            "account_name": "Acme",
            "date": "2026-01-01",
            "calllog_result": [
                {
                    "speaker": "customer",
                    "best_text": "We need SSO and audit logs for security review.",
                }
            ],
        },
        {
            "call_id": "call-002",
            "customer_id": "cust-002",
            "account_name": "Beta",
            "date": "2026-01-02",
            "calllog_result": [
                {
                    "speaker": "customer",
                    "best_text": "Pricing is blocking renewal.",
                }
            ],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
