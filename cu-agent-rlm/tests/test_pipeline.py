from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cu_agent_rlm.cli import parse_args
from cu_agent_rlm.pipeline import run_content_understanding
from cu_agent_rlm.extraction import LLMFieldExtractor
from cu_agent_rlm.io import load_calls
from cu_agent_rlm.llm import DEFAULT_OPENAI_MODEL
from cu_agent_rlm.pipeline import analyze_calls
from cu_agent_rlm.schema import LLMSchemaInducer, StaticSchemaInducer
from cu_agent_rlm.usage import UsageSummary


class ContentUnderstandingRLMTest(unittest.TestCase):
    def test_sample_call_records_generate_silver_contract(self) -> None:
        workspace = Path(__file__).resolve().parents[2]
        sample = workspace / "cu-agent" / "data" / "sample_calls.jsonl"
        source_sql = workspace / "call_records.sql"

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            artifact = run_content_understanding(sample, output, source_sql=source_sql)

            self.assertEqual(artifact.manifest["record_count"], 8)
            self.assertEqual(len(artifact.silver_calls), 8)
            self.assertTrue((output / "manifest.json").exists())
            self.assertTrue((output / "chunks.jsonl").exists())
            self.assertTrue((output / "silver_schema_catalog.json").exists())
            self.assertTrue((output / "silver_calls.jsonl").exists())
            self.assertTrue((output / "rlm_trace.jsonl").exists())

            self.assertEqual(artifact.manifest["schema_induction"]["inducer"], "heuristic")
            field_names = {field["name"] for field in artifact.silver_schema_catalog["fields"]}
            self.assertIn("conversation_topic", field_names)
            self.assertIn("risk_or_blocker", field_names)
            self.assertIn("next_action", field_names)
            self.assertIn("fetch_chunks", artifact.silver_schema_catalog["tools"])
            self.assertIn("run_sql", artifact.databricks_contract["allowlisted_tools"])
            self.assertIn("calllog_result", artifact.databricks_contract["source"]["observed_columns"])

            topic_calls = [call for call in artifact.silver_calls if call.fields.get("conversation_topic")]
            self.assertTrue(topic_calls)
            self.assertTrue(all(ref.startswith("chunk:") for call in topic_calls for ref in call.evidence_refs["conversation_topic"]))
            self.assertNotIn("manual work", json.dumps(artifact.manifest, ensure_ascii=False).lower())
            self.assertTrue(any(event.actor == "sub-rlm" for event in artifact.trace))
            self.assertTrue(artifact.evaluation_tasks)

    def test_qu_feedback_refines_schema_induction(self) -> None:
        workspace = Path(__file__).resolve().parents[2]
        sample = workspace / "cu-agent" / "data" / "sample_calls.jsonl"
        source_sql = workspace / "call_records.sql"

        feedback_payload = {
            "query": "Which calls mention founder-led sales calls?",
            "column_requests": [
                {
                    "action": "add_field",
                    "field_name": "founder_led_sales",
                    "field_type": "boolean",
                    "description": "Whether the call discusses founder-led sales calls.",
                    "reason": "QU needed retrieval because no current silver field represented this concept.",
                    "priority": "high",
                    "suggested_allowed_values": [],
                    "example_queries": ["Which calls mention founder-led sales calls?"],
                    "evidence_refs": ["chunk:call-005:chunk-001"],
                }
            ],
            "search_diagnostics": {
                "failures": [
                    {
                        "failure_reason": "BM25 found only lexical evidence; schema was missing founder-led sales.",
                        "query_terms": ["founder", "led", "sales"],
                        "missing_terms": [],
                        "tool": "bm25_search_chunks",
                    }
                ]
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            feedback_path = Path(tmp) / "feedback.jsonl"
            feedback_path.write_text(json.dumps(feedback_payload, ensure_ascii=False) + "\n", encoding="utf-8")

            artifact = run_content_understanding(
                sample,
                output,
                source_sql=source_sql,
                feedback_input=feedback_path,
            )

            manifest_fields = {field["name"] for field in artifact.manifest["schema_induction"]["fields"]}
            catalog_fields = {field["name"] for field in artifact.silver_schema_catalog["fields"]}
            self.assertEqual(artifact.manifest["schema_induction"]["inducer"], "heuristic+feedback")
            self.assertIn("founder_led_sales", manifest_fields)
            self.assertIn("founder_led_sales", catalog_fields)
            self.assertEqual(artifact.manifest["feedback_refinement"]["request_count"], 1)
            requested = artifact.feedback_report["requested_fields"][0]
            self.assertTrue(requested["accepted_into_schema"])
            self.assertTrue((output / "feedback_report.json").exists())

    def test_answerability_judgement_feedback_refines_schema_induction(self) -> None:
        workspace = Path(__file__).resolve().parents[2]
        sample = workspace / "cu-agent" / "data" / "sample_calls.jsonl"
        source_sql = workspace / "call_records.sql"

        feedback_payload = {
            "query": "Which calls mention founder-led sales calls?",
            "column_requests": [],
            "judgement": {
                "answerable": False,
                "evidence_sufficient": False,
                "success": False,
                "needs_cu_feedback": True,
                "confidence": "low",
                "failure_modes": ["no_records", "schema_gap"],
                "missing_field_requests": [
                    {
                        "action": "add_field",
                        "field_name": "founder_led_sales",
                        "field_type": "list",
                        "description": "Reusable founder-led sales signal.",
                        "reason": "Answerability judge found this concept missing from silver fields.",
                        "priority": "high",
                        "suggested_allowed_values": ["founder", "led", "sales"],
                        "example_queries": ["Which calls mention founder-led sales calls?"],
                        "evidence_refs": ["chunk:call-005:chunk-001"],
                    }
                ],
                "rationale": "No reusable silver field represented the query.",
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            feedback_path = Path(tmp) / "feedback.jsonl"
            feedback_path.write_text(json.dumps(feedback_payload, ensure_ascii=False) + "\n", encoding="utf-8")

            artifact = run_content_understanding(
                sample,
                output,
                source_sql=source_sql,
                feedback_input=feedback_path,
            )

            catalog_fields = {field["name"] for field in artifact.silver_schema_catalog["fields"]}
            self.assertIn("founder_led_sales", catalog_fields)
            self.assertEqual(artifact.manifest["feedback_refinement"]["answerability_failure_count"], 1)
            self.assertEqual(artifact.manifest["feedback_refinement"]["evidence_failure_count"], 1)

    def test_llm_extractor_validates_json_against_schema_and_evidence(self) -> None:
        workspace = Path(__file__).resolve().parents[2]
        sample = workspace / "cu-agent" / "data" / "sample_calls.jsonl"
        calls = load_calls(sample)[:1]
        payload = {
            "fields": [
                {
                    "name": "security_review_requested",
                    "value": True,
                    "confidence": "high",
                    "evidence_refs": ["chunk:call-001:chunk-001"],
                    "abstained": False,
                    "rationale": "The call discusses SSO, audit logs, and redaction.",
                },
                {
                    "name": "urgency",
                    "value": "extreme",
                    "confidence": "high",
                    "evidence_refs": ["chunk:call-001:chunk-001"],
                    "abstained": False,
                    "rationale": "Intentionally invalid enum for validator coverage.",
                },
            ]
        }

        artifact = analyze_calls(
            calls,
            schema_inducer=StaticSchemaInducer(),
            field_extractor=LLMFieldExtractor(FakeJSONClient(payload)),
        )

        security = next(item for item in artifact.extractions if item.field_name == "security_review_requested")
        urgency = next(item for item in artifact.extractions if item.field_name == "urgency")
        self.assertEqual(security.value, True)
        self.assertEqual(security.evidence_refs, ["chunk:call-001:chunk-001"])
        self.assertEqual(security.validation_errors, [])
        self.assertIn("value_not_allowed:extreme", urgency.validation_errors)
        self.assertTrue(any(event.tool == "extract_call_fields" for event in artifact.trace))

    def test_llm_schema_inducer_drives_extraction_schema(self) -> None:
        workspace = Path(__file__).resolve().parents[2]
        sample = workspace / "cu-agent" / "data" / "sample_calls.jsonl"
        calls = load_calls(sample)[:1]
        schema_payload = {
            "fields": [
                {
                    "name": "compliance_requirement",
                    "type": "list",
                    "description": "Compliance or governance requirements mentioned by the customer.",
                    "allowed_values": ["sso", "audit_logs", "redaction"],
                    "downstream_use_cases": ["filtering", "evidence-backed review"],
                    "filterable": True,
                    "facetable": True,
                    "aggregatable": True,
                }
            ]
        }
        extraction_payload = {
            "fields": [
                {
                    "name": "compliance_requirement",
                    "value": ["sso", "audit_logs", "redaction"],
                    "confidence": "high",
                    "evidence_refs": ["chunk:call-001:chunk-001"],
                    "abstained": False,
                    "rationale": "The call mentions SSO, audit logs, and redaction.",
                }
            ]
        }

        artifact = analyze_calls(
            calls,
            schema_inducer=LLMSchemaInducer(FakeJSONClient(schema_payload)),
            field_extractor=LLMFieldExtractor(FakeJSONClient(extraction_payload)),
        )

        field_names = {field["name"] for field in artifact.silver_schema_catalog["fields"]}
        self.assertIn("compliance_requirement", field_names)
        self.assertEqual(
            artifact.silver_calls[0].fields["compliance_requirement"],
            ["sso", "audit_logs", "redaction"],
        )
        self.assertEqual(artifact.manifest["schema_induction"]["inducer"], "fake-llm")
        self.assertEqual(artifact.manifest["prompt_state"]["schema_inducer"]["prompt_id"], "cu.schema_induction")
        self.assertEqual(artifact.manifest["prompt_state"]["field_extractor"]["prompt_id"], "cu.field_extraction")
        self.assertIn("prompt_hash", artifact.manifest["prompt_state"]["schema_inducer"])

    def test_usage_summary_accumulates_llm_schema_and_extraction_calls(self) -> None:
        workspace = Path(__file__).resolve().parents[2]
        sample = workspace / "legacy" / "cu-agent" / "data" / "sample_calls.jsonl"
        calls = load_calls(sample)[:1]
        schema_client = FakeJSONClient(
            {
                "fields": [
                    {
                        "name": "compliance_requirement",
                        "type": "list",
                        "description": "Compliance requirements mentioned by the customer.",
                        "allowed_values": ["sso"],
                        "downstream_use_cases": ["filtering"],
                    }
                ]
            },
            usage={"model": "fake-schema", "input_tokens": 100, "output_tokens": 20},
        )
        extraction_client = FakeJSONClient(
            {
                "fields": [
                    {
                        "name": "compliance_requirement",
                        "value": ["sso"],
                        "confidence": "high",
                        "evidence_refs": ["chunk:call-001:chunk-001"],
                        "abstained": False,
                        "rationale": "The call mentions SSO.",
                    }
                ]
            },
            usage={"model": "fake-extraction", "input_tokens": 200, "output_tokens": 30},
        )

        artifact = analyze_calls(
            calls,
            schema_inducer=LLMSchemaInducer(schema_client),
            field_extractor=LLMFieldExtractor(extraction_client),
        )

        usage = artifact.manifest["usage_summary"]
        self.assertEqual(usage["total_calls"], 2)
        self.assertEqual(usage["input_tokens"], 300)
        self.assertEqual(usage["output_tokens"], 50)
        self.assertEqual(usage["by_model"]["fake-schema"]["total_calls"], 1)
        self.assertEqual(usage["by_model"]["fake-extraction"]["total_calls"], 1)

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
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_cli_rejects_removed_non_llm_modes(self) -> None:
        for flag, value in (
            ("--schema-inducer", "heuristic"),
            ("--schema-inducer", "static"),
            ("--schema-inducer", "auto"),
            ("--extractor", "heuristic"),
            ("--extractor", "auto"),
        ):
            with self.subTest(flag=flag, value=value), self.assertRaises(SystemExit):
                parse_args([flag, value])


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


if __name__ == "__main__":
    unittest.main()
