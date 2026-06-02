from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from portable_extractor import derive_response_schema, validate as portable_validate

from cu_agent_rlm.chunking import build_chunks
from cu_agent_rlm.cli import parse_args
from cu_agent_rlm.extraction import ExtractionError, LLMFieldExtractor, validate_extraction_payload
from cu_agent_rlm.pipeline import run_content_understanding
from cu_agent_rlm.io import load_calls
from cu_agent_rlm.llm import DEFAULT_OPENAI_MODEL
from cu_agent_rlm.models import CallRecord, FieldSpec, TranscriptTurn
from cu_agent_rlm.pipeline import analyze_calls
from cu_agent_rlm.replay import REDACTED, redact_for_replay
from cu_agent_rlm.schema import LLMSchemaInducer, StaticSchemaInducer
from cu_agent_rlm.usage import UsageSummary, budget_unpriced_warning


class ContentUnderstandingRLMTest(unittest.TestCase):
    def test_sample_call_records_generate_silver_contract(self) -> None:
        workspace = Path(__file__).resolve().parents[2]
        sample = workspace / "data" / "sample_calls.jsonl"
        source_sql = next((p for p in [workspace / "call_records.sql"] if p.exists()), None)

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
            self.assertTrue((output / "extraction_contract.json").exists())
            self.assertTrue((output / "field_candidates.json").exists())
            self.assertTrue((output / "schema_negotiation.json").exists())

            self.assertEqual(artifact.manifest["schema_induction"]["inducer"], "heuristic")
            self.assertEqual(artifact.manifest["portable_extraction"]["contract"], "extraction_contract.json")
            self.assertEqual(artifact.manifest["schema_negotiation"]["contract"], "cu.qu_schema_negotiation@2026-06-02.1")
            self.assertEqual(artifact.extraction_contract["runtime"]["module"], "portable_extractor")
            self.assertFalse(artifact.extraction_contract["runtime"]["imports_cu_agent_rlm"])
            self.assertEqual(artifact.extraction_contract["aggregation"]["contract_id"], "qu.aggregate_silver.expression")
            field_names = {field["name"] for field in artifact.silver_schema_catalog["fields"]}
            self.assertIn("conversation_topic", field_names)
            self.assertIn("risk_or_blocker", field_names)
            self.assertIn("next_action", field_names)
            candidate_names = {candidate["field_name"] for candidate in artifact.field_candidates}
            self.assertIn("conversation_topic", candidate_names)
            self.assertTrue(
                any(decision["decision"] in artifact.schema_negotiation["decision_values"] for decision in artifact.schema_negotiation["decisions"])
            )
            self.assertIn("field_candidate_proposal", [event.tool for event in artifact.trace])
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
        sample = workspace / "data" / "sample_calls.jsonl"
        source_sql = next((p for p in [workspace / "call_records.sql"] if p.exists()), None)

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
        sample = workspace / "data" / "sample_calls.jsonl"
        source_sql = next((p for p in [workspace / "call_records.sql"] if p.exists()), None)

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
        sample = workspace / "data" / "sample_calls.jsonl"
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
        sample = workspace / "data" / "sample_calls.jsonl"
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

    def test_portable_extractor_matches_runtime_validator(self) -> None:
        call = sample_call_records()[0]
        chunks = build_chunks([call])
        specs = [
            FieldSpec(
                name="compliance_requirement",
                type="list",
                description="Compliance requirements mentioned by the customer.",
                allowed_values=["sso", "audit_logs"],
                downstream_use_cases=["filtering"],
            ),
            FieldSpec(
                name="security_review_requested",
                type="boolean",
                description="Whether security review is requested.",
                allowed_values=[],
                downstream_use_cases=["review queue"],
            ),
        ]
        payload = {
            "fields": [
                {
                    "name": "compliance_requirement",
                    "value": ["sso", "unexpected_value"],
                    "confidence": "high",
                    "evidence_refs": [f"chunk:{chunks[0].chunk_id}"],
                    "abstained": False,
                    "rationale": "The call mentions SSO.",
                }
            ]
        }

        runtime_rows = [
            extraction.__dict__
            for extraction in validate_extraction_payload(
                payload,
                call=call,
                chunks=chunks,
                specs=specs,
                extractor="fixture",
            )
        ]
        portable_rows = portable_validate(
            payload,
            specs,
            {chunk.chunk_id for chunk in chunks},
            call_id=call.call_id,
            extractor="fixture",
        )

        self.assertEqual(portable_rows, runtime_rows)
        response_schema = derive_response_schema(specs)
        item_schema = response_schema["properties"]["fields"]["items"]["anyOf"][0]
        self.assertEqual(item_schema["properties"]["name"]["enum"], ["compliance_requirement"])
        self.assertEqual(item_schema["properties"]["value"]["items"]["enum"], ["sso", "audit_logs"])

    def test_llm_schema_inducer_uses_strict_structured_output_when_supported(self) -> None:
        calls = sample_call_records()[:1]
        chunks = build_chunks(calls)
        schema_payload = {
            "fields": [
                {
                    "name": "compliance_requirement",
                    "type": "list",
                    "description": "Compliance or governance requirements mentioned by the customer.",
                    "allowed_values": ["sso"],
                    "downstream_use_cases": ["filtering"],
                    "filterable": True,
                    "facetable": True,
                    "aggregatable": True,
                }
            ]
        }
        client = SchemaAwareFakeJSONClient(schema_payload)

        specs = LLMSchemaInducer(client).induce_schema(calls, chunks)

        self.assertEqual([spec.name for spec in specs], ["compliance_requirement"])
        self.assertEqual(client.response_format["type"], "json_schema")
        self.assertTrue(client.response_format["strict"])
        item_schema = client.response_format["schema"]["properties"]["fields"]["items"]
        self.assertEqual(item_schema["properties"]["type"]["enum"], ["list", "enum", "boolean", "string"])
        self.assertFalse(item_schema["additionalProperties"])

    def test_usage_summary_accumulates_llm_schema_and_extraction_calls(self) -> None:
        calls = sample_call_records()[:1]
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

    def test_guardrail_returns_partial_artifact_after_extraction_errors(self) -> None:
        calls = sample_call_records()[:2]

        artifact = analyze_calls(
            calls,
            schema_inducer=StaticSchemaInducer(),
            field_extractor=FailingFieldExtractor(),
            max_errors=1,
        )

        self.assertTrue(artifact.manifest["guardrails"]["stopped"])
        self.assertEqual(artifact.manifest["guardrails"]["stop_reason"], "max_errors_exceeded")
        self.assertEqual(artifact.manifest["best_partial_result"]["extracted_field_rows"], 0)
        self.assertIn("guardrail_stop", [event.tool for event in artifact.trace])
        error_event = next(event for event in artifact.trace if event.tool == "extract_call_fields")
        self.assertEqual(error_event.validation_result, "error")
        self.assertTrue(error_event.fallback_reason)

    def test_batched_extraction_matches_sequential_and_runs_concurrently(self) -> None:
        calls = many_sample_call_records(6)
        payload = {"fields": []}

        sequential_client = ConcurrencyProbeClient(payload)
        sequential = analyze_calls(
            calls,
            schema_inducer=StaticSchemaInducer(),
            field_extractor=LLMFieldExtractor(sequential_client),
            batch_max_concurrent=1,
        )

        concurrent_client = ConcurrencyProbeClient(payload)
        concurrent = analyze_calls(
            calls,
            schema_inducer=StaticSchemaInducer(),
            field_extractor=LLMFieldExtractor(concurrent_client),
            batch_max_concurrent=4,
        )

        def signature(artifact):
            return [
                (item.call_id, item.field_name, json.dumps(item.value, sort_keys=True), item.abstained)
                for item in artifact.extractions
            ]

        self.assertEqual(signature(sequential), signature(concurrent))
        self.assertEqual(sequential_client.max_active, 1)
        self.assertGreaterEqual(concurrent_client.max_active, 2)
        extraction_events = [event for event in concurrent.trace if event.tool == "extract_call_fields"]
        self.assertEqual(len(extraction_events), len(calls))

    def test_redact_for_replay_strips_raw_text_keeps_structure(self) -> None:
        payload = {
            "chunk_id": "c1",
            "snippet": "We need SSO and audit logs.",
            "score": 0.42,
            "nested": [{"text": "raw transcript turn", "matched_terms": ["sso"]}],
            "empty": "",
        }
        redacted = redact_for_replay(payload)
        self.assertEqual(redacted["snippet"], REDACTED)
        self.assertEqual(redacted["nested"][0]["text"], REDACTED)
        self.assertEqual(redacted["chunk_id"], "c1")
        self.assertEqual(redacted["score"], 0.42)
        self.assertEqual(redacted["nested"][0]["matched_terms"], ["sso"])
        self.assertEqual(redacted["empty"], "")  # empty values are left as-is, not marked redacted
        self.assertEqual(payload["snippet"], "We need SSO and audit logs.")  # original untouched

    def test_replay_trace_artifact_written(self) -> None:
        calls = sample_call_records()[:1]
        from cu_agent_rlm.io import write_artifact

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            artifact = analyze_calls(calls, schema_inducer=StaticSchemaInducer())
            write_artifact(artifact, output)
            self.assertTrue((output / "rlm_trace.jsonl").exists())
            self.assertTrue((output / "rlm_trace.replay.jsonl").exists())

    def test_budget_unpriced_warning(self) -> None:
        keys = ("OPENAI_INPUT_USD_PER_MTOK", "OPENAI_OUTPUT_USD_PER_MTOK")
        previous = {key: os.environ.get(key) for key in keys}
        try:
            for key in keys:
                os.environ.pop(key, None)
            self.assertIsNone(budget_unpriced_warning(None))
            self.assertIsNotNone(budget_unpriced_warning(1.0))
            os.environ["OPENAI_INPUT_USD_PER_MTOK"] = "0.15"
            os.environ["OPENAI_OUTPUT_USD_PER_MTOK"] = "0.60"
            self.assertIsNone(budget_unpriced_warning(1.0))
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

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


class SchemaAwareFakeJSONClient(FakeJSONClient):
    supports_json_schema = True

    def complete_json_with_schema(self, *, system, user, response_format):
        self.response_format = response_format
        return self.complete_json(system=system, user=user)


class ConcurrencyProbeClient:
    provider_name = "probe-llm"

    def __init__(self, payload):
        self.payload = payload
        self.usage_summary = UsageSummary()
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def complete_json(self, *, system, user):
        del system, user
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self._lock:
            self.active -= 1
        return self.payload


def many_sample_call_records(count: int) -> list[CallRecord]:
    base = sample_call_records()
    records: list[CallRecord] = []
    for index in range(count):
        template = base[index % len(base)]
        records.append(
            CallRecord(
                call_id=f"call-{index:03d}",
                customer_id=template.customer_id,
                account_name=template.account_name,
                date=template.date,
                transcript=template.transcript,
                turns=template.turns,
                metadata=dict(template.metadata),
            )
        )
    return records


class FailingFieldExtractor:
    name = "failing"

    def extract_call_fields(self, call, chunks, specs):
        del call, chunks, specs
        raise ExtractionError("fixture extraction failure")


def sample_call_records() -> list[CallRecord]:
    return [
        CallRecord(
            call_id="call-001",
            customer_id="cust-001",
            account_name="Acme",
            date="2026-01-01",
            transcript="turn:1 speaker:customer: We need SSO and audit logs for security review.",
            turns=[
                TranscriptTurn(
                    call_id="call-001",
                    turn_index=1,
                    speaker="customer",
                    text="We need SSO and audit logs for security review.",
                )
            ],
            metadata={},
        ),
        CallRecord(
            call_id="call-002",
            customer_id="cust-002",
            account_name="Beta",
            date="2026-01-02",
            transcript="turn:1 speaker:customer: Pricing is blocking renewal.",
            turns=[
                TranscriptTurn(
                    call_id="call-002",
                    turn_index=1,
                    speaker="customer",
                    text="Pricing is blocking renewal.",
                )
            ],
            metadata={},
        ),
    ]


if __name__ == "__main__":
    unittest.main()
