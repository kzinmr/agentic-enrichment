from __future__ import annotations

import hashlib
import json
from typing import Any

from portable_extractor import derive_response_schema

from .prompt_registry import FIELD_EXTRACTION_PROMPT, prompt_hash


SCHEMA_INDUCTION_OUTPUT_CONTRACT_ID = "cu.schema_induction.output"
SCHEMA_INDUCTION_OUTPUT_CONTRACT_VERSION = "2026-06-02.1"


def build_extraction_contract(field_specs: list[Any], *, max_chunk_chars: int) -> dict[str, Any]:
    response_schema = derive_response_schema(field_specs)
    contract = {
        "contract_id": "cu.field_extraction.portable",
        "version": "2026-06-02.1",
        "prompt": {
            "prompt_id": FIELD_EXTRACTION_PROMPT.prompt_id,
            "prompt_version": FIELD_EXTRACTION_PROMPT.version,
            "prompt_role": FIELD_EXTRACTION_PROMPT.role,
            "prompt_ref": f"{FIELD_EXTRACTION_PROMPT.prompt_id}@{FIELD_EXTRACTION_PROMPT.version}",
            "prompt_hash": prompt_hash(FIELD_EXTRACTION_PROMPT.system, ""),
            "system": FIELD_EXTRACTION_PROMPT.system,
        },
        "artifacts": {
            "field_specs": "field_specs.jsonl",
            "chunks": "chunks.jsonl",
            "field_extractions": "field_extractions.jsonl",
            "silver_calls": "silver_calls.jsonl",
            "silver_schema_catalog": "silver_schema_catalog.json",
        },
        "runtime": {
            "module": "portable_extractor",
            "functions": ["build_user_payload", "derive_response_schema", "validate"],
            "imports_cu_agent_rlm": False,
        },
        "output_envelope": {
            "type": "object",
            "required": ["fields"],
            "field_required": ["name", "value", "confidence", "evidence_refs", "abstained", "rationale"],
            "confidence_values": ["low", "medium", "high"],
            "evidence_ref_prefix": "chunk:",
        },
        "response_format": {
            "type": "json_schema",
            "name": "cu_field_extraction",
            "description": "Validated field extraction rows for one call's chunk set.",
            "strict": True,
            "schema": response_schema,
        },
        "chunking": {
            "algorithm": "turn_window_max_chars",
            "max_chunk_chars": max_chunk_chars,
            "chunk_id_format": "{call_id}:chunk-{index:03d}",
            "external_runtime_policy": "consume chunks.jsonl as exported; do not re-chunk transcripts",
        },
        "aggregation": {
            "contract_id": "qu.aggregate_silver.expression",
            "version": "2026-06-02.1",
            "scope": "allowlisted expressions over exported silver_calls fields only",
            "allowed_functions": [
                "count",
                "group_count",
                "top_k",
                "nested_group_count",
                "count_if",
                "count_where",
                "numeric_range_count",
                "date_range_count",
                "cohort_count",
                "ratio",
            ],
            "field_policy": "expression field references must exist in silver_schema_catalog.json and be marked search.aggregatable=true",
        },
    }
    return {**contract, "contract_hash": contract_hash(contract)}


def schema_induction_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "cu_schema_induction",
        "description": "Reusable silver-field schema induced from call records.",
        "strict": True,
        "schema": schema_induction_output_schema(),
    }


def schema_induction_output_contract() -> dict[str, Any]:
    contract = {
        "contract_id": SCHEMA_INDUCTION_OUTPUT_CONTRACT_ID,
        "version": SCHEMA_INDUCTION_OUTPUT_CONTRACT_VERSION,
        "response_format": schema_induction_response_format(),
    }
    return {**contract, "contract_hash": contract_hash(contract)}


def schema_induction_output_metadata() -> dict[str, Any]:
    contract = schema_induction_output_contract()
    return {
        "contract_id": contract["contract_id"],
        "contract_version": contract["version"],
        "contract_hash": contract["contract_hash"],
        "response_format_name": contract["response_format"]["name"],
        "strict": contract["response_format"]["strict"],
    }


def schema_induction_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["fields"],
        "properties": {
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "name",
                        "type",
                        "description",
                        "allowed_values",
                        "downstream_use_cases",
                        "filterable",
                        "facetable",
                        "aggregatable",
                    ],
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": ["list", "enum", "boolean", "string"]},
                        "description": {"type": "string"},
                        "allowed_values": {"type": "array", "items": {"type": "string"}},
                        "downstream_use_cases": {"type": "array", "items": {"type": "string"}},
                        "filterable": {"type": "boolean"},
                        "facetable": {"type": "boolean"},
                        "aggregatable": {"type": "boolean"},
                    },
                },
            }
        },
    }


def contract_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
