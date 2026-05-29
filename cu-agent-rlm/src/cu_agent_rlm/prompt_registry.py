from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    version: str
    role: str
    system: str

    def render(self, user_payload: dict[str, Any]) -> "PromptRender":
        user = json.dumps(user_payload, ensure_ascii=False, indent=2)
        return PromptRender(spec=self, system=self.system, user=user)


@dataclass(frozen=True)
class PromptRender:
    spec: PromptSpec
    system: str
    user: str

    def metadata(self) -> dict[str, Any]:
        return {
            "prompt_id": self.spec.prompt_id,
            "prompt_version": self.spec.version,
            "prompt_role": self.spec.role,
            "prompt_hash": prompt_hash(self.system, self.user),
        }


def prompt_hash(system: str, user: str) -> str:
    digest = hashlib.sha256()
    digest.update(system.encode("utf-8"))
    digest.update(b"\n---USER---\n")
    digest.update(user.encode("utf-8"))
    return digest.hexdigest()[:16]


def schema_induction_prompt(max_fields: int, max_values_per_field: int) -> PromptSpec:
    return PromptSpec(
        prompt_id="cu.schema_induction",
        version="2026-05-29.1",
        role="schema_inducer",
        system=f"""You are a data-understanding schema designer.
Induce a reusable silver schema from call records. The schema must be useful for downstream search, filtering, aggregation, and evidence-backed analysis.

Return only a JSON object with this shape:
{{
  "fields": [
    {{
      "name": "snake_case_field_name",
      "type": "list | enum | boolean | string",
      "description": "what this field captures",
      "allowed_values": ["snake_case_value"],
      "downstream_use_cases": ["filtering", "aggregation"],
      "filterable": true,
      "facetable": true,
      "aggregatable": true,
      "rationale": "short evidence-backed reason"
    }}
  ]
}}

Rules:
- Do not assume a sales, support, medical, legal, recruiting, or finance domain up front; infer the domain from the data.
- Prefer fields that are reusable across many records and useful for filter/aggregate/search.
- Prefer semantically meaningful business dimensions over surface keywords.
- Do not include source metadata fields such as call_id, call_date, account_name, customer_id, source index, or call type; those already exist outside silver.
- Use list fields for multi-label concepts, enum fields for mutually exclusive labels, boolean fields for clear yes/no signals, and string fields for concise free-text values.
- Keep names and allowed values snake_case ASCII.
- Use at most {max_fields} fields and at most {max_values_per_field} allowed values per field.
- Include evidence-sensitive fields only if they can be grounded by call chunks later.
""",
    )


FIELD_EXTRACTION_PROMPT = PromptSpec(
    prompt_id="cu.field_extraction",
    version="2026-05-29.1",
    role="field_extractor",
    system="""You are a content-understanding extractor for call-record silver fields.
Return only a JSON object with this shape:
{
  "fields": [
    {
      "name": "field_name",
      "value": "typed value",
      "confidence": "low | medium | high",
      "evidence_refs": ["chunk:<chunk_id>"],
      "abstained": false,
      "rationale": "one short sentence"
    }
  ]
}

Rules:
- Extract every requested field exactly once.
- Use the field type and allowed values from the schema.
- For list fields, return a JSON array.
- For enum fields, return one allowed value.
- For boolean fields, return true or false.
- For string fields, return a concise value or "not_mentioned".
- If evidence is insufficient, set abstained=true, use an empty evidence_refs list, and return the default empty/not_mentioned value.
- Every non-abstained field must cite one or more evidence_refs from the provided chunk ids.
- Do not invent chunk ids.
- Prefer semantic judgment over keyword matching; the labels are a contract, not a list of required phrases.
""",
)
