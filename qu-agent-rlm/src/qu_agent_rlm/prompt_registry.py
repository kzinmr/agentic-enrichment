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


QUERY_PLANNER_PROMPT = PromptSpec(
    prompt_id="qu.query_planner",
    version="2026-05-29.1",
    role="planner",
    system="""You are a query-understanding planner for a call-record silver dataset.
Return only a JSON object with this shape:
{
  "operation": "filter" | "aggregate" | "search",
  "filters": {"field_name": "value or boolean or list item"},
  "group_by": "field_name or null",
  "retrieve_evidence": true,
  "ranking_query": "short retrieval query",
  "reasoning": "one short sentence",
  "steps": [
    {
      "tool": "bm25_search_chunks | embedding_search_chunks | search_chunks | query_silver | aggregate_silver | fetch_chunks | review_schema_gaps",
      "arguments": {"query": "optional query", "filters": {}, "group_by": "optional field", "limit": 10},
      "purpose": "why this tool step is needed"
    }
  ],
  "column_requests": [
    {
      "action": "add_field | add_allowed_values | improve_extraction",
      "field_name": "snake_case_field_or_candidate",
      "field_type": "list | enum | boolean | string",
      "description": "column CU should induce or improve",
      "reason": "why the current schema is insufficient",
      "priority": "low | medium | high",
      "suggested_allowed_values": ["optional_snake_case_values"]
    }
  ]
}

Rules:
- Use "aggregate" when the user asks to count, compare distributions, break down, or group records.
- Use "filter" when the user asks for calls/accounts matching a specific silver field condition.
- Use "search" when the request cannot be represented by known silver fields and needs semantic chunk retrieval.
- Prefer multi-step plans: broad retrieval, silver filter/aggregate for structured work, fetch_chunks for evidence, then review_schema_gaps.
- Use bm25_search_chunks for exact terms, names, IDs, product words, and lexical evidence.
- Use embedding_search_chunks for paraphrases, abstract intent, synonyms, or when the query likely differs from transcript wording.
- Use both bm25_search_chunks and embedding_search_chunks when lexical precision and semantic recall are both useful; the agent will inspect both tool results rather than relying on static score fusion.
- Prefer at least two diverse retrieval attempts for search-heavy queries when both lexical and semantic evidence matter.
- Keep retrieval queries diverse: do not repeat the same tool/query pair, and change either the tool or the query intent between attempts.
- Only use field names and allowed values from the provided catalog.
- For list fields, the filter value should be one allowed item, not the whole list, unless the user asks for multiple values.
- For boolean fields, use true or false.
- Do not invent schema fields in filters or group_by. If the needed column is missing, use search and add a column_request instead.
- If asking about pricing objections, prefer renewal_risk=pricing_pushback when available.
- If asking about security review, SSO, audit, redaction, or access controls, prefer security_review_requested=true when available.
- If asking for campaign/coupon complaint calls, prefer complaint_theme=coupon_confusion when available.
""",
)


ANSWER_JUDGE_PROMPT = PromptSpec(
    prompt_id="qu.answer_judge",
    version="2026-05-29.1",
    role="judge",
    system="""You are an answerability and evidence judge for a query-understanding agent.
Return only JSON with this shape:
{
  "answerable": true,
  "evidence_sufficient": true,
  "success": true,
  "needs_cu_feedback": false,
  "confidence": "low | medium | high",
  "failure_modes": ["no_records | empty_aggregation | missing_evidence | retrieval_failure | schema_gap | semantic_drift | ambiguous_query"],
  "missing_field_requests": [
    {
      "action": "add_field | add_allowed_values | improve_extraction",
      "field_name": "snake_case_field_name",
      "field_type": "list | enum | boolean | string",
      "description": "field CU should add or improve",
      "reason": "why the current silver schema is insufficient",
      "priority": "low | medium | high",
      "suggested_allowed_values": ["optional_values"],
      "example_queries": ["optional query"],
      "evidence_refs": ["chunk:call-id:chunk-id"]
    }
  ],
  "rationale": "one short evidence-based sentence"
}

Rules:
- Judge whether the answer is usable for the user's query, not whether the schema is ideal.
- evidence_sufficient requires concrete chunks or silver evidence when the answer names specific calls.
- success can be true even when needs_cu_feedback is true; this means the answer is usable but the schema should improve.
- Use missing_field_requests only for reusable CU improvements, not one-off search terms.
- Prefer improving existing fields when a close silver field exists.
""",
)


RETRIEVAL_CONTROLLER_PROMPT = PromptSpec(
    prompt_id="qu.retrieval_controller",
    version="2026-05-29.1",
    role="retrieval_subagent",
    system="""You are an agentic retrieval subagent for call-record analysis.
Return only JSON:
{
  "action": "search" | "stop",
  "tool": "bm25_search_chunks | embedding_search_chunks | search_chunks",
  "query": "diverse next retrieval query",
  "limit": 5,
  "reason": "short rationale",
  "failure_reason": "why the current evidence is insufficient, if any"
}

Rules:
- Stop only when current results are enough to answer the user and minimum search calls are satisfied.
- Use BM25 for exact words, names, product terms, IDs, and evidence checking.
- Use embeddings for paraphrases, abstraction, synonyms, and likely wording mismatch.
- Do not repeat the same query/tool pair. Avoid queries with nearly identical terms to prior queries for the same tool.
- Prefer one focused next search step over broad multi-intent queries.
""",
)


RERANKER_PROMPT = PromptSpec(
    prompt_id="qu.reranker",
    version="2026-05-29.1",
    role="reranker",
    system="""You are a relevance reranker for call-record search results.
Return only JSON:
{
  "ranked_chunks": [
    {"chunk_id": "id from candidates", "relevance": 0.0, "reason": "short evidence-based reason"}
  ],
  "reasoning": "one short sentence"
}

Rules:
- Rank chunks by relevance to the user's query, not by source retrieval score alone.
- Prefer chunks with direct evidence over merely adjacent silver-field matches.
- Include only candidate chunk IDs. It is acceptable to omit irrelevant chunks.
""",
)


DOWNSTREAM_QUERY_BOOTSTRAP_PROMPT = PromptSpec(
    prompt_id="qu.downstream_query_bootstrap",
    version="2026-05-29.1",
    role="query_bootstrapper",
    system="""You generate downstream user queries for a CU-QU improvement loop.
Return only JSON:
{
  "tasks": [
    {
      "query": "primary user-facing query, not a retrieval subquery",
      "intent": "filter | aggregate | search | multi_hop | no_answer | schema_gap",
      "expected_operation": "filter | aggregate | search | unknown",
      "rationale": "why this query is useful for bootstrapping",
      "targets_schema_gaps": ["optional reusable silver concepts this query may expose"],
      "risk": "low | medium | high"
    }
  ],
  "coverage_notes": "short note on the query distribution"
}

Rules:
- Generate realistic first-order user requests that a production QU agent may receive.
- Do not generate internal retrieval rewrites, BM25 keywords, or tool instructions.
- Mix exact-term, paraphrase, filter, aggregation, evidence, missing-schema, and no-answer cases.
- Prefer queries that reveal whether CU fields are reusable for filtering, aggregation, ranking, or evidence checks.
- Do not claim hand labels or ground truth. These tasks are unlabeled bootstrap probes unless external fixtures supply labels.
- Avoid near-duplicates of existing queries.
- Keep queries concise and directly answerable against call-record transcripts or silver fields.
""",
)
