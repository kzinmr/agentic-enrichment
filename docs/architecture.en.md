# Agent-Native RLM Architecture

A consolidated design view of the self-improving loop built around `cu-agent-rlm` + `qu-agent-rlm`, operating over Databricks `call_records` in an **RLM (Retrieval-based LM) style**.

This document is a retrospective consolidation of an incremental construction log spanning 2025-2026. It frames the entire prototype as three first-class states (Catalog / Prompts / Queries) layered with their tool boundary, promotion gates, and observability surfaces.

---

## 1. Overview

Two independent packages are chained by a one-way contract; QU pushes a reverse-channel feedback back into CU, forming what was originally a **half-closed loop and is now a full loop**.

```
call_records.sql ──┐
sample_calls.jsonl ┤→ [cu-agent-rlm]  →  silver/manifest/chunks/catalog  →  [qu-agent-rlm]  →  answer + judgement
                                                ↑                                                       │
                                                │                                                       ↓
                                                └─── prompt_repair_request.jsonl ←──────────────────────┤
                                                └─── column_requests.jsonl     ←───────────────────────┤
                                                └─── query_tasks.jsonl         ←───────────────────────┘
```

The core design principle is **"never pass raw transcripts to the root agent."** Instead:

- Expose **silver fields** — structured, lightweight, typed values
- Resolve evidence on demand through **`evidence_refs` (`chunk:<chunk_id>`)**
- Wrap every LLM call with a **promotion gate / validator / fallback**

---

## 2. Design Philosophy: RLM Tool Boundary

The agent can only call a fixed set of five tools (QU's retrieval is subdivided into three, making effectively seven step-tools). `call_records` itself never enters an LLM prompt directly.

| Tool | Purpose | Production mapping |
|---|---|---|
| `search_chunks` / `bm25_search_chunks` / `embedding_search_chunks` | Per-chunk retrieval | OpenSearch / Databricks Vector Index |
| `fetch_chunks` | Resolve real text from `evidence_ref` | Access-controlled snippet API |
| `query_silver` | Filter by silver fields | Materialized silver view |
| `aggregate_silver` | Aggregate on facetable fields | Same |
| `run_sql` | Databricks SELECT (contract only; not yet implemented) | Databricks SQL |
| `review_schema_gaps` | QU meta-tool: structures unanswerable shapes | — |

QU's inner loop adds a **search controller** and **reranker** as LLM-driven tools (see below).

---

## 3. Three Pillars (First-Class State)

Across CU/QU, there are exactly three mutable kinds of state, all with an identical lifecycle pattern.

| Pillar | Identity | Storage | Proposal channel | Promotion gate | Status |
|---|---|---|---|---|---|
| **Catalog (silver_schema)** | `field_name + schema_version` | `silver_schema_catalog.json` | `column_requests.jsonl` | `evidence_coverage >= 0.75` | Auto-promotion active |
| **Prompts** | `prompt_id + version` | `prompt_registry.py` (`PromptSpec(frozen=True)`) | `prompt_repair_request.jsonl` | `external_hand_labeled_eval_required` | **Blocked on EVAL-001** |
| **Queries** | `task_id + generation_id` | `query_tasks.jsonl` | `query_bootstrap.json` + production log | `label_status: hand_labeled` required | Unlabeled used only for exploration |

### 3.1 Catalog

- `silver_schema_catalog.json` is the source of truth for the schema
- Each field carries `filterable / facetable / aggregatable` flags, `allowed_values`, `evidence.ref_path`, and `quality` metadata
- Provenance is recorded as `manifest.schema_induction.inducer` (e.g. `openai`, `openai+feedback`); fallback details are tracked separately via validation errors / trace
- Promotion condition: `support_count > 0 AND evidence_coverage >= 0.75`

### 3.2 Prompts

Every LLM call goes through a versioned `PromptSpec`.

```python
@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str       # "qu.query_planner"
    version: str         # "2026-05-29.1"
    role: str            # "planner" | "judge" | "extractor" | ...
    system: str
```

Defined prompts (all at version `2026-05-29.1`):

| prompt_id | role | location |
|---|---|---|
| `qu.query_planner` | planner | `qu/prompt_registry.py` |
| `qu.retrieval_controller` | retrieval_subagent | same |
| `qu.reranker` | reranker | same |
| `qu.answer_judge` | judge | same |
| `qu.downstream_query_bootstrap` | query_bootstrapper | same |
| `cu.schema_induction` | inducer | `cu/prompt_registry.py` |
| `cu.field_extraction` | extractor | same |

`frozen=True` makes **automatic runtime mutation type-impossible**. Changes require a new `PromptSpec` instance (i.e. a version bump).

### 3.3 Queries

Primary queries that feed the CU-QU loop are also first-class artifacts.

```json
{
  "task_id": "synthetic_llm-001-a1b2c3d4",
  "query": "Which calls mention founder-led sales calls?",
  "intent": "schema_gap",
  "source_type": "curated | production_log | failed_query | synthetic_llm | synthetic_heuristic | hand_labeled_eval | manual",
  "label_status": "unlabeled | weak_label | hand_labeled",
  "split": "bootstrap | dev | eval",
  "expected_operation": "filter | aggregate | search | unknown",
  "weight": 1.0,
  "generation_id": "qgen-...",
  "adjusted_for": {"loop_id": "...", "iteration": 0, "schema_hash": "...", "schema_field_names": [...]},
  "provenance": {"generator": "openai", "model": "gpt-5.4-mini", "prompt": {prompt_id, version, hash}}
}
```

How each `source_type` is treated by promotion gates:

- `hand_labeled_eval` / `production_log` → primary signal for prompt/schema improvement
- `failed_query` / `curated` → schema improvement only
- `synthetic_llm` → exploration only; not eligible for promotion
- `synthetic_heuristic` → smoke-test / legacy helper only; not eligible for promotion

---

## 4. CU Pipeline

### 4.1 Seven stages

```
load_calls
  → build_chunks
  → induce_schema  (+ feedback layer)
  → search_chunks (per-spec, for candidate context)
  → extract_call_fields  (per call, sub-rlm)
  → quality_gate (evidence_coverage)
  → promote_fields
  → materialize_silver_calls
  → publish (manifest + catalog + silver + trace + feedback_report)
```

### 4.2 LLM-first Schema Inducer + Feedback Decorator

| Inducer | Role |
|---|---|
| `LLMSchemaInducer` (OpenAI / openai-compatible) | First-class decision maker in demo and production. The LLM proposes field names, types, and values; `validate_schema_payload` enforces shape constraints. |
| `HeuristicSchemaInducer` | Reliability fallback / smoke-test helper. Emits a minimum-viable contract when the LLM fails; not positioned as a substitute intelligence. |
| `StaticSchemaInducer` | Direct-injection helper for the legacy `FIELD_SPECS`. Not a CLI/demo mode. |

`FeedbackAwareSchemaInducer` wraps any of these and merges `column_requests.jsonl` content as `allowed_values` extensions or new field proposals. The manifest records this with a composed provenance string such as `"openai"` or `"openai+feedback"`. Detailed fallback provenance is being routed through SYS-001 into trace/budget reporting.

The LLM induction validator enforces:

- `RESERVED_FIELD_NAMES` (`call_id, customer_id, date, ...`) cannot be invaded
- snake_case is enforced; enums must begin with `not_mentioned`
- `list/enum` with empty `allowed_values` are dropped
- `string` fields are excluded from aggregation by default (`facetable = aggregatable = False`)
- `max_fields=12`, `max_values_per_field=12` cap vocabulary explosion

### 4.3 Extractor

| Extractor | Behavior |
|---|---|
| `LLMFieldExtractor(fallback=Heuristic)` | Standard for demo and production. One LLM call per record returns all fields; if the call fails, only that call falls back to heuristic. |
| `HeuristicFieldExtractor` | Reliability fallback / smoke-test helper. Combines legacy `LIST_RULES` with `generic_extract_field` for unknown fields. |

LLM output validator:

- `evidence_refs` are **constrained to the `chunk_id` set shown to the LLM** (anti-hallucination)
- Invalid values are coerced to a default and recorded in `validation_errors`
- `validation_errors` propagates through `quality_flags` into `silver_calls.jsonl`

### 4.4 Promotion Gate

```python
recommended_status = "promote" if support_count > 0 and evidence_coverage >= 0.75 else "hold"
```

`evidence_coverage = supported_with_refs / supported_total`. Producing a value is not enough — **a field without `evidence_refs` cannot reach silver**. This is a structural filter against hallucination.

### 4.5 Feedback Consumption

`--feedback-input <jsonl>` lets the next induction consume QU feedback. `summarize_feedback` runs a three-tier fallback aggregator:

1. Merge `column_requests` (same names are merged while preserving priority/count)
2. If empty, synthesize via `request_from_failures` using `search_diagnostics.failures`
3. If still empty, synthesize via `request_from_judgement` using `judgement.needs_cu_feedback`

`feedback_report.json` is emitted with each field's `accepted_into_schema` / `promoted_to_silver` state and `validation_error_counts`.

---

## 5. QU Pipeline

### 5.1 Multi-step executor

```
load_schema
  → plan_query  (LLM-first; provenance: plan_query:<planner_name>)
  → [steps executed sequentially, accumulating state]
       ├─ search_chunks / bm25_search_chunks / embedding_search_chunks
       ├─ query_silver
       ├─ aggregate_silver
       ├─ fetch_chunks
       └─ review_schema_gaps
  → run_search_iterations  (ReAct-style loop only after search steps)
  → rerank_search_results  (LLM reranker, optional)
  → answer_judge           (LLM-first; heuristic only as validation-failure fallback)
```

Default step sequences per operation:

| operation | step sequence |
|---|---|
| `aggregate` | `bm25_search → query_silver → aggregate_silver → fetch_chunks → review_schema_gaps` (5 steps) |
| `filter` | `query_silver → bm25_search → fetch_chunks → review_schema_gaps` (4 steps) |
| `search` | `bm25_search(promote_records=True) → fetch_chunks → review_schema_gaps` (3 steps) |

`validate_steps` enforces **invariants on operation → tool ordering** (e.g. `query_silver` is auto-inserted before `aggregate_silver` when missing; `fetch_chunks` is auto-appended when `retrieve_evidence=True`).

### 5.2 Retrieval backend

`SilverCorpus` exposes four methods:

| Method | Implementation |
|---|---|
| `lexical_overlap_search_chunks` | Legacy term-overlap, only via `--retrieval-mode lexical` |
| `bm25_search_chunks` | **bm25s (numpy-only)** + a pure-Python `FallbackBM25Index` |
| `embedding_search_chunks` | OpenAI `text-embedding-3-small` + cosine, cached at `<corpus>/.qu_agent_rlm/<model>.json` |
| `hybrid_search_chunks` | Rank-position zipper merge (kept for backwards compat; real fusion lives in the agent-state layer) |

#### BM25F-lite document composition

```python
def compose_bm25_text(chunk_text, field_text, account_text):
    # Global IDF; chunk_text x2 approximates field weighting via tf-bag duplication
    return "\n".join([chunk_text, chunk_text, field_text, account_text])
```

`field_text` only includes `catalog.fields[*].search.filterable=True`. The embedding side uses a different composition:

```python
return f"account: {account_text}\nsilver_fields: {field_text}\ntranscript_chunk: {chunk_text}"
```

**BM25 explainability**: every BM25 result returns `query_terms / matched_terms / missing_terms / score_details.terms[{term, tf, df, idf, contribution}]`. The LLM controller / reranker / judge receive this breakdown.

#### Embedding cache invalidation

`documents_hash = SHA256(model + sorted[(chunk_id, embedding_text)])` manages the hash. Any change to chunk text, catalog `filterable` fields, silver values, or model name produces a hash mismatch and triggers automatic re-embedding.

### 5.3 Retrieval Subagent

`AgenticRetrievalSubAgent` supersedes `--search-controller` (the old flag is Deprecated). With `--retrieval-subagent openai-compatible --llm-base-url <sid1-endpoint>`, SID-1-style retrieval models can be plugged in.

```python
class SearchExecutionPolicy:
    min_calls: int = 1
    max_iterations: int = 0
    query_diversity_threshold: float = 0.8
```

`--retrieval-mode agentic` automatically upgrades to `min_calls=2`, `max_iterations=2`. `validate_search_diversity`:

- Rejects identical `(tool, normalize_query)` pairs
- Rejects intra-tool Jaccard term overlap > 0.8

On rejection, `forced_search_iteration` constructs a `diversified_query` by **converting prior `missing_terms` into the next query**. BM25 explainability turns "what we missed" into actionable exploration.

### 5.4 Reranker

`rerank_search_results` runs an LLM rerank after all steps complete. It rewrites `state["chunks"]` and attaches `rerank_score` / `rerank_reason`. On failure, ordering is preserved and only `llm_rerank:error` is appended to the trace.

### 5.5 Answerability Judge

```python
{
  "judge": "openai | openai-compatible | <provider>->fallback:heuristic",
  "answerable": bool,
  "evidence_sufficient": bool,
  "success": bool,
  "needs_cu_feedback": bool,
  "confidence": "low | medium | high",
  "failure_modes": ["no_records" | "empty_aggregation" | "missing_evidence" |
                    "retrieval_failure" | "schema_gap" | "search_only_schema_gap" |
                    "semantic_drift" | "ambiguous_query"],
  "missing_field_requests": [<ColumnRequest>],
  "rationale": "...",
  "metrics": {record_count, evidence_count, ..., embedding_call_count, reranked_chunk_count}
}
```

The LLM judge is the always-on default. `HeuristicAnswerJudge` is a reliability fallback for JSON validation / API failures and a smoke-test helper — it is not an agent intelligence mode exposed in CLI/demo. The LLM judge can use a different model from the planner via `--judge-model / --judge-base-url / --judge-api-key` (an operational path for avoiding self-bias / collusion).

---

## 6. CU↔QU Feedback Loop

### 6.1 Loop shape

```
[Query Source Pool]
  ├ curated         (candidate for hand-labeled eval)
  ├ production_log  (production traffic)
  ├ failed_query    (derived from past column_requests)
  ├ synthetic_llm   (via qu.downstream_query_bootstrap)
  └ synthetic_heuristic (smoke-only helper)
            ↓ dedupe + adjusted_for snapshot
[query_tasks.jsonl] (first-class artifact)
            ↓ N queries replayed
[QU agent] × N
            ↓ each answer carries judgement
[column_requests + search_diagnostics + judgement.missing_field_requests + prompt_repair_request]
            ↓ append-only JSONL
[CU next run: --feedback-input]
            ↓ summarize_feedback (3-tier fallback aggregator)
[FeedbackAwareSchemaInducer]
            ↓ apply_feedback_to_specs
[catalog v2]   marked with "QU feedback refinement" provenance
            ↓ extract + promotion gate
[silver v2]
            ↓ accepted/promoted state recorded in feedback_report.json
[QU re-run] replays the same query_tasks.jsonl
            ↓
[orchestration_trace.jsonl] aggregates before/after
```

### 6.2 Three reverse channels

1. **`column_requests`** — emitted directly by QU's planner / `review_schema_gaps` step
2. **`search_diagnostics.failures`** — auto-generated from BM25 explainability's `missing_terms`
3. **`judgement.missing_field_requests`** — the answer judge autonomously proposes CU improvements

`summarize_feedback` priority-merges all three channels; same-name fields accumulate via `count`.

### 6.3 Reliability fallback's feedback-aware filter detection

`FeedbackAwareSchemaInducer` marks fields it adds with `usage.good_for: "QU feedback refinement"`. `detect_feedback_field_filters` detects this provenance, so even when the LLM planner fails and the heuristic fallback kicks in, runtime-added feedback columns can still be promoted to filter candidates by generic term overlap. This is a reliability layer keeping the feedback loop observable and continuous through LLM failures — it is not the primary path in demo/production.

---

## 7. Orchestration Layer

### 7.1 `scripts/qu_cu_loop_demo.py`

Runs the full loop in one command:

```bash
python3 scripts/qu_cu_loop_demo.py --output-root output/qu_cu_loop_demo
```

Seven phases (an entire loop is bound by `loop_id`, with `iteration` distinguishing reruns):

| phase | agent | event_type |
|---|---|---|
| `query_bootstrap` | qu | `query_tasks.created` |
| `baseline_cu` | cu | `cu.run_completed` |
| `baseline_qu` | qu | `qu.run_completed` |
| `feedback` | qu | `qu.feedback_emitted` |
| `refined_cu` | cu | `cu.run_completed` |
| `refined_qu` | qu | `qu.run_completed` |
| `loop_evaluation` | orchestrator | `loop.evaluated` |

### 7.2 Output artifacts

```
output/qu_cu_loop_demo/
├── 01_cu_baseline/
│   ├── manifest.json
│   ├── silver_schema_catalog.json
│   ├── silver_calls.jsonl
│   ├── chunks.jsonl
│   ├── rlm_trace.jsonl
│   └── feedback_report.json
├── 02_qu_feedback/
│   ├── query_tasks.jsonl       ← set of primary queries
│   ├── query_bootstrap.json    ← audit of the generation process
│   ├── baseline_answers/
│   │   └── <task_id>.json      ← per-query answer + judgement + trace
│   └── column_requests.jsonl   ← reverse channel from QU
├── 03_cu_refined/
│   └── (same shape as 01; feedback_report.json carries accepted/promoted state)
├── 04_qu_refined/
│   └── refined_answers/<task_id>.json
├── loop_summary.json           ← before / feedback / schema_delta / after / loop_result
└── orchestration_trace.jsonl   ← unified event stream of the 7 phases
```

---

## 8. Observability Stack

| artifact | role | main content |
|---|---|---|
| `manifest.json` | CU root entry | dataset_id, record/chunk count, `schema_induction`, `feedback_refinement`, `prompt_state` |
| `silver_schema_catalog.json` | QU's query contract | fields[].search/evidence/usage/quality |
| `silver_calls.jsonl` | Structured values + evidence_refs | fields, evidence_refs, quality_flags |
| `chunks.jsonl` | Retrievable chunks | chunk_id, text, snippet |
| `rlm_trace.jsonl` | CU internal trace | root / sub-rlm tool events |
| `feedback_report.json` | QU→CU feedback result | accepted_into_schema, promoted_to_silver, validation_error_counts |
| `evaluation_tasks.json` | CU self-generated smoke tasks | Self-referential; not an external fixture |
| `databricks_contract.json` | Production wiring contract | allowlisted_tools, sql_policy |
| QU answer JSON | Per-query result | answer, plan, records, aggregation, evidence, judgement, column_requests, search_diagnostics, rerank, prompt_state, trace |
| `column_requests.jsonl` | QU → CU reverse channel | Append-only; ships with search_diagnostics and judgement |
| `prompt_repair_request.jsonl` | Prompt improvement signal | **signal-only**, `action: collect_signal_only`, `promotion_gate: external_hand_labeled_eval_required` |
| `query_tasks.jsonl` | Primary query set | task_id, source_type, label_status, generation_id, adjusted_for, provenance |
| `query_bootstrap.json` | Synthetic query generation audit | generator, model, prompt metadata, coverage_notes |
| `orchestration_trace.jsonl` | Unified event stream for the outer loop | loop_id, iteration, phase, agent, metrics, decision |

---

## 9. Shared Pattern: LLM-first Pluggable Component

Nearly every semantic component is implemented behind the same `JSONChatClient` Protocol, and the CLI/demo only expose LLM backends as user-facing modes. Heuristic implementations are not substitute intelligences — they are reliability fallbacks behind validators/API failures, and helpers for direct injection in smoke tests.

| Component | User-facing modes | Flag |
|---|---|---|
| Schema inducer | openai / openai-compatible | `--schema-inducer` |
| Field extractor | openai / openai-compatible | `--extractor` |
| Query planner | openai / openai-compatible | `--planner` |
| Retrieval subagent | default / none / openai / openai-compatible | `--retrieval-subagent` |
| Reranker | none / openai / openai-compatible | `--reranker` |
| Answer judge | openai / openai-compatible | `--judge` |
| Query bootstrapper | openai / openai-compatible | `--bootstrap-queries` |

`auto` and no-key demo fallback have been retired. `--retrieval-mode agentic`'s `default` retrieval subagent uses an OpenAI controller. `none` remains for optional control planes (retrieval/reranker) but is **not** allowed for semantic decisions (planner / schema / extractor / judge / query bootstrap).

Every LLM path carries:

- input validators (catalog integrity / chunk_id whitelist / allowed_values checks)
- fallback orchestration (`<provider>->fallback:<base>` provenance + validation errors retained)
- `prompt_id + version + hash` trace records

---

## 10. Promotion Gates vs. Exploration

### 10.1 Design principle

**"Policy parameters are explored; safety contracts are held by humans"** — split across three layers:

| Layer | Examples | Auto-explore? |
|---|---|---|
| Observation space | catalog fields | ✅ (auto-promoted by evidence_coverage gate) |
| Policy parameters | prompts, query distribution | ⚠ Proposals accumulate; promotion requires external fixture |
| Safety contracts | validator rules, the `PROMOTION_GATE` constant, the `evidence_coverage >= 0.75` threshold | ❌ Code only; not auto-mutated |

### 10.2 The `PROMOTION_GATE` constant

```python
PROMOTION_GATE = "external_hand_labeled_eval_required"
```

Shared as the same string constant by `qu/prompt_repair.py` and `cu/prompt_repair.py`. It is baked into the `promotion_gate` field of every `prompt_repair_request`, so **"why this is not yet promoted" is written into the data itself.**

### 10.3 Bootstrap early-stopping strategy

From FB-002's design direction:

> Prefer early stopping during bootstrap: use self-improvement for 70-80 point startup quality, then rely on production query/outcome logs and external fixtures for higher-confidence prompt/model changes.

A two-stage policy: self-improvement carries bootstrap to 70-80% quality, after which **production query/outcome logs** and **EVAL-001's external fixtures** take over.

### 10.4 Judge model separation

`--judge-model / --judge-base-url / --judge-api-key / --judge-timeout-seconds` (env vars `JUDGE_OPENAI_*`) let the judge use a model distinct from the planner. The intended operational path is **shared during bootstrap, separated in production**.

---

## 11. Issue Backlog (Current)

Summary of `cu-agent-rlm/docs/issues.md`:

| ID | Title | Status |
|---|---|---|
| CU-001 | LLM-backed field extraction with schema/evidence validation | Done |
| CU-002 | Open-world schema discovery | Done |
| **SYS-001** | **Heuristic reliability layer and mode hygiene** | **Done** |
| CU-003 | Typed ontology policy (strict enum vs extensible vocab) | Open |
| CU-004 | QU-to-CU schema refinement loop | Done |
| CU-005 | Extraction quality calibration and adjudication | Open |
| QU-001 | Multi-step query plans | Done |
| QU-002 | Agentic search iteration, query diversity, reranking | Done |
| QU-003 | Query language expressiveness | Open |
| RET-001 | Semantic retrieval backend (BM25 + embedding) | Done |
| AGG-001 | Rich aggregation expressions | Open |
| EVID-001 | Span-level evidence and privacy filtering | Open |
| **EVAL-001** | **External evaluation fixtures** | **Open (prerequisite for FB-002)** |
| DBX-001 | Databricks production tool adapters | Open |
| OBS-001 | Budget, observability, replay | Partial (skeleton via orchestration_trace; latency/cost not yet captured) |
| FB-001 | Model-feedback and prompt-improvement loop | Partial (schema improvement done) |
| **FB-002** | **Prompt lab and eval-gated prompt promotion** | **Blocked on EVAL-001** |
| DE-001 | CU bootstrap context pack | Design enhancement |
| DE-002 | CU-QU schema negotiation loop | Design enhancement |
| DE-003 | Programmatic RLM sub-agent orchestration | Partial |
| DE-004 | Feedback routing beyond schema changes | Design enhancement |
| DE-005 | Semantic recall evaluation and drift control | Design enhancement |
| DE-006 | Schema lifecycle, backfill, and governance | Design enhancement |
| DE-007 | Human review and policy hooks for high-impact inferences | Design enhancement |

**Critical path**: EVAL-001 (external hand-labeled fixtures) unblocks FB-002, which then enables automatic prompt promotion. Until then, every prompt change is manual, and every synthetic query stays at `label_status="unlabeled"` outside the promotion gate.

---

## 12. Design Principles

1. **Tool boundary is fixed at five tools.** Raw transcripts never reach an LLM. All references go through `chunk:<id>`.
2. **All three pillars share `versioned identity + promotion gate`.** Catalog / prompts / queries follow the same lifecycle.
3. **Every LLM path carries validator + fallback + provenance.** Trace shows `<provider>->fallback:<base>`; prompt_id + hash; composite provenance strings like `<base>+feedback`.
4. **`frozen=True` is a type-level guarantee against auto-mutation.** The only way to change a prompt is to write a new instance (a version bump).
5. **Bake `PROMOTION_GATE` into the data.** "Why we don't promote" is recorded in the artifact itself.
6. **Operationally separate self-eval bias.** Judge ↔ planner model separation flags; synthetic queries stay outside the promotion gate.
7. **LLM-first + reliability fallback.** Semantic decisions belong to the LLM; heuristics only catch validation/API failures and support smoke tests.
8. **Append-only signal logs.** `column_requests.jsonl`, `prompt_repair_request.jsonl`, `orchestration_trace.jsonl` accumulate over time and are aggregated downstream.
9. **Keep deterministic control.** BM25 explainability, query diversity, forced search iteration, and smoke eval are retained as reliability infrastructure.
10. **Observability via structured JSON.** Trace tool names expand backend / provider / fallback chain into strings.

---

## Appendix A: File-level responsibilities

### cu-agent-rlm

| File | Responsibility |
|---|---|
| `pipeline.py` | Orchestrates the seven stages |
| `models.py` | Dataclass definitions (CallRecord, Chunk, FieldSpec, FieldExtraction, SilverCallRecord, TraceEvent, ContentUnderstandingArtifact) |
| `io.py` | JSONL/CSV load + writing all artifacts |
| `chunking.py` | Builds chunks at `max_chars=900` |
| `fields.py` | Legacy `FIELD_SPECS` and `LIST_RULES` (compat helpers for `StaticSchemaInducer`) |
| `schema.py` | `SchemaInducer` Protocol + 3 implementations |
| `extraction.py` | `FieldExtractor` Protocol + 2 implementations + validator |
| `feedback.py` | `FeedbackAwareSchemaInducer` + `summarize_feedback` + `build_feedback_report` |
| `prompt_registry.py` | `PromptSpec(frozen=True)` + CU-side prompt definitions |
| `prompt_repair.py` | Emits `prompt_repair_request.jsonl` (signal-only) |
| `llm.py` | `OpenAIResponsesClient` / `OpenAICompatibleChatClient` |
| `cli.py` | CLI argparse + component factories |

### qu-agent-rlm

| File | Responsibility |
|---|---|
| `agent.py` | `QueryUnderstandingAgent` (multi-step executor) |
| `corpus.py` | `SilverCorpus` (four retrieval methods) |
| `retrieval.py` | bm25s wrapper + `FallbackBM25` (explainer) + `EmbeddingIndex` + BM25F-lite composition |
| `retrieval_agent.py` | `AgenticRetrievalSubAgent` (controller + reranker + diversity check) |
| `planner.py` | `QueryPlan` / `QueryToolStep` / `ColumnRequest` + Heuristic/LLM planner + validator for the 7 `ALLOWED_STEP_TOOLS` |
| `judge.py` | `AnswerJudge` Protocol + Heuristic / LLM / Noop |
| `query_tasks.py` | `QueryTask` normalization + `LLMDownstreamQueryGenerator` + smoke-test heuristic bootstrap helper |
| `prompt_registry.py` | `PromptSpec(frozen=True)` + QU-side prompt definitions |
| `prompt_repair.py` | Emits `prompt_repair_request.jsonl` (signal-only) |
| `llm.py` | LLM client definitions (kept duplicated with CU to preserve dependency isolation) |
| `cli.py` | CLI argparse + planner/judge/subagent/reranker factories |
| `env.py` | `.env` file loader |
| `eval.py` | Smoke runner for `evaluation_tasks.json` |

### scripts

| File | Responsibility |
|---|---|
| `qu_cu_loop_demo.py` | Outer-loop orchestration runner; full run including `query_tasks` + `orchestration_trace` |
| `qu_user_query_demo.py` | Single user-query demo (defaults to agentic retrieval + OpenAI planner/reranker/judge) |

---

**Last updated**: 2026-05-30 / **schema_version**: `silver.rlm.v1` / **prompt baseline**: `2026-05-29.1`
