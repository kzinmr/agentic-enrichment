# agent-enrichment

A working prototype of an **agent-native RLM (Retrieval-based LM) stack** built around `call_records` table. Two Python packages — `cu-agent-rlm` (Content Understanding) and `qu-agent-rlm` (Query Understanding) — exchange catalog, evidence, and feedback through structured JSON artifacts, forming a self-improving outer loop that an orchestration runner can drive end-to-end with a single command.

The core design rule is simple: **the root agent never sees a raw transcript**. Every reference goes through `chunk:<id>`, every LLM call is wrapped by a validator, and every mutable piece of state (schema, prompts, queries) is versioned and gated.

For the full architecture, read [`docs/architecture.en.md`](docs/architecture.en.md) and open [`docs/architecture.en.html`](docs/architecture.en.html) for the visualization.

---

## What it does

```
call_records.sql ──┐
sample_calls.jsonl ┤→ [cu-agent-rlm]  →  silver/manifest/chunks/catalog  →  [qu-agent-rlm]  →  answer + judgement
                                                ↑                                                       │
                                                │                                                       ↓
                                                └─── prompt_repair_request.jsonl ←──────────────────────┤
                                                └─── column_requests.jsonl     ←───────────────────────┤
                                                └─── query_tasks.jsonl         ←───────────────────────┘
```

- **CU** ingests call records, chunks them, induces a silver schema via an LLM, extracts fields per call with evidence references, gates them on `evidence_coverage >= 0.75`, and publishes a versioned catalog plus materialized silver rows.
- **QU** loads the catalog (never the raw transcripts), plans a multi-step tool workflow, drives BM25 + embedding retrieval (with diversity checks and LLM reranking), and finishes with an answerability/evidence judge that decides success and emits structured CU-improvement requests.
- **Orchestrator** replays the loop: baseline CU → primary query set → QU answers + feedback → refined CU induction → QU re-run, with the entire trace serialized to `orchestration_trace.jsonl`.

Three pieces of state evolve over time and share an identical lifecycle (proposal → gate → accepted state):

| State | Identity | Promotion gate |
|---|---|---|
| **Catalog** (silver_schema) | `field_name + schema_version` | `evidence_coverage >= 0.75` (auto) |
| **Prompts** | `prompt_id + version` | `external_hand_labeled_eval_required` (gated until EVAL-001) |
| **Queries** | `task_id + generation_id` | `label_status: hand_labeled` (gated until external fixtures land) |

---

## Quick start

### 1. Prerequisites

- Python `>= 3.10`
- An `OPENAI_API_KEY` (the stack is LLM-first by default — `gpt-5.4-mini` is the default planner/extractor model, `text-embedding-3-small` is the default embedding model)
- A local `.env` at the repo root:

```bash
OPENAI_API_KEY=sk-...
# optional overrides:
OPENAI_MODEL=gpt-5.4-mini
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
# optional: a different model for the answerability judge
JUDGE_OPENAI_MODEL=gpt-5.4
```

### 2. Run the full CU ↔ QU loop demo

```bash
cd /path/to/agent-native
python3 scripts/qu_cu_loop_demo.py --output-root output/qu_cu_loop_demo
```

This runs:

1. **CU baseline** — LLM-induced schema, per-call extraction, evidence gate, materialized silver
2. **Query bootstrap** — LLM generates a downstream query set (`query_tasks.jsonl`)
3. **QU baseline** — multi-step plan + agentic retrieval + judge for each query
4. **Feedback emission** — column_requests / search_diagnostics / judgement → `column_requests.jsonl`
5. **CU refined** — re-induces the schema with `--feedback-input`, accepts feedback through the normal evidence gate
6. **QU refined** — replays the same queries against the refined catalog
7. **Loop evaluation** — `loop_summary.json` + `orchestration_trace.jsonl`

Inspect the result:

```
output/qu_cu_loop_demo/
├── 01_cu_baseline/silver_schema_catalog.json
├── 02_qu_feedback/query_tasks.jsonl
├── 02_qu_feedback/baseline_answers/<task_id>.json
├── 02_qu_feedback/column_requests.jsonl
├── 03_cu_refined/feedback_report.json
├── 04_qu_refined/refined_answers/<task_id>.json
├── loop_summary.json
└── orchestration_trace.jsonl
```

### 3. Or run a single user query against an existing corpus

```bash
python3 scripts/qu_user_query_demo.py \
    --corpus output/qu_cu_loop_demo/03_cu_refined \
    --query "Which calls mention founder-led sales conversations?"
```

Defaults: agentic retrieval mode + OpenAI planner + reranker + judge.

---

## Repository layout

```
agent-native/
├── cu-agent-rlm/         # Content Understanding (CU) package
│   ├── src/cu_agent_rlm/ # pipeline, schema, extraction, feedback, prompt_registry
│   ├── tests/            # unit tests (7 tests)
│   ├── docs/issues.md    # issue backlog (CU/SYS/FB IDs)
│   └── README.md         # CU-specific quick start
├── qu-agent-rlm/         # Query Understanding (QU) package
│   ├── src/qu_agent_rlm/ # agent, planner, retrieval, retrieval_agent, judge, query_tasks
│   ├── tests/            # unit tests (16 tests)
│   └── README.md         # QU-specific quick start
├── scripts/
│   ├── qu_cu_loop_demo.py     # full outer-loop runner
│   └── qu_user_query_demo.py  # single-query smoke runner
├── docs/
│   ├── architecture.en.md     # English architecture document
│   ├── architecture.en.html   # English single-file visualization
│   ├── architecture.md        # Japanese version
│   └── architecture.html      # Japanese version
├── call_records.sql      # Databricks source SQL for production wiring
├── cu-agent/             # legacy non-RLM CU (kept for reference)
├── qu-agent/             # legacy non-RLM QU (kept for reference)
└── output/               # demo outputs (gitignored)
```

The two `*-agent-rlm` packages are standalone — they ship `pyproject.toml`, are installable via `uv sync` or `pip install -e .`, and depend only on the standard library, `bm25s`, and `numpy` (no shared runtime — duplicated `llm.py` keeps them isolatable).

---

## Running the components individually

### CU only

```bash
cd cu-agent-rlm
python3 -m cu_agent_rlm.cli \
    --input ../cu-agent/data/sample_calls.jsonl \
    --source-sql ../call_records.sql \
    --output output/demo \
    --schema-inducer openai \
    --extractor openai
```

User-facing modes: `openai` / `openai-compatible`. Heuristic implementations exist only as automatic reliability fallbacks behind LLM validators (see [SYS-001](cu-agent-rlm/docs/issues.md)).

### QU only

```bash
cd qu-agent-rlm
python3 -m qu_agent_rlm.cli \
    --corpus ../cu-agent-rlm/output/demo \
    --query "Which accounts have pricing objections?" \
    --planner openai \
    --judge openai \
    --reranker openai \
    --retrieval-mode agentic
```

`--retrieval-mode agentic` automatically enables the LLM retrieval subagent and reranker with `min_search_calls=2`, query diversity threshold `0.8`, and forced-iteration fallback. See `qu-agent-rlm/README.md` for the full flag surface.

---

## Testing

```bash
# CU
cd cu-agent-rlm && python3 -m unittest discover -s tests   # 7 tests

# QU
cd qu-agent-rlm && python3 -m unittest discover -s tests   # 16 tests
```

Both test suites use `FakeJSONClient` / `FakeSequenceJSONClient` to stub LLM calls deterministically — no network access required.

---

## Status & roadmap

**Bootstrap-quality prototype**, intentionally capped at 70–80% self-improvement and not yet wired to production Databricks. The complete issue backlog lives at [`cu-agent-rlm/docs/issues.md`](cu-agent-rlm/docs/issues.md).

The critical path:

- **EVAL-001** (external hand-labeled fixtures) — currently open
- → unblocks **FB-002** (eval-gated prompt promotion)
- → enables automatic mutation of prompts and query distributions, both of which are currently signal-only

Until then:
- Prompt changes happen by hand-editing `prompt_registry.py` (versioned `PromptSpec(frozen=True)`)
- Synthetic queries (`source_type: synthetic_llm`) are replayed for exploration but never drive promotion
- Schema changes proposed by QU pass through the normal `evidence_coverage >= 0.75` gate

Production wiring (`DBX-001`) is also open: the contract is sketched in `databricks_contract.json` but `run_sql`, `query_silver`, and `search_chunks` adapters for Databricks/OpenSearch are not yet implemented.

---

## Design references

- [`docs/architecture.en.md`](docs/architecture.en.md) — the full architecture writeup (12 sections + file-level appendix)
- [`docs/architecture.en.html`](docs/architecture.en.html) — single-file visualization (open in a browser)
- [`cu-agent-rlm/docs/issues.md`](cu-agent-rlm/docs/issues.md) — issue backlog with statuses (`Done` / `Partial` / `Open` / `Blocked` / `Design`)
- [`cu-agent-rlm/README.md`](cu-agent-rlm/README.md) and [`qu-agent-rlm/README.md`](qu-agent-rlm/README.md) — per-package quick starts and CLI references

---

**schema_version**: `silver.rlm.v1` · **prompt baseline**: `2026-05-29.1` · **last updated**: 2026-05-30
