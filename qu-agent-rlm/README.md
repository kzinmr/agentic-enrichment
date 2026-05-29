# QU Agent RLM Prototype

`qu-agent-rlm` consumes `cu-agent-rlm` artifacts and performs query understanding over the silver contract instead of over raw transcripts.

```bash
cd qu-agent-rlm
uv run qu-agent-rlm \
  --corpus ../cu-agent-rlm/output/demo \
  --query "Which calls mention security review or access controls?"
```

The planner now emits an ordered tool workflow, not just a single operation. The agent can search chunks, query silver rows, aggregate, fetch evidence, and then review whether the current CU schema was sufficient. QU does not mutate the CU schema; missing or weak columns are returned as `column_requests` for a later CU run.

```bash
uv run qu-agent-rlm \
  --corpus ../cu-agent-rlm/output/demo \
  --query "Which calls mention founder-led sales calls?" \
  --feedback-output output/cu_column_requests.jsonl
```

Retrieval tools are modular:

- `bm25_search_chunks`: lexical retrieval backed by `xhluca/bm25s` when installed, with a dependency-free fallback for tests.
- `embedding_search_chunks`: semantic retrieval through OpenAI `text-embedding-3-small`.
- `search_chunks`: compatibility tool routed by `--retrieval-mode`.

```bash
uv run qu-agent-rlm \
  --corpus ../cu-agent-rlm/output/demo \
  --query "Which calls discuss founder-led sales conversations?" \
  --retrieval-mode embedding \
  --embedding-model text-embedding-3-small
```

Use `--retrieval-mode hybrid` to let the compatibility search tool call BM25 and embedding retrieval and merge by agent-visible rank order. LLM planners can also call `bm25_search_chunks` and `embedding_search_chunks` as separate tool steps so the agent can inspect and integrate results without hard-coded RRF fusion.

Use `--retrieval-mode agentic` to enable the fuller agentic search path. In this mode the agent delegates retrieval iteration to a specialized retrieval subagent, enforces at least two retrieval calls, sends search summaries back to the subagent controller for follow-up query/tool selection, rejects duplicate or low-diversity query/tool pairs, and sends BM25/embedding candidates to an LLM reranker for final ordering.

```bash
uv run qu-agent-rlm \
  --env-file ../.env \
  --planner openai \
  --retrieval-mode agentic \
  --llm-model gpt-5.4-mini \
  --corpus ../cu-agent-rlm/output/demo \
  --query "Which calls discuss founder-led sales conversations?"
```

The same pieces can be enabled independently with `--search-controller openai`, `--reranker openai`, `--min-search-calls`, `--max-search-iterations`, and `--query-diversity-threshold`.

`--retrieval-subagent` selects the controller backend for that specialized subagent. The default uses OpenAI when `--retrieval-mode agentic` is selected and otherwise respects the deprecated `--search-controller` override. To swap in a SID-1-style search model, expose it as an OpenAI-compatible JSON endpoint and run:

```bash
uv run qu-agent-rlm \
  --env-file ../.env \
  --planner openai \
  --retrieval-mode agentic \
  --retrieval-subagent openai-compatible \
  --llm-base-url "$SID1_BASE_URL" \
  --llm-model "$SID1_MODEL" \
  --corpus ../cu-agent-rlm/output/demo \
  --query "Which calls discuss founder-led sales conversations?"
```

Every answer now includes an answerability/evidence `judgement` and `prompt_state` with prompt ids, versions, and hashes for LLM-backed planner/retrieval/reranker/judge calls. `--judge` accepts `openai` or `openai-compatible`; heuristic judging remains only as a reliability fallback when LLM output fails validation and as a direct unit-test helper. The judge can use a separate stronger model with `--judge-model`, `--judge-base-url`, `--judge-api-key`, and `--judge-timeout-seconds`.

Embedding indexes are built lazily from CU artifacts and cached under `<corpus>/.qu_agent_rlm/text-embedding-3-small.json` by default. Reindexing happens when the CU chunk text, selected silver fields, or embedding model changes.

BM25 evidence includes `query_terms`, `matched_terms`, and per-term `score_details` so the controller can see which lexical terms matched, which terms were missing, and why a follow-up semantic or reformulated search may be needed. `review_schema_gaps` uses those search diagnostics and LLM controller failure reasons when emitting QU-to-CU `column_requests`.

`--feedback-output` writes `column_requests`, search diagnostics, rerank details, and the answerability judgement. `cu-agent-rlm --feedback-input` can consume that JSONL to re-induce or refine the next silver schema.

`--prompt-repair-output` writes prompt/model improvement signals to `prompt_repair_request.jsonl`. This is intentionally signal-only: prompts are not modified or promoted without external hand-labeled eval fixtures.

For a complete LLM-first QU-CU improvement-loop demo, run this from the repository root:

```bash
python3 scripts/qu_cu_loop_demo.py --output-root output/qu_cu_loop_demo
```

The loop can also bootstrap a downstream primary-query distribution before QU replay. `query_tasks.jsonl` records each generated or loaded user-facing query, its source type, generation id, prompt hash, and the CU schema snapshot it was adjusted against. Synthetic tasks are bootstrap probes only; external hand-labeled fixtures are still required for promotion gates.

```bash
python3 scripts/qu_cu_loop_demo.py \
  --env-file .env \
  --bootstrap-queries openai \
  --bootstrap-query-count 6 \
  --output-root output/qu_cu_loop_bootstrap
```

Use `--query-tasks path/to/query_tasks.jsonl` for curated or hand-labeled fixtures, and `--production-query-log path/to/query_log.jsonl` to replay observed production queries through the same outer loop.

After the loop has produced `03_cu_refined`, query the resulting agent with actual user analysis requests:

```bash
python3 scripts/qu_user_query_demo.py \
  --loop-output-root output/qu_cu_loop_bootstrap \
  --query "Which calls mention founder-led sales conversations and show evidence?" \
  --query "Break down calls by conversation topic."
```

This writes `05_user_queries/refined/user_query_summary.json`, per-query answers, trace events, and any QU-to-CU feedback generated by those user queries.

Both demo scripts render a terminal UI by default. Use `--no-tui` for JSON-only output, or `--print-json` to append the raw JSON summary after the TUI.

The detailed demo walkthrough is in `qu-agent-rlm/docs/qu_cu_loop_demo.md`.

By default, query planning and answer judging use the configured OpenAI model. Keep `OPENAI_API_KEY` in `/Users/kazukiinamura/agent-native/.env` and run:

```bash
uv run qu-agent-rlm \
  --planner openai \
  --llm-model gpt-5.4-mini \
  --judge openai \
  --judge-model gpt-5.4-mini \
  --env-file ../.env \
  --corpus ../cu-agent-rlm/output/demo \
  --query "Break down security review calls by product area."
```

For another OpenAI-compatible Chat Completions endpoint:

```bash
uv run qu-agent-rlm \
  --planner openai-compatible \
  --llm-base-url "$OPENAI_BASE_URL" \
  --llm-model gpt-5.4-mini \
  --env-file ../.env \
  --corpus ../cu-agent-rlm/output/demo \
  --query "Which accounts have pricing objections?"
```

Run the CU-generated smoke tasks:

```bash
uv run qu-agent-rlm \
  --corpus ../cu-agent-rlm/output/demo \
  --eval-tasks ../cu-agent-rlm/output/demo/evaluation_tasks.json
```

There is no no-key or `auto` demo path. Deterministic helpers remain for smoke tests and retrieval-control assertions, while user-facing CLI/demo runs are LLM-first to avoid accidentally benchmarking heuristic behavior as agent intelligence.

The agent uses the same tool boundary expected from SID-1 style agentic retrieval: load schema, plan filters/aggregations, query silver rows, search chunks only when needed, and fetch evidence refs before making claims.
