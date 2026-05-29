# CU Agent RLM Prototype

`cu-agent-rlm` turns exported Databricks `call_records` rows into a small RLM-style analysis substrate:

- `manifest.json`: dataset, record, and tool manifest for root agents
- `chunks.jsonl`: retrievable call chunks for evidence-only access
- `silver_schema_catalog.json`: query-planning contract consumed by `qu-agent-rlm`
- `silver_calls.jsonl`: materialized silver fields per call
- `rlm_trace.jsonl`: root/sub-RLM style tool trace
- `databricks_contract.json`: production mapping for Databricks-backed tools
- `evaluation_tasks.json`: downstream QU smoke tasks

The CLI path is LLM-first: schema induction and field extraction call the configured OpenAI/OpenAI-compatible model, then validators enforce field types, allowed enum values, and evidence refs before values are materialized. Generic heuristic induction/extraction remains in the codebase as a system reliability layer for LLM validation failures and smoke tests; it is not exposed as a user-facing intelligence mode.

```bash
cd cu-agent-rlm
uv run cu-agent-rlm \
  --input ../cu-agent/data/sample_calls.jsonl \
  --source-sql ../call_records.sql \
  --output output/demo
```

By default, the CLI searches upward for `.env`, so `/Users/kazukiinamura/agent-native/.env` can provide `OPENAI_API_KEY` without putting the key on the command line:

```bash
uv run cu-agent-rlm \
  --schema-inducer openai \
  --extractor openai \
  --llm-model gpt-5.4-mini \
  --env-file ../.env \
  --input ../cu-agent/data/sample_calls.jsonl \
  --source-sql ../call_records.sql \
  --output output/openai-demo
```

`--schema-inducer` and `--extractor` accept only `openai` or `openai-compatible`. The static and heuristic implementations are still available for direct unit-test injection and reliability fallback, but not as CLI modes.

Feed QU failures, column requests, and answerability/evidence judgement back into the next CU run with `--feedback-input`. This wraps the selected schema inducer with a feedback-aware refinement pass, merges repeated QU field requests, emits `feedback_report.json`, and lets requested fields compete for promotion under the normal extraction/evidence quality gate.

```bash
uv run cu-agent-rlm \
  --input ../cu-agent/data/sample_calls.jsonl \
  --source-sql ../call_records.sql \
  --feedback-input ../qu-agent-rlm/output/cu_column_requests.jsonl \
  --output output/refined
```

LLM-backed CU artifacts record prompt ids, versions, and hashes in `manifest.json` under `prompt_state`. Use `--prompt-repair-output output/prompt_repair_request.jsonl` to collect schema/extraction prompt improvement signals without changing prompts or promoting new prompt versions.

For a complete LLM-first QU-CU improvement-loop demo, run this from the repository root:

```bash
python3 scripts/qu_cu_loop_demo.py --output-root output/qu_cu_loop_demo
```

To let the orchestrator bootstrap downstream primary queries before QU feedback, use `--bootstrap-queries openai` or provide external tasks with `--query-tasks`. The generated `02_qu_feedback/query_tasks.jsonl` is consumed only as QU replay input and feedback provenance; it does not become a hand-labeled eval gate.

After the loop produces `03_cu_refined`, `scripts/qu_user_query_demo.py` can build a QU agent from that artifact and run user analysis queries, writing user-query answers and any new QU-to-CU feedback under `05_user_queries/`.

Both demo scripts render a terminal UI by default. Use `--no-tui` for JSON-only output, or `--print-json` to append the raw JSON summary after the TUI.

The detailed demo walkthrough is in `qu-agent-rlm/docs/qu_cu_loop_demo.md`.

For another OpenAI-compatible Chat Completions endpoint:

```bash
uv run cu-agent-rlm \
  --schema-inducer openai-compatible \
  --extractor openai-compatible \
  --llm-base-url "$OPENAI_BASE_URL" \
  --llm-model gpt-5.4-mini \
  --env-file ../.env \
  --input ../cu-agent/data/sample_calls.jsonl \
  --source-sql ../call_records.sql \
  --output output/api-demo
```

The intended production replacement is not to load `call_records` into the prompt. Production should map `search_chunks`, `fetch_chunks`, `query_silver`, `aggregate_silver`, and `run_sql` to Databricks/OpenSearch-backed allowlisted tools.
