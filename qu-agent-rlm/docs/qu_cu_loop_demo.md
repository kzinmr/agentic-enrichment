# QU-CU Feedback Loop Demo

This demo shows the orchestration loop:

1. CU builds an initial silver contract.
2. The orchestrator prepares downstream primary query tasks from manual input, external fixtures, production logs, or LLM/bootstrap generation.
3. QU answers each query against that contract.
4. QU judges answerability/evidence, then emits `column_requests`, search diagnostics, and judgement.
5. CU consumes that feedback and refines schema induction.
6. QU re-runs the same query distribution and checks whether the refined schema is more expressible.

Run from the repository root:

```bash
python3 scripts/qu_cu_loop_demo.py --output-root output/qu_cu_loop_demo
```

The demo now renders a terminal UI by default: phase status, CU/QU metrics, query-task tables, feedback counts, schema deltas, and artifact paths. Use `--no-tui` for JSON-only output, or `--print-json` to append the raw summary JSON after the TUI.

The default manual seed query is:

```text
Which calls mention founder-led sales calls?
```

Expected loop shape:

- Before refinement, QU uses `search` because no silver field represents `founder_led_sales`.
- The orchestrator writes `output/qu_cu_loop_demo/02_qu_feedback/query_tasks.jsonl`.
- QU writes `output/qu_cu_loop_demo/02_qu_feedback/column_requests.jsonl`.
- CU reads that feedback with `--feedback-input`, adds `founder_led_sales`, and emits `feedback_report.json`.
- After refinement, QU uses `filter` with `{"founder_led_sales": "founder"}`.
- `loop_summary.json` shows `feedback_promoted: true`, refined filter counts, and column-request deltas across the replayed query tasks.
- `orchestration_trace.jsonl` links each CU/QU/feedback step with one `loop_id`, iteration number, metrics, decisions, and artifact paths.

Key artifacts:

- `01_cu_baseline/`: first CU run.
- `02_qu_feedback/query_tasks.jsonl`: downstream primary query tasks used for QU replay, with source/provenance.
- `02_qu_feedback/query_bootstrap.json`: query generation report, prompt metadata, source counts, and promotion policy.
- `02_qu_feedback/baseline_answer.json`: QU answer before refinement.
- `02_qu_feedback/baseline_answers/`: one baseline QU answer per query task.
- `02_qu_feedback/column_requests.jsonl`: QU-to-CU feedback.
- `03_cu_refined/feedback_report.json`: accepted/promoted feedback status.
- `04_qu_refined/refined_answer.json`: QU answer after refinement.
- `04_qu_refined/refined_answers/`: one refined QU answer per query task.
- `loop_summary.json`: compact before/after summary for demos.
- `orchestration_trace.jsonl`: unified outer-loop trace for evaluation and debugging.

LLM query bootstrap:

```bash
python3 scripts/qu_cu_loop_demo.py \
  --env-file .env \
  --bootstrap-queries openai \
  --bootstrap-query-count 6 \
  --output-root output/qu_cu_loop_bootstrap
```

Each `synthetic_llm` query records a `generation_id`, generator/model, `qu.downstream_query_bootstrap` prompt id/version/hash, and `adjusted_for` with the CU schema version, schema hash, field names, record count, and chunk count. These queries are useful for bootstrap feedback, but remain `label_status=unlabeled`; prompt/schema promotion still requires external hand-labeled fixtures.

External query sources:

```bash
python3 scripts/qu_cu_loop_demo.py \
  --query-tasks fixtures/query_tasks.jsonl \
  --production-query-log logs/query_log.jsonl \
  --skip-default-query \
  --output-root output/qu_cu_loop_external
```

`query_tasks.jsonl` may contain strings or objects with `query`, `task_id`, `intent`, `source_type`, `label_status`, `expected_operation`, `expected_call_ids`, and `expected_evidence_ids`. Production logs are normalized with `source_type=production_log`.

The QU answer contains `judgement` with `answerable`, `evidence_sufficient`, `success`, `needs_cu_feedback`, `failure_modes`, and `missing_field_requests`. Demo runs use the LLM judge by default; heuristic judging is retained only as a reliability fallback and unit-test helper.

User analysis demo:

After `scripts/qu_cu_loop_demo.py` has produced loop artifacts, query the agent built from the refined CU artifact:

```bash
python3 scripts/qu_user_query_demo.py \
  --loop-output-root output/qu_cu_loop_demo \
  --query "Which calls mention founder-led sales conversations and show evidence?" \
  --query "Break down calls by conversation topic."
```

This user-query demo also renders a terminal UI by default, including the selected loop artifact, agent config, query list, per-query answer metrics, and emitted feedback/prompt-repair artifacts. Use `--no-tui` or `--print-json` when a machine-readable summary is needed.

By default this uses `03_cu_refined` as the corpus. Use `--stage baseline` to query `01_cu_baseline`, or `--corpus path/to/cu_artifact` to override the artifact directory.

Artifacts written under `05_user_queries/refined/`:

- `user_query_tasks.jsonl`: user analysis queries normalized into replayable query tasks.
- `answers/*.json`: full QU answer payloads with `agent_context`, `query_task`, evidence, judgement, diagnostics, and prompt state.
- `user_query_trace.jsonl`: session-level trace linking the selected loop artifact to each user query.
- `user_feedback.jsonl`: QU-to-CU feedback generated by these user queries.
- `prompt_repair_request.jsonl`: prompt/model improvement signals from user-query failures.
- `user_query_summary.json`: compact session summary for demos.

The user-query demo is also LLM-first by default:

```bash
python3 scripts/qu_user_query_demo.py \
  --env-file .env \
  --loop-output-root output/qu_cu_loop_bootstrap \
  --retrieval-mode agentic \
  --reranker openai \
  --query "Which calls mention pricing blockers and show the evidence?"
```

Equivalent manual LLM-first steps:

```bash
uv run --python 3.13 python -m cu_agent_rlm.cli \
  --env-file ../.env \
  --llm-model gpt-5.4-mini \
  --input ../cu-agent/data/sample_calls.jsonl \
  --source-sql ../call_records.sql \
  --output /tmp/qu_cu_loop/01_cu_baseline
```

```bash
uv run --python 3.13 python -m qu_agent_rlm.cli \
  --env-file ../.env \
  --llm-model gpt-5.4-mini \
  --corpus /tmp/qu_cu_loop/01_cu_baseline \
  --query "Which calls mention founder-led sales calls?" \
  --retrieval-mode agentic \
  --reranker openai \
  --feedback-output /tmp/qu_cu_loop/02_qu_feedback/column_requests.jsonl
```

```bash
uv run --python 3.13 python -m cu_agent_rlm.cli \
  --env-file ../.env \
  --llm-model gpt-5.4-mini \
  --input ../cu-agent/data/sample_calls.jsonl \
  --source-sql ../call_records.sql \
  --feedback-input /tmp/qu_cu_loop/02_qu_feedback/column_requests.jsonl \
  --output /tmp/qu_cu_loop/03_cu_refined
```

```bash
uv run --python 3.13 python -m qu_agent_rlm.cli \
  --env-file ../.env \
  --llm-model gpt-5.4-mini \
  --corpus /tmp/qu_cu_loop/03_cu_refined \
  --query "Which calls mention founder-led sales calls?" \
  --retrieval-mode agentic \
  --reranker openai
```
