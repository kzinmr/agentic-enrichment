# Prototype Issue Backlog

This backlog captures the main domain assumptions, expressiveness limits, and feedback-loop gaps currently encoded in `cu-agent-rlm` and `qu-agent-rlm`. Treat each item as an implementation ticket with acceptance criteria.

## CU-001: LLM-backed field extraction with schema/evidence validation

Status: Done in prototype.

Problem: Field extraction was previously keyword/rule based, so semantically equivalent wording and customer-specific expressions were missed.

Implemented scope:
- Add `FieldExtractor` interface with heuristic, OpenAI, and OpenAI-compatible implementations.
- Validate LLM output against field type, enum/list values, confidence, and evidence refs.
- Preserve heuristic extraction as a reliability fallback and smoke-test helper, not as a user-facing agent mode.

Remaining gaps:
- Validation checks that evidence refs exist, but does not verify that the cited text entails the extracted value.
- Boolean fields still conflate `false` with `not_mentioned` in several downstream paths.
- LLM failures fall back to heuristics after one failed JSON attempt; there is no repair prompt or per-field retry.

Acceptance criteria:
- Add evidence-entailment validation per field and mark unsupported claims as abstained.
- Represent boolean as true/false/unknown when absence and explicit negation matter.
- Add retry/repair prompts for invalid LLM JSON before fallback.
- Add per-field extraction latency and token/cost accounting.

## CU-002: Open-world schema discovery and candidate field proposal

Status: Done in prototype.

Problem: `FIELD_SPECS` was a fixed business-call schema. The system needed a schema-induction stage before extraction so new domains are not blocked on code changes.

Implemented scope:
- Add `SchemaInducer` interface with heuristic, static, OpenAI, and OpenAI-compatible implementations.
- Make the default CU path induce a generic schema from call content instead of using the fixed business-call schema.
- Validate LLM-induced field names, field types, allowed values, and search/aggregation hints.
- Keep static/heuristic in-process implementations for unit-test injection and reliability fallback while making CLI/demo LLM-first.

Remaining gaps:
- Schema induction is one-shot over local samples, not a long-running discovery loop over full Databricks scale.
- The inducer sees chunks directly; production still needs retrieval/manifests rather than prompt-loading large datasets.
- Query-task/bootstrap context is not yet a first-class input to schema induction prompts.

Acceptance criteria:
- Add iterative schema refinement from QU failures and human review.
- Let the inducer retrieve representative examples from Databricks-backed tools instead of sampling the first local chunks.
- Add schema version diffing, migration notes, and compatibility checks for existing QU tasks.

## SYS-001: Heuristic reliability layer and mode hygiene

Status: Done in prototype.

Problem: Public `heuristic` and `auto` modes made it easy to run a demo or production-like command that silently benchmarked deterministic rules instead of agentic LLM judgement.

Implemented scope:
- Remove `heuristic`, `static`, and `auto` from user-facing CU schema/extraction CLI modes.
- Remove `heuristic` and `auto` from user-facing QU planner/judge/retrieval-subagent/reranker modes.
- Make `scripts/qu_cu_loop_demo.py` and `scripts/qu_user_query_demo.py` LLM-first by default; API-key-free deterministic demo mode is intentionally unsupported.
- Retain heuristic components only as reliability fallbacks behind LLM validators, direct unit-test helpers, smoke-test support, and deterministic retrieval-control mechanisms such as BM25 explainability, query diversity, and forced search iteration.

Remaining gaps:
- Schema-induction fallback provenance is not yet as explicit as planner/judge fallback provenance in every artifact.
- Fallback use is not yet aggregated into budget/observability reports.

Acceptance criteria:
- Trace every fallback with component, provider, prompt hash, validation/API error, and selected fallback.
- Add smoke tests that assert public parsers reject removed modes while direct component injection still works for deterministic reliability tests.
- Keep deterministic retrieval controls documented separately from semantic agent judgement.

## CU-003: Replace closed labels with typed ontology policy

Status: Open.

Problem: List and enum values are currently closed normalized labels, which improves validation but limits expressiveness across domains.

Implicit assumption:
- Useful silver fields can be represented as `list`, `enum`, `boolean`, or `string` with snake_case ASCII values.

Acceptance criteria:
- Distinguish strict enums from extensible controlled vocabularies.
- Preserve raw LLM label, normalized label, and ontology mapping confidence.
- Add a human-review state for new labels before they become aggregatable.
- Add numeric, date/time, entity, relation, and event field types.

## CU-004: QU-to-CU schema refinement loop

Status: Done in prototype.

Problem: QU can emit `column_requests`, but CU does not yet treat them as first-class inputs for the next schema induction or extraction run.

Implemented scope:
- Add `--feedback-input` to `cu-agent-rlm`.
- Read QU feedback JSONL and merge repeated requests into candidate field proposals with frequency, query examples, evidence refs, and search failure reasons.
- Wrap the selected base schema inducer with a feedback-aware refinement pass.
- Gate feedback-derived fields through the normal extraction/evidence promotion policy.
- Emit `feedback_report.json` with accepted/promoted status and validation error counts.

Remaining gaps:
- There is no human approval queue or schema migration workflow.
- Rejected/deferred fields are reported, but not yet routed into a review backlog.
- Regression evaluation is not automatically required before promotion.

Acceptance criteria:
- Add human review states for accepted, rejected, deferred, and superseded feedback requests.
- Add schema migration notes and compatibility checks for accepted feedback fields.
- Require before/after evals before promoting feedback-derived fields in production.

## CU-005: Extraction quality calibration and adjudication

Status: Open.

Problem: Current promotion uses simple support and evidence coverage. It does not estimate precision, recall, or confidence calibration.

Acceptance criteria:
- Track per-field false positives, false negatives, abstentions, and evidence-hit rate on hand-labeled fixtures.
- Add a second-pass adjudicator for low-confidence or high-impact fields.
- Persist model confidence, validator confidence, and final calibrated confidence separately.
- Prevent low-quality fields from becoming filterable or aggregatable by default.

## QU-001: Multi-step query plans instead of single operation plans

Status: Done in prototype.

Problem: QU previously chose one of `filter`, `aggregate`, or `search`. Complex analysis often needs chained retrieval, filtering, grouping, and evidence checks.

Implemented scope:
- Planner returns a list of typed tool steps.
- Agent executes dependent steps and carries intermediate call ids, chunks, aggregation, evidence, and schema-gap requests.
- Trace captures each tool input/output summary and final payload.

Remaining gaps:
- Steps are still a linear workflow; there is no branching, join, or conditional retry over non-search tools.
- Final answer synthesis is still template-based, not an LLM synthesis over evidence and uncertainty.

Acceptance criteria:
- Add typed state inputs/outputs for each tool step.
- Add conditional branches for "no records", "low evidence", and "aggregation denominator mismatch".
- Add LLM answer synthesis with strict citation and abstention policy.

## QU-002: Agentic search iteration, query diversity, and reranking

Status: Done in prototype.

Problem: Search should not stop at the first retrieval call. The agent should inspect results, reformulate queries, switch tools, and rerank final candidates.

Implemented scope:
- Add `SearchExecutionPolicy` with `min_calls`, `max_iterations`, and `query_diversity_threshold`.
- Split retrieval iteration behind `RetrievalSubAgent` / `AgenticRetrievalSubAgent` so SID-1-style specialized search agents can replace the default controller.
- Add LLM retrieval subagent controller that sees prior search summaries and proposes follow-up search steps.
- Reject duplicate query/tool pairs and low-diversity same-tool queries.
- Add LLM reranker over BM25/embedding candidates.
- Expose `search_diagnostics`, `rerank`, and BM25 `query_terms` / `matched_terms` / `score_details`.

Remaining gaps:
- The loop is search-specific, not a general ReAct loop over all tools.
- The specialized retrieval subagent is still in-process; remote SID-1 hosting, auth, latency budgets, and tool-call transcript normalization are not implemented.
- Diversity is English-token Jaccard and does not understand phrase/proximity or non-English queries.
- The reranker only ranks chunks; it does not produce a calibrated answer, confidence, or abstention.
- Agentic mode can increase cost and latency without a quality-aware budget policy.

Acceptance criteria:
- Add budget-aware stopping criteria based on marginal result gain, reranker confidence, and evidence sufficiency.
- Add LLM answer synthesis that can reject low-relevance reranked candidates.
- Add query diversity checks based on semantic embeddings as well as lexical terms.
- Record controller/reranker prompt hashes, model, latency, token usage, and validation result.

## QU-003: Query language expressiveness

Status: Open.

Problem: QU supports simple filters, one group-by count, and retrieval. It cannot express many analysis questions users will naturally ask.

Implicit assumption:
- Most queries can be reduced to `filter`, `aggregate`, or `search` over one silver catalog and one chunk corpus.

Acceptance criteria:
- Support date ranges, numeric comparisons, distinct counts, ratios, top-k, nested group-bys, and cohort comparisons.
- Add entity/account-level joins and cross-call aggregation.
- Add contradiction/exception queries such as "calls that look like X but are missing Y".
- Add an allowlisted `run_sql` tool with query planning and SELECT-only validation.

## RET-001: BM25 and embedding retrieval backend

Status: Done in prototype.

Problem: Local `search_chunks` was lexical term overlap. LLM planning could understand semantics, but retrieval could not.

Implemented scope:
- Add modular BM25, embedding, hybrid, and agentic retrieval modes.
- Use `xhluca/bm25s` for BM25 and OpenAI `text-embedding-3-small` for embeddings.
- Cache embeddings by document hash and model.
- Keep a dependency-free fallback BM25 path for tests.

Remaining gaps:
- Tokenization is English-only and shallow; no stemming, phrase matching, proximity, synonyms, or Japanese support.
- BM25F is approximated by synthetic weighted text, not a true fielded BM25 implementation.
- Embedding search is in-memory linear scan, not a scalable vector index.
- Score explanations are useful diagnostics but are not guaranteed to numerically match the `bm25s` backend score.

Acceptance criteria:
- Add tokenizer/language policy and tests for multilingual data.
- Add true field-aware retrieval or explicit BM25F scoring when silver fields are included in search text.
- Add production adapters for OpenSearch/Databricks vector search.
- Add retrieval eval metrics for lexical, embedding, hybrid, and agentic modes.

## EVID-001: Span-level evidence and privacy filtering

Status: Open.

Problem: Evidence refs point to chunks, not exact spans. This is too coarse for audit and privacy-sensitive display.

Acceptance criteria:
- Evidence refs include chunk id, turn range, character offsets, and redaction status.
- Fetching evidence applies role/access filters and PII redaction.
- Answers cite minimal snippets instead of whole chunks.
- Validators check that each extracted value is supported by the cited span.

## FB-001: Model-feedback and prompt-improvement loop

Status: Partially done in prototype.

Problem: The prototype produces traces, validation errors, search diagnostics, and column requests, but does not yet use them to improve prompts, schemas, or model routing.

Implemented scope:
- QU feedback JSONL now carries `column_requests`, `search_diagnostics`, and rerank metadata.
- QU answers now carry `judgement` with answerability, evidence sufficiency, failure modes, confidence, and judge-proposed missing field requests.
- Hardcoded CU/QU prompts are factored into prompt registries with prompt ids, versions, roles, and prompt hashes.
- CU artifacts and QU answers expose `prompt_state` for LLM-backed schema induction, extraction, planning, retrieval, reranking, and judging.
- QU supports separate judge model routing with `--judge-model`, `--judge-base-url`, `--judge-api-key`, and `--judge-timeout-seconds`.
- CU and QU can write `prompt_repair_request.jsonl` as signal-only backlog items through `--prompt-repair-output`.
- CU consumes that feedback through `--feedback-input` and uses it to refine schema induction.
- CU emits `feedback_report.json`, including requested field status and extraction validation error counts.

Remaining gaps:
- Feedback currently improves schema candidates, not planner/extractor prompts or model routing.
- Validation errors are counted, but not mined into repair prompts or routing policies.
- There is no accepted/rejected/superseded lifecycle for feedback items.
- The answer judge is available, but its decisions do not yet route feedback to prompt, retrieval, chunking, validator, or model-routing improvements.
- Prompt repair requests are collected, but there is no prompt lab, eval gate, or promotion workflow yet.

Acceptance criteria:
- Mine repeated validation errors, failed searches, duplicate query rejections, and low reranker scores into a feedback dataset.
- Version prompts, schemas, and model configurations.
- Run before/after evals before accepting prompt or schema changes.
- Add a human-review queue for high-impact schema changes and uncertain LLM judgments.
- Track whether each feedback item was accepted, rejected, superseded, or needs more evidence.

## FB-002: Prompt lab and eval-gated prompt promotion

Status: Blocked on EVAL-001.

Problem: Prompt state and repair signals now exist, but automatic prompt promotion would currently overfit to self-generated judgements because external hand-labeled fixtures are still missing.

Design direction:
- Keep prompt changes manual until EVAL-001 provides an external correctness benchmark.
- Use `prompt_repair_request.jsonl` to accumulate candidate failures, not to mutate prompts directly.
- Prefer early stopping during bootstrap: use self-improvement for 70-80 point startup quality, then rely on production query/outcome logs and external fixtures for higher-confidence prompt/model changes.

Acceptance criteria:
- Add a `prompt_lab` workspace that can propose prompt variants without changing production defaults.
- Require EVAL-001 fixtures before prompt promotion can be machine-approved.
- Compare planner, retrieval, extractor, and judge prompt variants on held-out external fixtures.
- Record prompt promotion decisions with old/new prompt ids, eval deltas, model route, cost, and rollback notes.

## DE-001: CU bootstrap context pack for downstream-driven induction

Status: Partially done in prototype.

Problem: CU bootstrap still mostly infers schema from observed call content. That is useful for discovery, but weak for deciding which extracted signals will be reusable for QU filters, aggregations, ranking, evidence checks, or routing.

Design direction:
- Build an explicit bootstrap context pack passed into schema induction.
- Include downstream query distribution, failed query examples, evaluation tasks, task taxonomy, domain glossary, source schema/stats, evidence policy, and cost policy.
- Make the inducer optimize for query expressibility, not just salient transcript concepts.

Implemented scope:
- `scripts/qu_cu_loop_demo.py` now accepts `--query-tasks` and `--production-query-log` and can generate unlabeled downstream query probes with `--bootstrap-queries`.
- Generated query tasks are written to `02_qu_feedback/query_tasks.jsonl` with `source_type`, `label_status`, `generation_id`, generation prompt metadata, and `adjusted_for` schema provenance.
- The outer loop replays the same query task set before and after CU refinement, and records per-query decisions in `orchestration_trace.jsonl`.

Remaining gaps:
- Query tasks are passed to QU replay and feedback provenance, but not yet fed directly into CU schema induction as a first-class `bootstrap_context.json`.
- Synthetic query tasks are unlabeled bootstrap probes and cannot serve as EVAL-001 promotion fixtures.

Acceptance criteria:
- Define a `bootstrap_context.json` contract with query families, task intents, glossary terms, extraction constraints, and policy metadata.
- Include Databricks table stats and representative examples without loading full raw data into prompts.
- Score candidate fields by expected downstream gain, stability, extraction cost, evidence requirement, and false-positive risk.
- Add tests showing the same data can induce different schemas when downstream query distributions differ.

## DE-002: CU-QU schema negotiation loop

Status: Design enhancement.

Problem: QU can request columns and CU can accept them, but there is no negotiation around whether a field is reusable, which allowed values are appropriate, or whether the concept should remain retrieval-only.

Design direction:
- Add an interaction step where CU proposes candidate fields and asks QU to simulate filter/search/aggregate use cases before promotion.
- Treat schema induction as a negotiation over query expressibility, extraction reliability, and operational cost.

Acceptance criteria:
- CU emits `field_candidates` with intended query patterns, example filters, example aggregations, evidence requirements, and uncertainty.
- QU evaluates each candidate against a small set of generated and observed queries.
- Candidate decisions include `promote`, `defer`, `retrieval_only`, `merge_with_existing`, and `needs_human_review`.
- Rejected/deferred candidates are persisted with rationale and can be revisited when more QU failures accumulate.

## DE-003: Programmatic RLM sub-agent orchestration

Status: Partially done in prototype.

Problem: Current workflows are mostly fixed Python pipelines with LLM components plugged into planner, extractor, reranker, and judge slots. They do not yet use the full RLM pattern where a root agent programmatically spawns specialized sub-agents, inspects intermediate artifacts, and branches.

Implemented scope:
- Retrieval iteration is factored behind a `RetrievalSubAgent` protocol.
- The default `AgenticRetrievalSubAgent` preserves current behavior while allowing SID-1-like controllers to be injected.
- `orchestration_trace.jsonl` already uses an `agent` column, so future process-level subagent separation does not require a trace schema change.

Design direction:
- Let a root orchestration agent decompose schema/search/evaluation work into typed sub-agent calls with explicit budgets and artifacts.
- Use sub-agents for field proposal, field validation, evidence entailment, retrieval branch exploration, answer judging, Databricks stats inspection, and prompt repair.

Acceptance criteria:
- Add a typed sub-agent call interface with input artifact refs, output schemas, budget limits, and validation.
- Support parallel retrieval branches for BM25, embedding, silver filters, and SQL checks, followed by a join/judge step.
- Support CU sub-agents for field-candidate clustering, allowed-value ontology mapping, extraction validator repair, and evidence entailment.
- Persist parent/child run ids in unified orchestration trace so sub-agent decisions are replayable.

## DE-004: Feedback routing beyond schema changes

Status: Design enhancement.

Problem: The improvement loop is biased toward adding or improving CU fields. Many failures should instead update prompts, model routing, retrieval strategy, chunking, validators, or evaluation fixtures.

Design direction:
- Introduce a feedback router that classifies each failure into an improvement target before applying changes.

Acceptance criteria:
- Define routing targets: `schema`, `allowed_values`, `extractor_prompt`, `schema_prompt`, `planner_prompt`, `search_strategy`, `reranker`, `answer_judge`, `chunking`, `indexing`, `model_route`, `eval_fixture`, and `human_review`.
- Route based on validation errors, search diagnostics, answer judgement, reranker confidence, duplicate-query rejections, and regression evals.
- Require before/after eval evidence before accepting prompt/model/retrieval changes.
- Track accepted, rejected, superseded, and reverted improvements with artifact lineage.

## DE-005: Semantic recall evaluation and drift control

Status: Design enhancement.

Problem: Embedding retrieval participates in agentic search, but the system cannot yet measure when semantic recall genuinely improves answer quality versus introducing semantic drift.

Design direction:
- Evaluate BM25, embedding, hybrid, and agent-directed retrieval as separate branches with query-family-level metrics.

Acceptance criteria:
- Add paraphrase and abstraction fixtures where embedding should recover evidence that BM25 misses.
- Track top-k evidence correctness, marginal gain by retrieval branch, false positives, and semantic drift.
- Require the answer judge to distinguish "usable semantic evidence" from adjacent-but-unsupported chunks.
- Add branch-level metrics to `orchestration_trace.jsonl` so retrieval policy can be tuned over time.

## DE-006: Schema lifecycle, backfill, and governance

Status: Design enhancement.

Problem: Feedback-derived fields can be promoted in the prototype, but production needs schema lifecycle control, backfill behavior, compatibility checks, and review policy.

Design direction:
- Treat silver schema as a versioned product contract rather than an agent side effect.

Acceptance criteria:
- Record schema diffs with migration notes, expected query gain, affected downstream tasks, and backfill plan.
- Add field states: `candidate`, `experimental`, `active`, `deprecated`, `rejected`, and `superseded`.
- Separate extraction availability from filterability/aggregatability until quality thresholds are met.
- Add rollback criteria when new fields degrade evals, latency, cost, or false-positive rate.

## DE-007: Human review and policy hooks for high-impact inferences

Status: Design enhancement.

Problem: Some domains and fields need policy constraints that cannot be inferred safely from transcripts alone, especially medical, legal, finance, hiring, compliance, and customer-sensitive inferences.

Design direction:
- Add policy-aware gates before fields become materialized, filterable, or aggregatable.

Acceptance criteria:
- Annotate fields with sensitivity, required evidence level, review requirement, and allowed downstream uses.
- Route high-impact or low-confidence field candidates to human review.
- Prevent unsupported sensitive inferences from being promoted even if they improve QU answerability.
- Include policy decisions in trace and feedback reports.

## EVAL-001: External evaluation fixtures

Status: Open.

Problem: Current `evaluation_tasks.json` is generated from the system's own silver output, so it is a smoke test rather than a correctness benchmark.

Acceptance criteria:
- Add hand-labeled fixtures for extraction values, abstention, and evidence correctness.
- Add QU fixtures for filter, search, aggregation, multi-hop analysis, agentic search, and reranking.
- Report precision, recall, evidence hit rate, invalid-plan rate, reranker NDCG, cost, latency, and tool-call count.
- Include adversarial queries, paraphrases, missing-schema queries, and no-answer queries.

## DBX-001: Databricks production tool adapters

Status: Open.

Problem: `run_sql`, `query_silver`, `aggregate_silver`, `search_chunks`, and `fetch_chunks` are local contracts.

Acceptance criteria:
- Add Databricks read adapter for `call_records` and materialized silver views.
- Enforce SELECT-only SQL policy for agent calls.
- Add batch write/export path for silver materialization.
- Add incremental indexing lifecycle for updated call records and schema versions.

## OBS-001: Budget, observability, and replay

Status: Partially done in prototype.

Problem: RLM traces are lightweight and do not yet capture enough to debug model quality, cost, or safety.

Implemented scope:
- QU answers now include answerability/evidence `judgement`.
- The QU-CU loop demo writes `orchestration_trace.jsonl` with loop id, iteration, phase, agent, event type, artifact refs, metrics, decisions, and summary.

Remaining gaps:
- The unified trace is currently demo-runner scoped, not a shared library used by every CLI path.
- LLM prompt hashes, token usage, latency, budget counters, and fallback reasons are not consistently recorded.
- Trace redaction and replay packaging are not implemented.

Acceptance criteria:
- Trace every LLM call with prompt hash, model, latency, token usage, validation result, fallback reason, and budget counters.
- Provide replay artifacts that exclude raw sensitive text by default.
- Add budget limits for calls, tokens, retries, tool calls, and wall-clock time.
- Add diffable run reports for schema, extraction, retrieval, reranking, and QU answers.
