# Agent-Native RLM Architecture

`cu-agent-rlm` + `qu-agent-rlm` を中心にした、Databricks `call_records` 上の **RLM (Retrieval-based LM) スタイル** の自己改善ループの全体設計。

このドキュメントは 2025-2026 にかけての段階的構築ログを後付けで整理したものです。プロトタイプ全体を 3 つの first-class state（Catalog / Prompts / Queries）と、それらを取り巻くツール境界・promotion gate・observability の階層として俯瞰します。

---

## 1. 全体像

二段の独立パッケージが one-way contract で連結し、QU 側から CU 側へ逆チャネルでフィードバックを返す **半閉ループ → full loop** の構造を持ちます。

```
call_records.sql ──┐
sample_calls.jsonl ┤→ [cu-agent-rlm]  →  silver/manifest/chunks/catalog  →  [qu-agent-rlm]  →  answer + judgement
                                                ↑                                                       │
                                                │                                                       ↓
                                                └─── prompt_repair_request.jsonl ←──────────────────────┤
                                                └─── column_requests.jsonl     ←───────────────────────┤
                                                └─── query_tasks.jsonl         ←───────────────────────┘
```

中核的な設計原則は **「root agent に生の transcript を渡さない」**。代わりに：

- **silver fields** で構造化済みの軽量フィールドを提供
- **evidence_refs (`chunk:<chunk_id>`)** で必要時のみ実テキスト解決
- すべての LLM 呼び出しに **promotion gate / validator / fallback** を挟む

---

## 2. 設計哲学：RLM Tool Boundary

agent が呼べるツールは固定 5 種類（QU の retrieval は 3 種に細分化されて実質 7 ステップツール）。`call_records` 本体は決して LLM のプロンプトに直接入らない。

| ツール | 用途 | 本番マッピング |
|---|---|---|
| `search_chunks` / `bm25_search_chunks` / `embedding_search_chunks` | チャンク単位の検索 | OpenSearch / Databricks Vector Index |
| `fetch_chunks` | evidence ref から実テキスト解決 | access-controlled snippet API |
| `query_silver` | silver フィールドでフィルタ | materialized silver view |
| `aggregate_silver` | facetable フィールドで集計 | 同上 |
| `run_sql` | Databricks SELECT（契約のみ、現状未実装） | Databricks SQL |
| `review_schema_gaps` | QU メタツール：解けなかった形を構造化 | — |

加えて、QU は **外側の planner/replanner loop** と **内側の retrieval subagent loop** を持つ。どちらも同じ state 辞書を参照渡しで更新し、raw transcript は `fetch_chunks` が返す snippet 以上には昇格しない（後述）。

---

## 3. 3 本柱（First-Class State）

CU/QU を通じて mutable な state は 3 つに整理されており、すべて同じ lifecycle pattern を持ちます。

| Pillar | Identity | Storage | 提案 channel | Promotion gate | 現状 |
|---|---|---|---|---|---|
| **Catalog (silver_schema)** | `field_name + schema_version` | `silver_schema_catalog.json` | `column_requests.jsonl` | `evidence_coverage >= 0.75` | 自動 promotion 稼働中 |
| **Prompts** | `prompt_id + version` | `prompt_registry.py` (`PromptSpec(frozen=True)`) | `prompt_repair_request.jsonl` | `external_hand_labeled_eval_required` | **EVAL-001 待ちで blocked** |
| **Queries** | `task_id + generation_id` | `query_tasks.jsonl` | `query_bootstrap.json` + `production_log` | `label_status: hand_labeled` 必須 | unlabeled は探索のみ |

### 3.1 Catalog

- `silver_schema_catalog.json` がスキーマの真実の源
- 各 field に `filterable / facetable / aggregatable` フラグ、`allowed_values`、`evidence.ref_path`、`quality` 情報
- 出自は `manifest.schema_induction.inducer`（例：`openai`, `openai+feedback`）。fallback は validation error / trace 側で追跡する
- promotion 条件：support_count > 0 AND evidence_coverage >= 0.75

### 3.2 Prompts

すべての LLM 呼び出しが版管理された `PromptSpec` を経由。

```python
@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str       # "qu.query_planner"
    version: str         # "2026-06-01.1"
    role: str            # "planner" | "judge" | "extractor" | ...
    system: str
```

主な prompt 定義：

| prompt_id | role | 現行 version | 場所 |
|---|---|---|---|
| `qu.query_planner` | planner | `2026-06-01.1` | `qu/prompt_registry.py` |
| `qu.query_replanner` | planner | `2026-06-01.1` | 同上 |
| `qu.retrieval_controller` | retrieval_subagent | `2026-06-01.1` | 同上 |
| `qu.reranker` | reranker | `2026-05-29.1` | 同上 |
| `qu.answer_judge` | judge | `2026-05-29.1` | 同上 |
| `qu.downstream_query_bootstrap` | query_bootstrapper | `2026-05-29.1` | 同上 |
| `cu.schema_induction` | inducer | `2026-05-29.1` | `cu/prompt_registry.py` |
| `cu.field_extraction` | extractor | `2026-05-29.1` | 同上 |

`frozen=True` により **runtime での自動編集は型レベルで禁止**。変更には新しい `PromptSpec` インスタンス（＝バージョン bump）が必要。

### 3.3 Queries

CU-QU ループに投入される一次クエリも first-class artifact。

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

`source_type` ごとに promotion gate での扱いが異なる：

- `hand_labeled_eval` / `production_log` → prompt/schema 改善の主信号
- `failed_query` / `curated` → schema 改善のみ
- `synthetic_llm` → 探索 only、promotion 不可
- `synthetic_heuristic` → smoke test / legacy helper only、promotion 不可

---

## 4. CU パイプライン

### 4.1 7 ステージ

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

| Inducer | 役割 |
|---|---|
| `LLMSchemaInducer` (OpenAI / openai-compatible) | 本番・demo の一次判断。field 名・型・値すべて LLM が提案し、`validate_schema_payload` で形式チェック |
| `HeuristicSchemaInducer` | reliability fallback / smoke test 用。LLM 失敗時に最低限の contract を出すが、agent の代替知能としては扱わない |
| `StaticSchemaInducer` | 旧 `FIELD_SPECS` 互換の直接注入用 helper。CLI/demo mode ではない |

これらを `FeedbackAwareSchemaInducer` が wrap し、`column_requests.jsonl` の内容を `allowed_values` 追加・新フィールド提案として merge。manifest 上では `"openai"` や `"openai+feedback"` のような合成 provenance 文字列で表現される。fallback の詳細 provenance は SYS-001 の残課題として trace/budget 側へ集約する。

LLM induce 時の validator は以下を強制：

- `RESERVED_FIELD_NAMES`（`call_id, customer_id, date, ...`）の侵入を防ぐ
- snake_case 強制、enum には `not_mentioned` を必ず prepend
- `list/enum` で allowed_values 空のものは drop
- `string` 型は集計対象から自動除外（`facetable = aggregatable = False`）
- `max_fields=12`、`max_values_per_field=12` で語彙爆発を抑制

### 4.3 Extractor

| Extractor | 動作 |
|---|---|
| `LLMFieldExtractor(fallback=Heuristic)` | 本番・demo の標準。1 call につき 1 LLM 呼び出しで全フィールドを返させ、呼び出し失敗時はその call だけ heuristic に退避 |
| `HeuristicFieldExtractor` | reliability fallback / smoke test 用。旧 `LIST_RULES` ＋ 未知 field 用 `generic_extract_field` |

LLM 出力の validator は：

- `evidence_refs` を **LLM に見せた `chunk_id` 集合内に限定**（hallucination 防御）
- 不正値は default value に置換、`validation_errors` に列挙して残す
- `validation_errors` は `quality_flags` 経由で silver_calls.jsonl まで貫通

### 4.4 Promotion Gate

```python
recommended_status = "promote" if support_count > 0 and evidence_coverage >= 0.75 else "hold"
```

`evidence_coverage` = `supported_with_refs / supported_total`。「LLM が値を出した」だけでは昇格せず、**evidence_refs が付かなければ silver に出ない** という構造的なフィルタ。

### 4.5 Feedback Consumption

`--feedback-input <jsonl>` で QU からのフィードバックを次回 induction に取り込む。`summarize_feedback` が 3 段フォールバック型で集約：

1. `column_requests` を merge（同名は priority/count を保持して合算）
2. なければ `search_diagnostics.failures` から `request_from_failures` で合成
3. それもなければ `judgement.needs_cu_feedback` から `request_from_judgement` で合成

`feedback_report.json` が出力され、各 field の `accepted_into_schema` / `promoted_to_silver` 状態と `validation_error_counts` を構造化記録。

---

## 5. QU パイプライン

### 5.1 観測駆動 planner loop + 多段 step executor

```
load_schema
  → for plan_iteration in range(max_plan_iterations):
       ├─ plan_query / replan_query
       │    iteration 0: original query + catalog
       │    iteration >0: previous state/judgement/search diagnostics/column_requests
       ├─ [steps を順次実行] state を蓄積
       │    ├─ search_chunks / bm25_search_chunks / embedding_search_chunks
       │    ├─ query_silver
       │    ├─ aggregate_silver
       │    ├─ fetch_chunks
       │    └─ review_schema_gaps
       ├─ run_search_iterations  (search step 後の retrieval subagent loop)
       ├─ rerank_search_results  (LLM reranker、optional)
       ├─ answer_judge           (LLM-first、heuristic は validation failure fallback)
       └─ stop if success / guardrail stop / iteration limit, else plan_observation → replan_query
```

`--max-plan-iterations`（既定 2、`1` で従来相当）が外側ループを有界化する。`LLMQueryPlanner` は `replan(query, catalog, observation)` を実装し、heuristic/static planner は後方互換のため single-plan path のまま動く。

replan observation は raw transcript ではなく圧縮ビューだけを渡す：

- `records` / `aggregation` / `evidence` の compact view
- `chunks_for_llm(state["chunks"])` と `summarize_search_calls_for_llm(state["search_calls"])`
- `search_failures`, judge の `failure_modes`, `needs_cu_feedback`
- 既存 `column_requests`
- `guardrails` と件数系 `state_counts`

設計上の重要点は、replanner が「既存 silver field へ無理に押し込む」だけでなく、**より適した CU field 設計を `column_requests` として提案して止まる**選択肢を持つこと。つまり T6 の外側ループは「何とか答える」だけでなく、「暫定回答 + 建設的な CU 改善提案」を first-class に扱う。

Operation ごとの default step 列：

| operation | step 列 |
|---|---|
| `aggregate` | `bm25_search → query_silver → aggregate_silver → fetch_chunks → review_schema_gaps` (5 step) |
| `filter` | `query_silver → bm25_search → fetch_chunks → review_schema_gaps` (4 step) |
| `search` | `bm25_search(promote_records=True) → fetch_chunks → review_schema_gaps` (3 step) |

`validate_steps` が **operation→tool 順序の不変条件**を強制（aggregate なら `query_silver` を `aggregate_silver` の前に自動挿入、`retrieve_evidence=True` なら `fetch_chunks` を末尾に自動追加）。

### 5.2 plan/replan trace

QU answer JSON には最終 `plan` に加えて `plan_iterations[]` が残る。各要素は `iteration`, `plan`, `observation`, `judgement`, `best_partial_answer` を持つ。trace には次の構造化 event が追加される：

| trace tool | 意味 |
|---|---|
| `plan_query:<planner>` | iteration 0 の初期計画 |
| `plan_observation` | judge 後に replan へ渡した観測要約 |
| `replan_query:<planner>` | iteration >0 の更新計画 |
| 任意 tool event の `arguments.plan_iteration` | どの plan iteration で実行されたか |

終了判定は planner 自己申告ではなく **answer_judge をループ内に残す**。これは RLM の `answer["ready"]` に相当する自己終了だけへ寄せず、planner と judge の役割分離を保つため。

### 5.3 Retrieval Backend

`SilverCorpus` に 4 メソッド：

| メソッド | 実装 |
|---|---|
| `lexical_overlap_search_chunks` | 旧 term-overlap、`--retrieval-mode lexical` のみ |
| `bm25_search_chunks` | **bm25s (numpy only)** + Python `FallbackBM25Index` |
| `embedding_search_chunks` | OpenAI `text-embedding-3-small` + cosine、`<corpus>/.qu_agent_rlm/<model>.json` cache |
| `hybrid_search_chunks` | rank-position zipper merge（後方互換、本流の fusion は agent state レイヤ） |

#### BM25F-lite document composition

```python
def compose_bm25_text(chunk_text, field_text, account_text):
    # global IDF、chunk_text ×2 で「フィールド重み」を tf bag 重複で近似
    return "\n".join([chunk_text, chunk_text, field_text, account_text])
```

`field_text` には `catalog.fields[*].search.filterable=True` のものだけが入る。embedding 側は別 composition：

```python
return f"account: {account_text}\nsilver_fields: {field_text}\ntranscript_chunk: {chunk_text}"
```

**BM25 explainability**：すべての BM25 結果が `query_terms / matched_terms / missing_terms / score_details.terms[{term, tf, df, idf, contribution}]` を返す。LLM controller / reranker / judge にこの内訳が渡る。

#### Embedding cache invalidation

`documents_hash = SHA256(model + sorted[(chunk_id, embedding_text)])` で hash 管理。chunk text、catalog の filterable フィールド、silver 値、モデル名のいずれが変わっても hash 不一致で自動 re-embed。

### 5.4 Retrieval Subagent

`AgenticRetrievalSubAgent` が `--search-controller` を置き換え（旧フラグは Deprecated）。`--retrieval-subagent openai-compatible --llm-base-url <sid1-endpoint>` で SID-1 風の検索専用モデルを差し込める。

```python
class SearchExecutionPolicy:
    min_calls: int = 1
    max_iterations: int = 0
    query_diversity_threshold: float = 0.8
```

`--retrieval-mode agentic` で自動的に `min_calls=2`, `max_iterations=2` にアップグレード。`validate_search_diversity` が：

- 同じ `(tool, normalize_query)` ペア → reject
- 同 tool 内で Jaccard term overlap > 0.8 → reject

reject 時は `forced_search_iteration` で **過去の `missing_terms` をクエリ語に転換**した `diversified_query` を試す。BM25 explainability が effective に「外した語を次に試す」という探索戦略を可能にしている。

### 5.5 Reranker

`rerank_search_results` が各 plan iteration の step 終了後に LLM rerank を実行。`state["chunks"]` を上書きし、`rerank_score` / `rerank_reason` を付与。失敗時は元の順序を維持して `llm_rerank:error` を trace に積むだけ。

### 5.6 Answerability Judge

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

LLM judge が default で常時稼働し、外側 planner loop の各 iteration 末尾で answerability を判定する。`HeuristicAnswerJudge` は JSON validation failure や API failure 時の reliability fallback と smoke test helper であり、CLI/demo の agent intelligence mode ではない。LLM judge は `--judge-model / --judge-base-url / --judge-api-key` で planner とモデルを分離可能（self-bias / 共謀防止のための運用 path）。

---

## 6. CU↔QU Feedback Loop

### 6.1 Loop の形状

```
[Query Source Pool]                  
  ├ curated         (hand-labeled eval 候補)
  ├ production_log  (本番運用ログ)
  ├ failed_query    (過去 column_requests 由来)
  ├ synthetic_llm   (qu.downstream_query_bootstrap 経由)
  └ synthetic_heuristic (smoke-only helper)
            ↓ dedupe + adjusted_for snapshot
[query_tasks.jsonl] (first-class artifact)
            ↓ N query を replay
[QU agent] × N
            ↓ 各回答に judgement 付き
[column_requests + search_diagnostics + judgement.missing_field_requests + prompt_repair_request]
            ↓ append-only JSONL
[CU 次回起動: --feedback-input]
            ↓ summarize_feedback (3 段フォールバック型集約)
[FeedbackAwareSchemaInducer]
            ↓ apply_feedback_to_specs
[catalog v2]   "QU feedback refinement" provenance マーカー付き
            ↓ extract + promotion gate
[silver v2]
            ↓ feedback_report.json に accepted/promoted 記録
[QU 再実行] 同じ query_tasks.jsonl を replay
            ↓
[orchestration_trace.jsonl] で before/after を集計
```

### 6.2 3 つの逆チャネル

1. **`column_requests`** — QU の planner / replanner / `review_schema_gaps` step が直接出す
2. **`search_diagnostics.failures`** — BM25 explainability の `missing_terms` から自動生成
3. **`judgement.missing_field_requests`** — answer judge が自律的に CU 改善を提案

`summarize_feedback` が 3 チャネルを priority-merge し、同名 field は `count` を加算。replanner は観測済みの `column_requests` を見て次の計画を立てるため、同じクエリ内でも「追加検索で回収する」「別フィルタへ切り替える」「CU に新フィールドを提案して止まる」を分岐できる。

### 6.3 Reliability fallback の Feedback-aware フィルタ検出

`FeedbackAwareSchemaInducer` が追加した field には `usage.good_for: "QU feedback refinement"` という provenance マーカーが入る。`detect_feedback_field_filters` がこれを検出し、LLM planner が失敗して heuristic fallback へ退避した場合でも runtime で追加された feedback 列を generic な term overlap で filter 候補化できる。これは demo/prod の主経路ではなく、LLM failure 時に feedback loop の最低限の観測性と継続性を保つ reliability layer。

---

## 7. Orchestration Layer

### 7.1 `scripts/qu_cu_loop_demo.py`

1 コマンドで full loop を実行：

```bash
python3 scripts/qu_cu_loop_demo.py --output-root output/qu_cu_loop_demo
```

7 phase（loop_id で全体を縛り、iteration で再実行を区別）：

| phase | agent | event_type |
|---|---|---|
| `query_bootstrap` | qu | `query_tasks.created` |
| `baseline_cu` | cu | `cu.run_completed` |
| `baseline_qu` | qu | `qu.run_completed` |
| `feedback` | qu | `qu.feedback_emitted` |
| `refined_cu` | cu | `cu.run_completed` |
| `refined_qu` | qu | `qu.run_completed` |
| `loop_evaluation` | orchestrator | `loop.evaluated` |

### 7.2 出力 artifact

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
│   ├── query_tasks.jsonl       ← 一次クエリの集合
│   ├── query_bootstrap.json    ← 生成過程の audit
│   ├── baseline_answers/
│   │   └── <task_id>.json      ← 各 query の回答 + judgement + trace
│   └── column_requests.jsonl   ← QU からの逆チャネル
├── 03_cu_refined/
│   └── (01 と同形式、feedback_report.json に accepted/promoted 状態)
├── 04_qu_refined/
│   └── refined_answers/<task_id>.json
├── loop_summary.json           ← before / feedback / schema_delta / after / loop_result
└── orchestration_trace.jsonl   ← 7 phase の単一イベント列
```

---

## 8. Observability Stack

| artifact | 役割 | 主な内容 |
|---|---|---|
| `manifest.json` | CU root entry | dataset_id, record/chunk count, `schema_induction`, `feedback_refinement`, `prompt_state` |
| `silver_schema_catalog.json` | QU 用の query 契約 | fields[].search/evidence/usage/quality |
| `silver_calls.jsonl` | 構造化された value + evidence_refs | fields, evidence_refs, quality_flags |
| `chunks.jsonl` | 検索可能チャンク | chunk_id, text, snippet |
| `rlm_trace.jsonl` | CU 内部 trace | root / sub-rlm の tool event |
| `feedback_report.json` | QU→CU feedback の処理結果 | accepted_into_schema, promoted_to_silver, validation_error_counts |
| `evaluation_tasks.json` | CU 自己生成 smoke task | self-referential、外部 fixture ではない |
| `databricks_contract.json` | 本番接続契約 | allowlisted_tools, sql_policy |
| QU answer JSON | クエリ単位の結果 | answer, plan, plan_iterations, records, aggregation, evidence, judgement, column_requests, search_diagnostics, rerank, prompt_state, usage_summary, guardrails, best_partial_answer, trace |
| `column_requests.jsonl` | QU→CU 逆チャネル | append-only、source_diagnostics と judgement を同梱 |
| `prompt_repair_request.jsonl` | prompt 改善 signal | **signal-only**、`action: collect_signal_only`、`promotion_gate: external_hand_labeled_eval_required` |
| `query_tasks.jsonl` | 一次クエリ集合 | task_id, source_type, label_status, generation_id, adjusted_for, provenance |
| `query_bootstrap.json` | 合成クエリの生成 audit | generator, model, prompt metadata, coverage_notes |
| `orchestration_trace.jsonl` | outer loop の単一イベント列 | loop_id, iteration, phase, agent, metrics, decision |

---

## 9. 共通 Pattern：LLM-first Pluggable Component

ほぼすべての semantic component は同じ `JSONChatClient` Protocol で実装され、CLI/demo は LLM backend のみを user-facing mode として公開する。heuristic は agent の代替知能ではなく、validator/API failure 時の reliability fallback と smoke test のための直接注入 helper として残す。

| Component | user-facing mode | フラグ |
|---|---|---|
| Schema inducer | openai / openai-compatible | `--schema-inducer` |
| Field extractor | openai / openai-compatible | `--extractor` |
| Query planner | openai / openai-compatible | `--planner` |
| Retrieval subagent | default / none / openai / openai-compatible | `--retrieval-subagent` |
| Reranker | none / openai / openai-compatible | `--reranker` |
| Answer judge | openai / openai-compatible | `--judge` |
| Query bootstrapper | openai / openai-compatible | `--bootstrap-queries` |

`auto` と no-key demo fallback は廃止。`--retrieval-mode agentic` の `default` retrieval subagent は OpenAI controller を使う。`none` は retrieval/reranker のような optional control plane でのみ残し、semantic判断（planner/schema/extractor/judge/query bootstrap）には使わない。

LLM 経路にはすべて：

- 入力 validator（catalog 整合性 / chunk_id ホワイトリスト / allowed_values チェック）
- fallback orchestration（`<provider>->fallback:<base>` の provenance / validation error を残す）
- `prompt_id + version + hash` の trace 記録

が乗っている。

---

## 10. Promotion Gate と探索の分離

### 10.1 設計原則

「**方策パラメータは探索する、安全規約は人間が握る**」を 3 層に分けて実装：

| 層 | 例 | 自動探索可？ |
|---|---|---|
| 観測空間 | catalog の fields | ✅（evidence_coverage gate で自動 promotion） |
| 方策パラメータ | prompts, query distribution | ⚠ 提案蓄積のみ、promotion は外部 fixture 待ち |
| 安全規約 | validator 規約、`PROMOTION_GATE` 定数、`evidence_coverage >= 0.75` 値 | ❌ コードのみ、自動変更しない |

### 10.2 `PROMOTION_GATE` 定数

```python
PROMOTION_GATE = "external_hand_labeled_eval_required"
```

`qu/prompt_repair.py` と `cu/prompt_repair.py` で同一文字列定数として共有。すべての `prompt_repair_request` の `promotion_gate` フィールドに焼き込まれており、**「なぜ promotion しないか」がデータ自身に書かれている**。

### 10.3 Bootstrap の Early Stopping 戦略

issue FB-002 の design direction：

> Prefer early stopping during bootstrap: use self-improvement for 70-80 point startup quality, then rely on production query/outcome logs and external fixtures for higher-confidence prompt/model changes.

→ 自己改善ループは bootstrap の 70-80 点まで使い、それ以降は **production query/outcome log** と **EVAL-001 の外部 fixture** に頼る、という二段戦略が明文化されている。

### 10.4 Judge model 分離

`--judge-model / --judge-base-url / --judge-api-key / --judge-timeout-seconds`（環境変数 `JUDGE_OPENAI_*`）で judge を planner と別モデルにできる。**bootstrap 期は共用、production は別モデル**という運用 path が想定されている。

---

## 11. Issue Backlog 現況

`cu-agent-rlm/docs/issues.md` を要約：

| ID | タイトル | 状態 |
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
| **EVAL-001** | **External evaluation fixtures** | **Open（FB-002 の前提）** |
| DBX-001 | Databricks production tool adapters | Open |
| OBS-001 | Budget, observability, replay | Done（usage/cost, latency, prompt hash, fallback reason, replay redaction） |
| FB-001 | Model-feedback and prompt-improvement loop | Partial（schema 改善まで） |
| **FB-002** | **Prompt lab and eval-gated prompt promotion** | **Blocked on EVAL-001** |
| DE-001 | CU bootstrap context pack | Design enhancement |
| DE-002 | CU-QU schema negotiation loop | Design enhancement |
| DE-003 | Programmatic RLM sub-agent orchestration | Partial |
| DE-004 | Feedback routing beyond schema changes | Design enhancement |
| DE-005 | Semantic recall evaluation and drift control | Design enhancement |
| DE-006 | Schema lifecycle, backfill, and governance | Design enhancement |
| DE-007 | Human review and policy hooks for high-impact inferences | Design enhancement |

**critical path**：EVAL-001（外部 hand-labeled fixture）が解決すれば FB-002 が unblock し、prompt の自動 promotion が可能になる。それまではすべての prompt change は manual review、すべての synthetic query は `label_status="unlabeled"` で promotion gate 外。

---

## 12. 設計原則まとめ

1. **Tool boundary は 5 ツール固定** — `call_records` 本体を LLM に直接渡さない。すべての参照は `chunk:<id>` 経由
2. **3 本柱はすべて versioned identity + promotion gate** — catalog / prompts / queries の lifecycle を統一
3. **LLM 経路にはすべて validator + fallback + provenance** — `<provider>->fallback:<base>` の trace、prompt_id + hash、`<base>+feedback` の合成 provenance 文字列
4. **frozen=True は「自動編集禁止」の型保証** — prompts は新インスタンスを書く（version bump する）以外に変更経路がない
5. **`PROMOTION_GATE` 文字列をデータに焼き込む** — 「なぜ promotion しないか」を artifact 自身に残す
6. **Self-eval bias を運用で分離** — judge と planner の model 分離フラグ、`label_status="unlabeled"` の synthetic は promotion gate 外
7. **LLM-first + reliability fallback** — semantic 判断は LLM が担当し、heuristic は validation/API failure と smoke test だけを支える
8. **観測してから次手を変える** — QU は `plan → execute → judge/observe → replan` を有界ループ化し、成功判定は judge に残す
9. **Append-only signal logs** — `column_requests.jsonl`, `prompt_repair_request.jsonl`, `orchestration_trace.jsonl` は時系列で蓄積、後段で集計
10. **Deterministic control は残す** — BM25 explainability、query diversity、forced search iteration、smoke eval は system reliability layer として維持
11. **観測可能性は構造化 JSON で出す** — trace の tool 名に backend / provider / fallback chain をすべて文字列展開

---

## 付録 A：ファイル別の責務

### cu-agent-rlm

| ファイル | 責務 |
|---|---|
| `pipeline.py` | 7 ステージのオーケストレーション |
| `models.py` | dataclass 定義（CallRecord, Chunk, FieldSpec, FieldExtraction, SilverCallRecord, TraceEvent, ContentUnderstandingArtifact） |
| `io.py` | JSONL/CSV load + 全 artifact の write |
| `chunking.py` | turn を `max_chars=900` で chunk 化 |
| `fields.py` | レガシー FIELD_SPECS と LIST_RULES（StaticSchemaInducer の互換用） |
| `schema.py` | SchemaInducer Protocol + 3 実装 |
| `extraction.py` | FieldExtractor Protocol + 2 実装 + validator |
| `feedback.py` | FeedbackAwareSchemaInducer + summarize_feedback + build_feedback_report |
| `prompt_registry.py` | PromptSpec(frozen=True) と CU 側 prompt 定義 |
| `prompt_repair.py` | prompt_repair_request.jsonl の生成（signal-only） |
| `llm.py` | OpenAIResponsesClient / OpenAICompatibleChatClient |
| `cli.py` | CLI argparse + 各コンポーネントの factory |

### qu-agent-rlm

| ファイル | 責務 |
|---|---|
| `agent.py` | QueryUnderstandingAgent（bounded plan/execute/judge/replan loop + 多段 step executor） |
| `corpus.py` | SilverCorpus（4 retrieval method） |
| `retrieval.py` | bm25s wrapper + FallbackBM25 (explainer) + EmbeddingIndex + BM25F-lite composition |
| `retrieval_agent.py` | AgenticRetrievalSubAgent（controller + reranker + diversity 検証） |
| `planner.py` | QueryPlan / QueryToolStep / ColumnRequest + Heuristic/LLM planner/replanner + 7 ALLOWED_STEP_TOOLS の validator |
| `judge.py` | AnswerJudge Protocol + Heuristic / LLM / Noop |
| `query_tasks.py` | QueryTask normalization + LLMDownstreamQueryGenerator + smoke-test heuristic bootstrap helper |
| `prompt_registry.py` | PromptSpec(frozen=True) と QU 側 prompt 定義（planner / replanner / retrieval / judge） |
| `prompt_repair.py` | prompt_repair_request.jsonl の生成（signal-only） |
| `llm.py` | LLM クライアント定義（CU と重複維持、依存性分離のため） |
| `cli.py` | CLI argparse + planner/judge/subagent/reranker の factory |
| `env.py` | `.env` ファイルローダー |
| `eval.py` | evaluation_tasks.json を実行する smoke runner |

### scripts

| ファイル | 責務 |
|---|---|
| `qu_cu_loop_demo.py` | CU↔QU orchestration runner、`query_tasks` + `orchestration_trace` を含む完全実行 |
| `qu_user_query_demo.py` | 単発ユーザクエリの実行 demo（default で agentic retrieval + OpenAI planner/reranker/judge） |

---

**最終更新**: 2026-06-01 / **schema_version**: `silver.rlm.v1` / **prompt baseline**: `2026-06-01.1`
