# RLM 取り込み TODO（cu-agent / qu-agent loop）

別の実装者がそのまま着手できる粒度に落とした作業リスト。RLM（`rlm-minimal` / `rlm` フル版）の機序解説と、cu/qu の**実コード監査**、および `cu-agent-rlm/docs/issues.md` のバックログを統合して導出した。

- **導出元（読む順）**
  1. `rlm/rlm-minimal/docs/rlm-minimal-explained.md` … pass-by-reference / CodeAct / model-driven topology の最小核
  2. `rlm/rlm/docs/walkthrough/rlm-explained.md` … 本番化6拡張（プロセス分離・能力別4ツール・多段再帰・ガードレール・コスト追跡・compaction）
  3. `docs/architecture.en.md` … cu/qu の設計意図（ただし実態より RLM 寄りに描かれている点に注意）
  4. `cu-agent-rlm/docs/issues.md` … OBS-001 / DE-003 / QU-003 / DE-002 / AGG-001
  5. 本書 … 上記を実コード（`agent.py` / `pipeline.py` / `retrieval_agent.py` / `extraction.py`）と突き合わせた結論

- **凡例**: 優先度 `P0`(基盤・低リスク) > `P1`(並列化) > `P2`(ループ駆動) > `P3`(表現力・再帰)。各項目に *現状(file:line)* / *狙い* / *手順* / *受け入れ条件* / *RLM 参照* / *依存* を付す。

---

## 0. 前提：現実装の到達点と、やらないこと

**実装の事実（コード監査結論）**

- cu/qu は「RLM 風」と描かれるが、**実態は LLM をスロットに挿した決定論パイプライン**。LLM はパラメータ（どのフィールド/フィルタ/次クエリ語）を埋めるだけで、**データフローの形もループ駆動も担っていない**。
- すでに RLM 以上に堅い部分：`state` 辞書による pass-by-reference（`agent.py:76-87`）と、evidence_refs を提示 chunk_id に制約する検証器（`extraction.py:146,267`）。**ここは壊さない**。
- 欠落：観測駆動ループ、fan-out/batching、コスト/トークン/予算の可視性。

**取り込まない（現コードの安全設計と矛盾するため）**

| 候補 | 却下理由 |
|---|---|
| 全面 CodeAct（任意 Python REPL） | 安全モデルが「LLM は JSON、validator が whitelist で検証」（`extraction.py:146,267` / `retrieval_agent.py:235`）。任意コードは全検証器をバイパスする。→ T7 で `aggregate_silver` 内に限定導入のみ |
| 多段再帰（depth/max_depth） | CU にサブエージェント構造が無く挿す場所が無い。T6/T8 でループ駆動を作ってからの将来課題 |
| プロセス分離 + ソケット橋渡し（LMHandler） | ツールはローカル成果物上のインプロセス関数。隔離境界が不要 |

---

## P0 — 基盤（即効・低リスク・コードに完全欠落）

> **実装状況（2026-06-01）: T1 ✅ / T2 ✅ / T3 ✅ 完了**。`usage.py`（両パッケージ）・`replay.py`（両パッケージ）・`GuardrailState`（`pipeline.py` / `agent.py`）・CLI/scripts への配線・ユニットテスト追加済み。検証: CU 12/12・QU 20/20 PASS、実データでの end-to-end スモーク通過。
> - 当初 T2 で未達だった **replay redaction** を実装: `redact_for_replay`（`snippet/text/best_text/transcript/...` を `[redacted]` 化、構造は保持）。QU は各回答に `*.replay.json` を既定出力（`--no-replay` で抑止）、CU は `rlm_trace.replay.jsonl` を出力。
> - **pricing 可視化**: `budget_unpriced_warning` を全 CLI/scripts で stderr 警告。metrics に `llm_cost_basis`(=`pricing.source`) を追加し、`--max-budget-usd` が価格未設定で no-op になる事象を可視化。
> - 既存課題だったテストフィクスチャの stale パス（`cu-agent/data/...`）を `data/` 相対へ修正、`call_records.sql` 未追跡に依存しないよう `observed_columns` を SQL 無しでも baseline 列を返す形に変更。

### T1. LLM 呼び出しのコスト/トークン計測（`UsageSummary` 相当）

- **現状**: コスト・トークン計測が**両パッケージに皆無**（`grep usage|token|cost|budget` のヒットはカタログ metadata とトークナイザのみ）。CU は N コール×1 LLM、QU は agentic 時に検索ループが回るのに可視性ゼロ。
- **狙い**: 全 LLM パスの calls/input/output tokens/cost をモデル別に集計し、ループ単位で親に積み上げる。OBS-001 の前提基盤。
- **手順**:
  1. `cu/llm.py` と `qu/llm.py` の `JSONChatClient`（`complete_json` / `OpenAIResponsesClient` 等）に、レスポンスの usage を拾う薄いフックを追加。
  2. RLM の `UsageSummary` / `ModelUsageSummary`（モデル別 `total_calls/input/output/cost`）に倣ったデータクラスを共通定義（両パッケージで重複定義は現状の dependency isolation 方針を踏襲）。
  3. `QueryUnderstandingAgent.answer()` と CU `analyze_calls()` の戻り（`result_payload` / `ContentUnderstandingArtifact.manifest`）に `usage_summary` を載せる。
  4. 子の集計を親へ合算（`_cumulative_cost` 相当）。
- **受け入れ条件**: 1クエリ / 1 CU ラン後に「モデル別 calls/tokens/cost」が成果物 JSON に出る。`orchestration_trace.jsonl` の metrics に cost/tokens 列が入る。
- **RLM 参照**: フル版 §10（`UsageSummary` / `token_utils`）。
- **依存**: なし（最初に着手すべき）。

### T2. 統一トレース：prompt hash / latency / fallback 理由 / budget counter（OBS-001）

- **現状**: trace は軽量。`prompt_state` に prompt メタは載るが、**latency・token・budget・fallback 理由が一貫して記録されない**。統一トレースは loop demo ランナー限定で、全 CLI パス共通のライブラリではない（issues.md OBS-001 "Remaining gaps"）。
- **狙い**: 全 LLM 呼び出しを `prompt_hash / model / latency / tokens(=T1) / validation_result / fallback_reason / budget_counter` で記録。replay 用に生テキストを既定で除外。
- **手順**:
  1. `ToolEvent` / `TraceEvent` に `latency_ms` / `tokens` / `fallback_reason` / `prompt_hash` を追加。
  2. 既存の fallback 経路（`extraction.py:70-77` の `llm_fallback:`、`retrieval_agent.py:112-120` の `search_iteration:error`、`judge` の `->fallback:heuristic`）を**構造化フィールド**に統一。
  3. demo ランナー専用の trace 生成を共通ヘルパへ抽出し、単発 CLI（`qu_user_query_demo.py` 等）からも使えるようにする。
  4. replay 成果物は snippet 化済み（chunk:ref）データのみ含め、生 transcript を除外。
- **受け入れ条件**: 任意の CLI パスで「LLM 1呼び出し = 1構造化 trace 行（hash/model/latency/tokens/fallback）」。replay パッケージに生 transcript が含まれない。
- **RLM 参照**: フル版 §12（trajectory / `RLMLogger`）。
- **依存**: T1（tokens/cost 列）。

### T3. ガードレール：`max_errors` / `max_budget` / `max_timeout` + `best_partial_answer`

- **現状**: 大域的な打ち切り回路が無い。CU 抽出は per-call fallback はあるが「連続失敗で停止」が無い（`pipeline.py:115`）。QU 検索ループは多様性ゲートのみでエラー/予算ゲートが無い（`retrieval_agent.py`）。
- **狙い**: 連続エラー数・累計コスト・経過時間で安全に打ち切り、**打ち切っても直近最良を返す**。
- **手順**:
  1. RLM の `_check_iteration_limits` に倣い、CU の per-call ループと QU の検索ループに「連続エラー閾値（成功でリセット）/ 予算（=T1）/ タイムアウト」を入れる。
  2. `_best_partial_answer` 相当を QU `state` と CU 抽出蓄積に持たせ、超過時に直近最良を返す。
  3. QU は超過を judge の `failure_modes` に新値（`budget_exceeded` / `timeout`）として流し、観測を一元化。
- **受け入れ条件**: 予算/時間超過で例外死せず、`judgement.failure_modes` または CU trace に理由付きで部分結果が残る。
- **RLM 参照**: フル版 §3「ガードレール」。
- **依存**: T1。

---

## P1 — 並列化（トポロジ変更不要の素直な fan-out）

> **実装状況（2026-06-01）: T4 ✅ / T5 ✅ 完了**。検証: CU 13/13・QU 21/21 PASS、ヒューリスティック実データで逐次↔並列の抽出結果一致を end-to-end 確認。
> - **T4**: `LLMFieldExtractor.extract_batch`（bounded `ThreadPoolExecutor`、入力順保持）と汎用ディスパッチ `run_call_extractions`（`extract_batch` 非対応の抽出器は決定論的逐次フォールバック）を追加。`pipeline.analyze_calls` を「バッチ単位で実行→各 outcome を観測」に再構成し、T3 ガードレール（連続エラー/予算/タイムアウト）と T1 usage を並列下でも保持。CLI に `--batch-max-concurrent`(既定8、1で逐次=従来同一)。usage 集計は `UsageSummary.add_call` を `threading.Lock` で直列化し、`complete_json_with_usage` で per-call トークンを正確に trace。
> - **T5**: 検索ステップに `queries` 配列（map-reduce fan-out）を導入。`execute_step` の検索分岐が `run_subquery_searches` で並列実行し、既存 `merge_chunks`/`merge_records` で重複排除統合、各サブ検索を個別 trace（`fanout` 注記付き）。後方互換：単一クエリは要素1として従来挙動を完全保持。多様性検証は `diverse_subqueries`（fan-out 内）と `validate_search_diversity`（サブクエリ間にも適用）で二重化。RETRIEVAL_CONTROLLER / QUERY_PLANNER プロンプトに fan-out 指南（太いプロンプト×小さいバッチ、~20上限、極小×巨大はアンチパターン）を追記しバージョン更新。差集合系クエリ（「Xに見えるが Y が無い」）が1プランの2サブクエリとして表現可能に。

### T4. CU per-call 抽出の batching / 並列化（map）

- **現状**: `for call in calls:` で**完全逐次**（`pipeline.py:115`）。各コールは独立＝embarrassingly parallel なのに直列。各コールの抽出は単発 `complete_json`（`extraction.py:59-77`）。
- **狙い**: RLM の `llm_query_batched`（並行実行）に倣い、独立な per-call 抽出を**バウンド付き並列**（semaphore ~16）で実行。
- **手順**:
  1. `FieldExtractor` Protocol に `extract_batch(calls, chunks_by_call, specs)` を追加（既定は逐次フォールバック）。
  2. `LLMFieldExtractor` で `asyncio` or ThreadPool による並行 `complete_json`、`batch_max_concurrent` を CLI 引数化。
  3. per-call の検証器（`validate_extraction_payload`）と fallback はそのまま各タスク内で維持。
  4. T1/T3 と結線（並列でも cost 合算・エラー予算が効くこと）。
- **受け入れ条件**: 同一入力で結果が逐次版と一致し、壁時計時間が短縮。並列上限が設定可能。失敗コールのみ fallback。
- **RLM 参照**: フル版 §4「バッチ並列」/ §6（`*_batched`）。
- **依存**: T1, T3。

### T5. QU 検索の map-reduce（1クエリ → 複数サブクエリ並列 → merge）

- **現状**: 検索は単数クエリ単発。`forced_search_iteration` は決定論フォールバックのみ（`retrieval_agent.py:247`）。「分割して並列検索→統合」が表現できない。
- **狙い**: planner / retrieval controller が**複数サブクエリを宣言 → 並列検索 → `state["chunks"]` 上で merge**できるようにする。QU-003 の矛盾クエリ「Xに見えるが Y が無い」は2検索の差集合として fan-out で表現可能になる。
- **手順**:
  1. `QueryToolStep` を「単一クエリ」から「サブクエリ配列」を許す形に拡張（後方互換：単一は要素1の配列）。
  2. `execute_step` の検索分岐で配列を並列実行し `merge_chunks` で統合（既存 merge を流用）。
  3. RETRIEVAL_CONTROLLER_PROMPT に「太いプロンプト×小さいバッチ（fan-out ~20 上限）/ 極小×巨大はアンチパターン」のオーケストレーション指南を追記。
  4. 多様性検証器（`validate_search_diversity`）はサブクエリ間にも適用。
- **受け入れ条件**: 1プランで複数サブクエリが並列実行され、trace に各サブ検索が記録され、結果が重複排除統合される。差集合系クエリが1プランで解ける。
- **RLM 参照**: minimal §5.3（map-reduce）/ フル版 §9（`ORCHESTRATOR_ADDENDUM`: 容量/fan-out 指南）。
- **依存**: T1, T3。

---

## P2 — ループ駆動（LLM をパラメータ係から orchestrator へ昇格）

> **実装状況（2026-06-01）: T6 ✅ 完了**。QU `answer()` を bounded `plan → execute → judge/observe → replan` ループへ再構成。初回 plan は従来どおり温存し、`replan` 対応 planner（現状 `LLMQueryPlanner`）だけが judge 観測後に次 iteration へ進むため、heuristic/static planner の single-plan 動作は維持。観測には records/chunks/search failures/judge failure modes/既存 `column_requests` を圧縮して渡し、trace に `plan_observation` / `replan_query:*` / `plan_iteration` を記録。CLI/scripts に `--max-plan-iterations`（既定2、1で従来相当）を追加。replanner prompt では「既存フィールドへ無理に押し込まず、より適した CU フィールド設計がある場合は constructive `column_requests` として提案する」方針を明示し、暫定回答と schema feedback の両立を保つ。

### T6. 観測駆動の外側ループ：plan 凍結を解く（最重要・設計変更大）

- **現状**: `plan = planner.plan(...)` は**冒頭1回のみ**（`agent.py:53`）。以後 `for step in plan.steps`（`agent.py:88`）を回すだけで、**結果を観測して高レベルの手を変える上位ループが無い**。観測駆動は retrieval 内部に閉じ、しかも既定休眠（`max_iterations=0`, `retrieval_agent.py:16`）。
- **狙い**: RLM の心臓部「応答→実行→**観測**→次手」をプラン全体に引き上げる。「step 群実行 → `state` 観測 → planner 再投入で残り計画更新」のバウンドループ化。
- **手順**:
  1. `answer()` を「plan→execute→observe→(re)plan」の `for iteration in range(max_plan_iterations)` に再構成。
  2. planner に「これまでの `state` 要約（`chunks_for_llm` / `summarize_search_calls_for_llm` の圧縮ビュー）を渡して残りステップを更新」するインターフェースを追加（既存の単発 plan は iteration0 として温存）。
  3. 終了判定は judge を**ループ内**に移すか、planner が `done` を宣言（RLM の `answer["ready"]` 相当だが、判断は別モデル judge に委譲＝現行の自己バイアス回避を維持）。
  4. T3 のガードレールでループを必ず有界化。
- **受け入れ条件**: 「初回検索が空 → planner が別オペレーション/別フィルタに切替えて回収」のような**観測に応じた経路変更**が trace に現れる。固定ステップ数を超える適応が起きる。
- **RLM 参照**: minimal §3（ReAct/CodeAct ループ）/ フル版 §3 + §9（orchestrator として分解計画を先に宣言）。
- **依存**: T1, T2, T3（有界化と観測の前提）。

### T7. `aggregate_silver` 限定 CodeAct（QU-003 / AGG-001 の表現力）

- **現状**: 集計は単一 group-by count（`agent.py:238-251` / `corpus.aggregate_silver`）。date range / 数値比較 / 比率 / top-k / nested group-by / cohort が表現できない（issues.md QU-003, AGG-001）。
- **狙い**: **構造化済み silver（低リスク）に限った式評価 REPL** を `aggregate_silver` 内部に閉じ込め、表現力だけ RLM 的に上げる（全面 CodeAct は §0 の通り却下）。
- **手順**:
  1. silver records（生 transcript ではない）に対する allowlisted な式評価器を実装（`run_sql` SELECT-only と同じ思想の延長）。builtins を制限、`open/__import__/eval/exec` 封鎖（RLM `_SAFE_BUILTINS` を参考）。
  2. planner が「集計式」を生成 → 検証器が許可演算・参照フィールドを catalog の `aggregatable` フラグで制約。
  3. 結果は既存 `aggregation` 形へ正規化し evidence_refs を維持。
- **受け入れ条件**: top-k / 比率 / nested group-by / 数値レンジ集計が、固定ツール境界を壊さず（=検証器を通って）解ける。
- **RLM 参照**: minimal §4(a)（制限 builtins）/ §7（model-driven topology）。
- **依存**: T6（planner がループ内で式を観測・修正できると効果的）。

---

## P3 — 構造化サブエージェント（DE-003 / DE-002、将来）

### T8. 型付きサブエージェント呼び出し I/F（DE-003）

- **現状**: protocol 化されているのは retrieval subagent のみ。CU はサブエージェント構造ゼロ。`orchestration_trace.jsonl` は `agent` 列を既に持つ（将来のプロセス分離に trace 変更不要）。
- **狙い**: RLM の「能力別ツール語彙（flat `llm_query` / recursive `rlm_query`）」に倣い、**typed sub-agent call（入力 artifact ref / 出力スキーマ / 予算 / 検証）**を定義。BM25/embedding/silver/SQL の**並列ブランチ → join/judge** を表現。
- **手順**:
  1. sub-agent call I/F を定義（`input_refs: list[chunk:/silver:]` / `output_schema` / `budget`(=T1) / `validator`）。
  2. T5 の並列検索を「並列 retrieval ブランチ → join → rerank/judge」として再表現。
  3. CU 側に field-candidate clustering / allowed-value ontology mapping / evidence entailment を sub-agent として切り出し（まずは flat、再帰は導入しない）。
- **受け入れ条件**: 並列ブランチ結果が join/judge ステップに集約され、各 sub-call が予算・出力スキーマ検証付きで trace される。
- **RLM 参照**: フル版 §6（能力別4ツール）/ §7（`_subcall`、ただし再帰は P3 外）。
- **依存**: T1, T3, T5, T6。

### T9. CU↔QU スキーマ交渉ループ（DE-002）

- **現状**: QU が column 要求、CU が受理する一方向。フィールドの再利用性・適切な allowed_values・retrieval-only 妥当性の交渉が無い。
- **狙い**: CU が `field_candidates`（想定クエリ/例フィルタ/例集計/evidence 要件/不確実性）を提案 → QU が生成・観測クエリでシミュレート → `promote/defer/retrieval_only/merge_with_existing/needs_human_review` を判定。
- **手順**:
  1. CU induce_schema に「候補提示」段を追加（promote 前）。
  2. QU 側に「候補をミニクエリ集合で評価」ステップを追加（T6 のループに同居可能）。
  3. 却下/保留候補を rationale 付きで永続化し、QU 失敗が蓄積したら再検討。
- **受け入れ条件**: 候補が promote される前に QU の filter/search/aggregate シミュレーションを通過し、判定理由が成果物に残る。
- **RLM 参照**: フル版 §6（子の能力契約を型で開示）の発想を schema 交渉へ転用。
- **依存**: T6, T8。

---

## 依存グラフ（着手順）

```
T1 (usage/cost) ──┬─▶ T2 (unified trace, OBS-001)
                  ├─▶ T3 (guardrails + best_partial)
                  │
T3 ──┬─▶ T4 (CU batched extraction)        [P1]
     └─▶ T5 (QU search map-reduce)          [P1]
                  │
T2,T3 ──▶ T6 (observe-then-replan loop)     [P2] ★最重要
                  │
T6 ──▶ T7 (bounded CodeAct in aggregate)    [P3]
T5,T6 ──▶ T8 (typed sub-agent I/F, DE-003)  [P3]
T6,T8 ──▶ T9 (CU-QU negotiation, DE-002)    [P3]
```

**推奨スプリント**: ① T1→T2→T3（基盤）→ ② T4・T5（並列化、独立着手可）→ ③ T6（昇格の核）→ ④ T7/T8/T9（表現力・構造化）。

---

## 付記：プロンプト昇格ガバナンス（RLM 訓練報酬との対比、本 TODO 外）

RLM は「良い分解行動」を**訓練側の報酬（`RLMTrainRubric` の behavior gate: min_iterations/min_subcall を満たさない trajectory は報酬0）**で方策に焼き込む。cu/qu は逆に `frozen=True` + 外部 hand-labeled eval（EVAL-001）でプロンプト変更を人間ゲートに置く設計で、これは**意図的な相違**。RLM の behavior-gate は「十分に委譲・検証した trajectory だけを良しとする」評価軸として **EVAL-001 の fixture 設計の着想元**にはなるが、自動 mutation は cu/qu の安全契約と衝突するため移植しない。FB-002（eval-gated prompt promotion）は EVAL-001 待ちのまま据え置く。

---

**Last updated**: 2026-06-01 / **対象コミット**: main / **典拠**: コード監査（`agent.py` `pipeline.py` `retrieval_agent.py` `extraction.py`）+ `issues.md` + RLM 解説2本
