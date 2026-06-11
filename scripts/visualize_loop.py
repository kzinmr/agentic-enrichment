#!/usr/bin/env python3
"""Reconstruct and visualize the data flow of a completed CU<->QU loop run.

Reads the artifacts produced by ``scripts/qu_cu_loop_demo.py`` (orchestration
trace, per-stage CU manifests, per-query QU answers, schema-negotiation files)
and emits a single self-contained HTML file. No third-party dependencies and no
network access -- all run data is embedded inline so the page is portable.

Usage:
    python3 scripts/visualize_loop.py --run-dir output/verify_t789
    python3 scripts/visualize_loop.py --run-dir output/verify_t789 --output output/verify_t789/loop_dataflow.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Fixed phase order of the outer loop (matches qu_cu_loop_demo.py).
PHASE_ORDER = [
    "baseline_cu",
    "query_bootstrap",
    "baseline_qu",
    "feedback",
    "refined_cu",
    "refined_qu",
    "loop_evaluation",
]
PHASE_LABEL = {
    "baseline_cu": "1. Baseline CU",
    "query_bootstrap": "2. Query Bootstrap",
    "baseline_qu": "3. Baseline QU",
    "feedback": "4. QU→CU Feedback",
    "refined_cu": "5. Refined CU",
    "refined_qu": "6. Refined QU",
    "loop_evaluation": "7. Loop Evaluation",
}
AGENT_OF_PHASE = {
    "baseline_cu": "cu",
    "query_bootstrap": "orchestrator",
    "baseline_qu": "qu",
    "feedback": "qu",
    "refined_cu": "cu",
    "refined_qu": "qu",
    "loop_evaluation": "orchestrator",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path_str: str | None, run_dir: Path) -> str | None:
    if not path_str:
        return None
    try:
        return str(Path(path_str).resolve().relative_to(run_dir.resolve()))
    except (ValueError, OSError):
        return path_str


def summarize_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """Pull the data-flow-relevant signals out of one QU answer payload."""
    judgement = answer.get("judgement") or {}
    plan_iterations = answer.get("plan_iterations") or []
    subagents = answer.get("subagent_diagnostics") or {}
    calls = subagents.get("calls") or []
    branch_count = sum(
        1 for c in calls if (c.get("call") or {}).get("capability") == "retrieval_branch"
    )
    # Aggregation expression (T7), if the planner emitted one.
    expression = None
    used_fields: list[str] = []
    for event in answer.get("trace") or []:
        if event.get("tool") == "aggregate_silver":
            args = event.get("arguments") or {}
            if args.get("expression"):
                expression = args.get("expression")
                used_fields = args.get("used_fields") or []
                break
    diagnostics = answer.get("search_diagnostics") or {}
    return {
        "query": answer.get("query"),
        "operation": (answer.get("plan") or {}).get("operation") or "?",
        "record_count": len(answer.get("records") or []),
        "evidence_count": len(answer.get("evidence") or []),
        "search_call_count": diagnostics.get("search_call_count")
        or diagnostics.get("total_search_calls"),
        "search_failure_count": diagnostics.get("search_failure_count")
        or diagnostics.get("failed_search_calls"),
        "judge_success": bool(judgement.get("success")),
        "judge_confidence": judgement.get("confidence"),
        "needs_cu_feedback": bool(judgement.get("needs_cu_feedback")),
        "failure_modes": judgement.get("failure_modes") or [],
        "plan_iterations": len(plan_iterations),
        "fanout_branches": branch_count,
        "subagent_joins": len(subagents.get("joins") or []),
        "aggregation_expression": expression,
        "aggregation_used_fields": used_fields,
        "column_requests": [c.get("field_name") for c in (answer.get("column_requests") or [])],
        "usage": answer.get("usage_summary") or {},
    }


def build_model(run_dir: Path) -> dict[str, Any]:
    trace = read_jsonl(run_dir / "orchestration_trace.jsonl")
    loop_summary = read_json(run_dir / "loop_summary.json") or {}

    # Group orchestration events by phase.
    events_by_phase: dict[str, list[dict[str, Any]]] = {p: [] for p in PHASE_ORDER}
    loop_id = None
    model_name = None
    for ev in trace:
        loop_id = loop_id or ev.get("loop_id")
        events_by_phase.setdefault(ev.get("phase"), []).append(ev)
        dec = ev.get("decision") or {}
        boot = dec.get("bootstrap") or {}
        model_name = model_name or boot.get("model")

    # CU stage manifests (T1 usage, schema sizes, contracts).
    def cu_stage(stage_dir: str) -> dict[str, Any]:
        manifest = read_json(run_dir / stage_dir / "manifest.json") or {}
        negotiation = read_json(run_dir / stage_dir / "schema_negotiation.json") or {}
        candidates = read_json(run_dir / stage_dir / "field_candidates.json") or []
        catalog = read_json(run_dir / stage_dir / "silver_schema_catalog.json") or {}
        decisions: dict[str, int] = {}
        for item in negotiation.get("decisions") or []:
            key = item.get("decision") or "?"
            decisions[key] = decisions.get(key, 0) + 1
        fields = catalog.get("fields") or manifest.get("fields") or []
        return {
            "dir": stage_dir,
            "usage": manifest.get("usage_summary") or {},
            "schema_field_count": len(fields),
            "field_names": [f.get("name") or f.get("field_name") for f in fields],
            "negotiation_decisions": decisions,
            "negotiation_values": negotiation.get("decision_values") or [],
            "candidate_count": len(candidates),
            "has_extraction_contract": (run_dir / stage_dir / "extraction_contract.json").exists(),
        }

    baseline_cu = cu_stage("01_cu_baseline")
    refined_cu = cu_stage("03_cu_refined")

    # Per-query answers (baseline + refined).
    def stage_answers(stage_dir: str, sub: str) -> dict[str, dict[str, Any]]:
        folder = run_dir / stage_dir / sub
        out: dict[str, dict[str, Any]] = {}
        if folder.exists():
            for f in sorted(folder.glob("*.json")):
                if f.name.endswith(".replay.json"):
                    continue
                out[f.stem] = summarize_answer(read_json(f) or {})
        return out

    baseline_answers = stage_answers("02_qu_feedback", "baseline_answers")
    refined_answers = stage_answers("04_qu_refined", "refined_answers")

    # QU-side candidate evaluation (T9).
    candidate_eval = read_json(run_dir / "02_qu_feedback" / "field_candidate_evaluation.json") or {}
    cand_eval_decisions: dict[str, int] = {}
    for item in candidate_eval.get("evaluations") or []:
        key = item.get("decision") or "?"
        cand_eval_decisions[key] = cand_eval_decisions.get(key, 0) + 1

    # Query task ordering / metadata.
    tasks = (loop_summary.get("query_tasks") or {}).get("tasks") or []
    task_meta = {t.get("task_id"): t for t in tasks}

    # Build stage cards for the spine.
    stages = []
    for phase in PHASE_ORDER:
        evs = events_by_phase.get(phase) or []
        inputs: dict[str, str] = {}
        outputs: dict[str, str] = {}
        metrics: dict[str, Any] = {}
        for ev in evs:
            for k, v in (ev.get("input_artifacts") or {}).items():
                inputs[k] = rel(v, run_dir)
            for k, v in (ev.get("output_artifacts") or {}).items():
                outputs[k] = rel(v, run_dir)
        # Aggregate metrics per phase.
        if phase in ("baseline_cu", "refined_cu") and evs:
            metrics = evs[0].get("metrics") or {}
        elif phase in ("query_bootstrap", "feedback", "loop_evaluation") and evs:
            metrics = evs[0].get("metrics") or {}
        elif phase in ("baseline_qu", "refined_qu"):
            metrics = {
                "task_count": len(evs),
                "judge_success": sum(1 for e in evs if (e.get("metrics") or {}).get("judge_success")),
                "needs_feedback": sum(
                    1 for e in evs if (e.get("metrics") or {}).get("needs_cu_feedback")
                ),
                "llm_calls": sum((e.get("metrics") or {}).get("llm_call_count") or 0 for e in evs),
            }
        stages.append(
            {
                "phase": phase,
                "label": PHASE_LABEL.get(phase, phase),
                "agent": AGENT_OF_PHASE.get(phase, "orchestrator"),
                "event_count": len(evs),
                "inputs": inputs,
                "outputs": outputs,
                "metrics": metrics,
            }
        )

    # Per-query comparison rows (baseline vs refined).
    query_rows = []
    order = list(task_meta.keys()) or list(baseline_answers.keys())
    for task_id in order:
        meta = task_meta.get(task_id, {})
        query_rows.append(
            {
                "task_id": task_id,
                "query": meta.get("query") or (baseline_answers.get(task_id) or {}).get("query"),
                "intent": meta.get("intent"),
                "source_type": meta.get("source_type"),
                "targets_schema_gaps": meta.get("targets_schema_gaps") or [],
                "baseline": baseline_answers.get(task_id),
                "refined": refined_answers.get(task_id),
            }
        )

    # Totals (T1 roll-up across CU + QU).
    def usage_total(usage: dict[str, Any]) -> tuple[int, int, int, float]:
        return (
            usage.get("total_calls") or 0,
            usage.get("input_tokens") or 0,
            usage.get("output_tokens") or 0,
            usage.get("total_cost_usd") or 0.0,
        )

    total_calls = total_in = total_out = 0
    total_cost = 0.0
    for u in [baseline_cu["usage"], refined_cu["usage"]]:
        c, i, o, cost = usage_total(u)
        total_calls += c
        total_in += i
        total_out += o
        total_cost += cost
    for ans in list(baseline_answers.values()) + list(refined_answers.values()):
        c, i, o, cost = usage_total(ans.get("usage") or {})
        total_calls += c
        total_in += i
        total_out += o
        total_cost += cost

    loop_eval = events_by_phase.get("loop_evaluation") or [{}]
    loop_metrics = (loop_eval[0].get("metrics") or {}) if loop_eval else {}
    schema_delta = ((loop_eval[0].get("decision") or {}).get("schema_delta") or {}) if loop_eval else {}

    return {
        "run_dir": str(run_dir),
        "loop_id": loop_id,
        "model": model_name,
        "stages": stages,
        "query_rows": query_rows,
        "baseline_cu": baseline_cu,
        "refined_cu": refined_cu,
        "candidate_eval_decisions": cand_eval_decisions,
        "candidate_eval_total": len(candidate_eval.get("evaluations") or []),
        "loop_metrics": loop_metrics,
        "schema_delta": schema_delta,
        "totals": {
            "calls": total_calls,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "cost_usd": total_cost,
            "queries": len(query_rows),
        },
        "pricing_source": (baseline_cu["usage"].get("pricing") or {}).get("source"),
    }


# --------------------------------------------------------------------------- #
# HTML rendering
# --------------------------------------------------------------------------- #

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CU↔QU Loop データフロー · __LOOP_ID__</title>
<style>
  :root {
    --cu: #2563eb; --qu: #059669; --orch: #d97706;
    --bg: #0f172a; --panel: #1e293b; --panel2: #273449;
    --ink: #e2e8f0; --muted: #94a3b8; --line: #334155;
    --ok: #22c55e; --warn: #f59e0b; --bad: #ef4444;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
    font: 14px/1.5 -apple-system, "Segoe UI", "Hiragino Sans", Meiryo, sans-serif; }
  header { padding: 22px 28px; border-bottom: 1px solid var(--line); }
  h1 { margin: 0 0 4px; font-size: 19px; }
  h2 { font-size: 15px; margin: 28px 0 12px; letter-spacing: .03em; color: var(--muted); text-transform: uppercase; }
  .sub { color: var(--muted); font-size: 12.5px; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 0 28px 60px; }
  .totals { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }
  .stat { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px; min-width: 120px; }
  .stat .n { font-size: 20px; font-weight: 650; }
  .stat .l { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }

  /* spine */
  .spine { display: flex; flex-direction: column; gap: 0; }
  .stage { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; position: relative; }
  .stage.cu { border-left: 5px solid var(--cu); }
  .stage.qu { border-left: 5px solid var(--qu); }
  .stage.orchestrator { border-left: 5px solid var(--orch); }
  .stage .row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .stage .title { font-weight: 650; font-size: 15px; }
  .badge { font-size: 11px; padding: 2px 8px; border-radius: 999px; font-weight: 600; }
  .badge.cu { background: rgba(37,99,235,.18); color: #93c5fd; }
  .badge.qu { background: rgba(5,150,105,.18); color: #6ee7b7; }
  .badge.orchestrator { background: rgba(217,119,6,.18); color: #fcd34d; }
  .arrow { text-align: center; color: var(--muted); padding: 4px 0; font-size: 18px; }
  .arrow small { display: block; font-size: 11px; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
  .chip { font-size: 11.5px; background: var(--panel2); border: 1px solid var(--line); border-radius: 6px; padding: 3px 8px; color: var(--muted); }
  .chip b { color: var(--ink); font-weight: 600; }
  .chip.in { border-color: #3b4a63; }
  .chip.out { border-color: #3f5170; color: #cbd5e1; }
  .metrics { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12px; color: var(--muted); }
  .metrics b { color: var(--ink); }

  /* tables */
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th, td { text-align: left; padding: 7px 9px; border-bottom: 1px solid var(--line); vertical-align: top; }
  th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .03em; }
  td.q { max-width: 320px; }
  .pill { font-size: 10.5px; padding: 1px 6px; border-radius: 5px; font-weight: 600; }
  .pill.ok { background: rgba(34,197,94,.16); color: #86efac; }
  .pill.bad { background: rgba(239,68,68,.16); color: #fca5a5; }
  .pill.warn { background: rgba(245,158,11,.16); color: #fcd34d; }
  .pill.t { background: rgba(148,163,184,.16); color: #cbd5e1; }
  .delta-up { color: var(--ok); } .delta-down { color: var(--bad); } .delta-zero { color: var(--muted); }
  .mono { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 11.5px; }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 16px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .fieldwrap { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }
  .f { font-size: 11px; padding: 2px 7px; border-radius: 5px; background: var(--panel2); border: 1px solid var(--line); }
  .f.new { background: rgba(5,150,105,.18); border-color: #0c7a5a; color: #6ee7b7; }
  .legend { display:flex; gap:14px; flex-wrap:wrap; font-size:12px; color:var(--muted); margin-bottom:6px;}
  .legend span b{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:middle;}
  details { margin-top: 6px; }
  summary { cursor: pointer; color: var(--muted); font-size: 12px; }
  .tnote { font-size: 11px; color: var(--orch); font-weight: 600; }
  @media (max-width: 760px) { .grid2 { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <div class="wrap" style="padding-bottom:0">
    <h1>CU↔QU ループ データフロー（実 API ランの再現）</h1>
    <div class="sub">loop_id: <span class="mono">__LOOP_ID__</span> · model: <span class="mono">__MODEL__</span> · run: <span class="mono">__RUN_DIR__</span></div>
    <div class="totals">__TOTALS__</div>
    <div class="sub" style="margin-top:8px">__PRICING_NOTE__</div>
  </div>
</header>
<div class="wrap">

  <h2>パイプライン（artifact で連結された 7 フェーズ）</h2>
  <div class="legend">
    <span><b style="background:var(--cu)"></b>CU (Content Understanding)</span>
    <span><b style="background:var(--qu)"></b>QU (Query Understanding)</span>
    <span><b style="background:var(--orch)"></b>Orchestrator</span>
  </div>
  <div class="spine">__SPINE__</div>

  <h2>クエリ別 baseline → refined（QU 回答の推移）</h2>
  <div class="panel" style="overflow-x:auto">__QUERY_TABLE__</div>

  <h2>スキーマ進化（T9 交渉 + feedback 反映）</h2>
  <div class="grid2">__SCHEMA_PANELS__</div>

  <h2>RLM ラダーのシグナル（このランで実際に観測されたもの）</h2>
  <div class="panel">__SIGNALS__</div>

</div>
</body>
</html>
"""


def esc(s: Any) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_totals(m: dict[str, Any]) -> str:
    t = m["totals"]
    stats = [
        (t["queries"], "queries"),
        (f"{m['baseline_cu']['schema_field_count']}→{m['refined_cu']['schema_field_count']}", "schema fields"),
        (t["calls"], "LLM calls"),
        (f"{t['total_tokens']:,}", "tokens (in+out)"),
        (f"${t['cost_usd']:.4f}", "cost (USD)"),
    ]
    return "".join(
        f'<div class="stat"><div class="n">{esc(n)}</div><div class="l">{esc(l)}</div></div>'
        for n, l in stats
    )


def render_chip(kind: str, key: str, val: str | None) -> str:
    if not val:
        return ""
    return f'<span class="chip {kind}"><b>{esc(key)}</b> {esc(val)}</span>'


def render_metrics(phase: str, metrics: dict[str, Any]) -> str:
    if not metrics:
        return ""
    keys_by_phase = {
        "baseline_cu": ["record_count", "chunk_count", "schema_field_count", "validation_error_count", "llm_call_count", "llm_total_tokens"],
        "refined_cu": ["record_count", "schema_field_count", "feedback_request_count", "validation_error_count", "llm_call_count", "llm_total_tokens"],
        "query_bootstrap": ["task_count"],
        "baseline_qu": ["task_count", "judge_success", "needs_feedback", "llm_calls"],
        "refined_qu": ["task_count", "judge_success", "needs_feedback", "llm_calls"],
        "feedback": ["feedback_record_count", "column_request_count", "query_task_count"],
        "loop_evaluation": ["task_count", "feedback_promoted", "judge_success_delta", "column_request_count_delta", "query_uses_refined_filter_count"],
    }
    keys = keys_by_phase.get(phase, list(metrics.keys()))
    parts = []
    for k in keys:
        if k in metrics and metrics[k] is not None:
            parts.append(f"{k.replace('_', ' ')}: <b>{esc(metrics[k])}</b>")
    return '<div class="metrics">' + " · ".join(parts) + "</div>" if parts else ""


def render_spine(m: dict[str, Any]) -> str:
    out = []
    stages = m["stages"]
    for i, st in enumerate(stages):
        agent = st["agent"]
        chips_in = "".join(render_chip("in", k, v) for k, v in st["inputs"].items())
        chips_out = "".join(render_chip("out", k, v) for k, v in st["outputs"].items())
        chips = ""
        if chips_in:
            chips += '<div class="chips"><span class="chip" style="border:none;background:none;color:var(--muted)">in ←</span>' + chips_in + "</div>"
        if chips_out:
            chips += '<div class="chips"><span class="chip" style="border:none;background:none;color:var(--muted)">out →</span>' + chips_out + "</div>"
        out.append(
            f'<div class="stage {agent}">'
            f'<div class="row"><span class="title">{esc(st["label"])}</span>'
            f'<span class="badge {agent}">{esc(agent.upper())}</span></div>'
            f"{render_metrics(st['phase'], st['metrics'])}"
            f"{chips}"
            f"</div>"
        )
        if i < len(stages) - 1:
            out.append('<div class="arrow">↓</div>')
    return "".join(out)


def pill(ok: bool, label: str) -> str:
    return f'<span class="pill {"ok" if ok else "bad"}">{esc(label)}</span>'


def render_query_table(m: dict[str, Any]) -> str:
    head = (
        "<tr><th>query</th><th>intent</th>"
        "<th>baseline<br>op / rec / judge</th>"
        "<th>refined<br>op / rec / judge</th>"
        "<th>T6 iters</th><th>T8 fan-out</th><th>T7 expr</th><th>feedback → fields</th></tr>"
    )
    rows = []
    for r in m["query_rows"]:
        b = r.get("baseline") or {}
        rf = r.get("refined") or {}

        def cell(a: dict[str, Any]) -> str:
            if not a:
                return '<span class="pill t">—</span>'
            j = pill(a.get("judge_success"), "judge " + ("ok" if a.get("judge_success") else "x"))
            conf = a.get("judge_confidence") or ""
            return (
                f'<span class="pill t">{esc(a.get("operation"))}</span> '
                f'<b>{esc(a.get("record_count"))}</b> rec {j} '
                f'<span class="sub">{esc(conf)}</span>'
            )

        iters = f'{(b.get("plan_iterations") or "-")} → {(rf.get("plan_iterations") or "-")}'
        fanb = (b.get("fanout_branches") or 0)
        fanr = (rf.get("fanout_branches") or 0)
        fan = f'{fanb} → {fanr}'
        expr = rf.get("aggregation_expression") or b.get("aggregation_expression")
        expr_html = f'<span class="mono tnote">{esc(expr)}</span>' if expr else '<span class="sub">—</span>'
        reqs = rf.get("column_requests") or b.get("column_requests") or []
        gaps = r.get("targets_schema_gaps") or []
        req_html = ""
        if reqs:
            req_html = ", ".join(esc(x) for x in reqs[:4]) + ("…" if len(reqs) > 4 else "")
        elif gaps:
            req_html = '<span class="sub">gap: ' + ", ".join(esc(x) for x in gaps[:3]) + "</span>"
        else:
            req_html = '<span class="sub">—</span>'
        rows.append(
            f'<tr><td class="q">{esc(r.get("query"))}'
            f'<div class="sub mono">{esc(r.get("source_type"))}</div></td>'
            f'<td><span class="pill t">{esc(r.get("intent"))}</span></td>'
            f"<td>{cell(b)}</td><td>{cell(rf)}</td>"
            f'<td class="mono">{esc(iters)}</td>'
            f'<td class="mono">{esc(fan)}</td>'
            f"<td>{expr_html}</td>"
            f"<td>{req_html}</td></tr>"
        )
    return f"<table>{head}{''.join(rows)}</table>"


def render_schema_panels(m: dict[str, Any]) -> str:
    base = m["baseline_cu"]
    ref = m["refined_cu"]
    base_fields = set(base["field_names"])
    added = m["schema_delta"].get("added_fields") or [
        f for f in ref["field_names"] if f not in base_fields
    ]
    added_set = set(added)
    field_html = "".join(
        f'<span class="f {"new" if f in added_set else ""}">{esc(f)}</span>'
        for f in ref["field_names"]
        if f
    )

    def dec_html(decisions: dict[str, int], values: list[str]) -> str:
        if not decisions:
            return '<span class="sub">—</span>'
        order = values or list(decisions.keys())
        parts = []
        for v in order:
            if v in decisions:
                parts.append(f'<span class="pill t">{esc(v)}: <b>{decisions[v]}</b></span>')
        return " ".join(parts)

    left = (
        '<div class="panel"><b>CU 側 スキーマ交渉</b>'
        f'<div class="sub" style="margin:6px 0">baseline {base["schema_field_count"]} fields '
        f'→ refined {ref["schema_field_count"]} fields '
        f'(<span class="delta-up">+{len(added)}</span>)</div>'
        f'<div style="margin:8px 0 4px"><span class="sub">field_candidates:</span> '
        f'<b>{ref["candidate_count"]}</b> · decisions: {dec_html(ref["negotiation_decisions"], ref["negotiation_values"])}</div>'
        f'<div class="fieldwrap">{field_html}</div>'
        '<div class="sub" style="margin-top:8px">緑 = このループで追加されたフィールド</div>'
        "</div>"
    )
    right = (
        '<div class="panel"><b>QU 側 候補評価（T9 simulation）</b>'
        f'<div class="sub" style="margin:6px 0">evaluated candidates: <b>{m["candidate_eval_total"]}</b></div>'
        f'<div>{dec_html(m["candidate_eval_decisions"], ["promote","defer","retrieval_only","merge_with_existing","needs_human_review"])}</div>'
        '<div class="sub" style="margin-top:10px">QU が filter / aggregate / search シミュレーションで各候補を検証し、'
        'promote 前に実 silver records に対して可否を判定している。</div>'
        "</div>"
    )
    return left + right


def render_signals(m: dict[str, Any]) -> str:
    rows = m["query_rows"]
    total_fanout = sum((r.get("refined") or {}).get("fanout_branches") or 0 for r in rows)
    total_fanout += sum((r.get("baseline") or {}).get("fanout_branches") or 0 for r in rows)
    exprs = [
        (r.get("refined") or {}).get("aggregation_expression")
        or (r.get("baseline") or {}).get("aggregation_expression")
        for r in rows
    ]
    exprs = [e for e in exprs if e]
    max_iters = max(
        [(r.get("refined") or {}).get("plan_iterations") or 0 for r in rows]
        + [(r.get("baseline") or {}).get("plan_iterations") or 0 for r in rows]
        + [0]
    )
    lm = m["loop_metrics"]
    items = [
        ("T1 usage/cost", f'{m["totals"]["calls"]} LLM calls · {m["totals"]["total_tokens"]:,} tokens roll-up across CU+QU stages'),
        ("T2 unified trace", "各 LLM 呼び出しが prompt_hash / latency / tokens / validation / fallback 付きの trace 行として answer.trace に記録"),
        ("T3 guardrails", "各 answer に guardrails(max_errors/budget/timeout) + best_partial_answer"),
        ("T5/T8 fan-out", f'retrieval_branch sub-call 計 <b>{total_fanout}</b> 件を並列実行し subagent_join で集約'),
        ("T6 replan loop", f'plan_iterations 最大 <b>{max_iters}</b>（観測駆動の再計画）'),
        ("T7 aggregate expr", (f'実行された集計式: ' + ", ".join(f'<span class="mono tnote">{esc(e)}</span>' for e in set(exprs))) if exprs else "このランでは planner は単純 group_by/filter を選択"),
        ("T9 negotiation", f'CU decisions {m["refined_cu"]["negotiation_decisions"]} · QU eval {m["candidate_eval_decisions"]}'),
        ("loop delta", f'judge_success_delta: <b>{lm.get("judge_success_delta")}</b> · feedback_promoted: <b>{lm.get("feedback_promoted")}</b> · refined filter を使うクエリ: <b>{lm.get("query_uses_refined_filter_count")}</b>'),
    ]
    return "<table>" + "".join(
        f'<tr><td style="white-space:nowrap"><span class="pill warn">{esc(k)}</span></td><td>{v}</td></tr>'
        for k, v in items
    ) + "</table>"


def render_html(m: dict[str, Any]) -> str:
    pricing = m.get("pricing_source")
    pricing_note = (
        "※ cost は pricing 未設定（source=%s）のため $0 表示。tokens/calls は実測値。"
        % esc(pricing)
        if pricing in (None, "unpriced")
        else "pricing source: %s" % esc(pricing)
    )
    html = HTML_TEMPLATE
    replacements = {
        "__LOOP_ID__": esc(m.get("loop_id")),
        "__MODEL__": esc(m.get("model")),
        "__RUN_DIR__": esc(m.get("run_dir")),
        "__TOTALS__": render_totals(m),
        "__PRICING_NOTE__": pricing_note,
        "__SPINE__": render_spine(m),
        "__QUERY_TABLE__": render_query_table(m),
        "__SCHEMA_PANELS__": render_schema_panels(m),
        "__SIGNALS__": render_signals(m),
    }
    for k, v in replacements.items():
        html = html.replace(k, v)
    return html


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize a completed CU<->QU loop run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory produced by qu_cu_loop_demo.py")
    parser.add_argument("--output", type=Path, default=None, help="HTML output path (default: <run-dir>/loop_dataflow.html)")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not (run_dir / "orchestration_trace.jsonl").exists():
        raise SystemExit(f"orchestration_trace.jsonl not found under {run_dir}; is this a loop run dir?")
    model = build_model(run_dir)
    output = args.output or (run_dir / "loop_dataflow.html")
    output.write_text(render_html(model), encoding="utf-8")
    t = model["totals"]
    print(f"Wrote {output}")
    print(
        f"  loop_id={model['loop_id']} model={model['model']} "
        f"queries={t['queries']} fields={model['baseline_cu']['schema_field_count']}->{model['refined_cu']['schema_field_count']} "
        f"llm_calls={t['calls']} tokens={t['total_tokens']:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
