from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import time
from typing import Any, Iterable

try:
    from rich import box
    from rich.console import Console, Group
    from rich.markup import escape
    from rich.panel import Panel
    from rich.table import Table
except ImportError:  # pragma: no cover - exercised when rich is not installed locally.
    box = None
    Console = None
    Group = None
    Panel = None
    Table = None
    escape = None


class DemoTUI:
    def __init__(self, *, title: str, enabled: bool = True) -> None:
        self.title = title
        self.enabled = enabled
        self.rich = enabled and Console is not None
        self.console = Console() if self.rich else None
        self.started_at = time.perf_counter()

    def header(self, *, subtitle: str, config: dict[str, Any]) -> None:
        if not self.enabled:
            return
        if self.rich:
            table = Table.grid(padding=(0, 2))
            table.add_column(style="bold cyan", no_wrap=True)
            table.add_column(style="white")
            for key, value in config.items():
                table.add_row(str(key), format_value(value))
            self.console.print(
                Panel(
                    Group(str(subtitle), table),
                    title=f"[bold]{self.title}[/bold]",
                    border_style="cyan",
                    box=box.ROUNDED,
                )
            )
            return
        print(f"\n== {self.title} ==")
        print(subtitle)
        for key, value in config.items():
            print(f"  {key}: {format_value(value)}")

    @contextmanager
    def phase(self, title: str, detail: str = ""):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        if self.rich:
            status_text = f"[bold cyan]{escape_text(title)}[/bold cyan]"
            if detail:
                status_text += f"\n[dim]{escape_text(detail)}[/dim]"
            with self.console.status(status_text, spinner="dots"):
                yield
            self.console.print(f"[green]done[/green] {escape_text(title)} [dim]{elapsed(start)}[/dim]")
            return
        print(f"\n[start] {title}")
        if detail:
            print(f"        {detail}")
        try:
            yield
        finally:
            print(f"[done]  {title} ({elapsed(start)})")

    def message(self, text: str, *, style: str = "dim") -> None:
        if not self.enabled:
            return
        if self.rich:
            self.console.print(text, style=style)
            return
        print(text)

    def show_cu_artifact(self, label: str, artifact: Any, output_dir: Path) -> None:
        if not self.enabled:
            return
        manifest = artifact.manifest
        prompt_state = manifest.get("prompt_state", {})
        feedback = artifact.feedback_report.get("feedback", {})
        metrics = {
            "output": output_dir,
            "records": manifest.get("record_count"),
            "chunks": manifest.get("chunk_count"),
            "schema_fields": len(artifact.silver_schema_catalog.get("fields", [])),
            "inducer": manifest.get("schema_induction", {}).get("inducer"),
            "schema_prompt": prompt_hash(prompt_state.get("schema_inducer")),
            "extractor_prompt": prompt_hash(prompt_state.get("field_extractor")),
            "feedback_requests": feedback.get("request_count", 0),
            "validation_errors": sum(artifact.feedback_report.get("validation_error_counts", {}).values()),
        }
        self.kv_panel(label, metrics, border_style="blue")

    def show_query_tasks(self, tasks: list[dict[str, Any]], report: Any, path: Path) -> None:
        if not self.enabled:
            return
        self.kv_panel(
            "Downstream Query Distribution",
            {
                "tasks": len(tasks),
                "source_counts": json.dumps(source_counts(tasks), ensure_ascii=False),
                "generator": getattr(report, "generator", ""),
                "model": getattr(report, "model", ""),
                "generation_id": getattr(report, "generation_id", ""),
                "artifact": path,
            },
            border_style="magenta",
        )
        rows = [
            [
                str(task.get("task_id", "")),
                str(task.get("source_type", "")),
                str(task.get("intent", "")),
                truncate(str(task.get("query", "")), 88),
            ]
            for task in tasks[:8]
        ]
        if rows:
            self.table("Query Tasks", ["task_id", "source", "intent", "query"], rows)
        if len(tasks) > 8:
            self.message(f"... {len(tasks) - 8} more query tasks written to {path}")

    def show_answers(self, label: str, answers: list[dict[str, Any]], output_dir: Path) -> None:
        if not self.enabled:
            return
        rows = []
        for answer in answers:
            task = answer.get("query_task") if isinstance(answer.get("query_task"), dict) else {}
            judgement = answer.get("judgement") if isinstance(answer.get("judgement"), dict) else {}
            diagnostics = answer.get("search_diagnostics") if isinstance(answer.get("search_diagnostics"), dict) else {}
            rows.append(
                [
                    str(task.get("task_id", "")),
                    str(answer.get("plan", {}).get("operation", "")),
                    str(len(answer.get("records", []))),
                    str(len(answer.get("evidence", []))),
                    str(len(diagnostics.get("calls", []))),
                    truth(judgement.get("success")),
                    truth(judgement.get("needs_cu_feedback")),
                    ", ".join(requested_field_names(answer)) or "-",
                ]
            )
        self.kv_panel(
            label,
            {
                "answers": len(answers),
                "artifact_dir": output_dir,
                "success": sum(1 for item in answers if item.get("judgement", {}).get("success")),
                "needs_feedback": sum(1 for item in answers if item.get("judgement", {}).get("needs_cu_feedback")),
                "column_requests": sum(len(item.get("column_requests", [])) for item in answers),
            },
            border_style="green",
        )
        if rows:
            self.table(
                f"{label} Results",
                ["task_id", "op", "records", "evidence", "search", "judge", "feedback", "requested_fields"],
                rows,
            )

    def show_feedback(self, feedback_path: Path, answers: list[dict[str, Any]]) -> None:
        if not self.enabled:
            return
        self.kv_panel(
            "QU to CU Feedback",
            {
                "artifact": feedback_path,
                "records": jsonl_count(feedback_path),
                "requested_fields": ", ".join(sorted(set(field for answer in answers for field in requested_field_names(answer)))) or "-",
            },
            border_style="yellow",
        )

    def show_loop_summary(self, summary: dict[str, Any]) -> None:
        if not self.enabled:
            return
        loop = summary.get("loop_result", {})
        schema = summary.get("schema_delta", {})
        self.kv_panel(
            "Loop Result",
            {
                "tasks": loop.get("task_count"),
                "feedback_records": loop.get("feedback_record_count"),
                "feedback_promoted": truth(loop.get("feedback_promoted")),
                "judge_success_delta": loop.get("judge_success_delta"),
                "column_request_delta": loop.get("column_request_count_delta"),
                "baseline_fields": schema.get("baseline_field_count"),
                "refined_fields": schema.get("refined_field_count"),
                "added_fields": ", ".join(schema.get("added_fields", [])[:10]) or "-",
            },
            border_style="cyan",
        )
        artifacts = summary.get("artifacts", {})
        self.kv_panel("Artifacts", artifacts, border_style="dim")

    def show_user_context(self, context: dict[str, Any]) -> None:
        if not self.enabled:
            return
        config = context.get("agent_config", {})
        self.kv_panel(
            "User Query Agent",
            {
                "corpus": context.get("corpus_dir"),
                "stage": context.get("stage"),
                "schema_fields": len(context.get("schema_field_names", [])),
                "records": context.get("record_count"),
                "chunks": context.get("chunk_count"),
                "planner": config.get("planner"),
                "retrieval_mode": config.get("retrieval_mode"),
                "retrieval_subagent": config.get("retrieval_subagent"),
                "reranker": config.get("reranker"),
                "judge": config.get("judge"),
                "model": config.get("llm_model"),
            },
            border_style="blue",
        )

    def show_user_queries(self, tasks: list[dict[str, Any]], path: Path) -> None:
        if not self.enabled:
            return
        rows = [
            [str(task.get("task_id", "")), truncate(str(task.get("query", "")), 96)]
            for task in tasks[:10]
        ]
        self.kv_panel("User Queries", {"queries": len(tasks), "artifact": path}, border_style="magenta")
        if rows:
            self.table("Queries", ["task_id", "query"], rows)

    def show_user_summary(self, summary: dict[str, Any]) -> None:
        if not self.enabled:
            return
        metrics = summary.get("metrics", {})
        self.kv_panel(
            "User Query Result",
            {
                "queries": metrics.get("query_count"),
                "success": metrics.get("success_count"),
                "needs_feedback": metrics.get("needs_cu_feedback_count"),
                "column_requests": metrics.get("column_request_count"),
                "output": summary.get("artifacts", {}).get("output_dir"),
            },
            border_style="cyan",
        )
        rows = []
        for item in summary.get("results", []):
            rows.append(
                [
                    str(item.get("task_id", "")),
                    str(item.get("operation", "")),
                    str(len(item.get("record_ids", []))),
                    str(item.get("evidence_count", 0)),
                    truth(item.get("judge_success")),
                    truth(item.get("needs_cu_feedback")),
                    ", ".join(item.get("column_requests", [])) or "-",
                    truncate(str(item.get("query", "")), 74),
                ]
            )
        if rows:
            self.table(
                "Answers",
                ["task_id", "op", "records", "evidence", "judge", "feedback", "requested_fields", "query"],
                rows,
            )

    def kv_panel(self, title: str, values: dict[str, Any], *, border_style: str = "cyan") -> None:
        if not self.enabled:
            return
        if self.rich:
            table = Table.grid(padding=(0, 2))
            table.add_column(style="bold", no_wrap=True)
            table.add_column()
            for key, value in values.items():
                table.add_row(str(key), escape_text(format_value(value)))
            self.console.print(Panel(table, title=escape_text(title), border_style=border_style, box=box.ROUNDED))
            return
        print(f"\n-- {title} --")
        for key, value in values.items():
            print(f"  {key}: {format_value(value)}")

    def table(self, title: str, columns: list[str], rows: list[list[str]]) -> None:
        if not self.enabled:
            return
        if self.rich:
            table = Table(title=escape_text(title), box=box.SIMPLE_HEAVY, show_lines=False)
            for column in columns:
                table.add_column(column, overflow="fold")
            for row in rows:
                table.add_row(*[escape_text(value) for value in row])
            self.console.print(table)
            return
        print(f"\n-- {title} --")
        widths = [len(column) for column in columns]
        for row in rows:
            for index, value in enumerate(row):
                widths[index] = min(max(widths[index], len(value)), 80)
        print("  ".join(column.ljust(widths[index]) for index, column in enumerate(columns)))
        print("  ".join("-" * width for width in widths))
        for row in rows:
            print("  ".join(truncate(value, widths[index]).ljust(widths[index]) for index, value in enumerate(row)))

    def finish(self) -> None:
        if not self.enabled:
            return
        self.message(f"Completed in {elapsed(self.started_at)}", style="bold green")


def requested_field_names(answer: dict[str, Any]) -> list[str]:
    return [str(request.get("field_name")) for request in answer.get("column_requests", []) if request.get("field_name")]


def source_counts(tasks: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        source = str(task.get("source_type") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def prompt_hash(value: Any) -> str:
    if not isinstance(value, dict):
        return "-"
    prompt_id = value.get("prompt_id")
    digest = value.get("prompt_hash")
    if prompt_id and digest:
        return f"{prompt_id}@{digest}"
    return "-"


def format_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return "-"
    return str(value)


def truth(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "-"


def elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


def truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3] + "..."


def escape_text(value: str) -> str:
    if escape is None:
        return value
    return escape(value)
