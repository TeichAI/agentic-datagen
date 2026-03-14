from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .formatter import Formatter

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    box = None
    Console = None
    Panel = None
    Table = None
    RICH_AVAILABLE = False


def _safe_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _preview(value: str, limit: int = 120) -> str:
    normalized = " ".join(value.split())
    return normalized[:limit]


def _percent(count: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{(count / total) * 100:.1f}%"


def _parse_tool_content(content: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def analyze_entry(entry: Dict[str, Any], index: int, formatter: Optional[Formatter] = None) -> Dict[str, Any]:
    formatter = formatter or Formatter()
    metadata = entry.get("metadata") or {}
    metadata = metadata if isinstance(metadata, dict) else {}
    usage = entry.get("usage") or {}
    usage = usage if isinstance(usage, dict) else {}
    messages = entry.get("messages") or []
    messages = messages if isinstance(messages, list) else []
    roles = [msg.get("role") for msg in messages if isinstance(msg, dict)]
    assistant_messages = [msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "assistant"]
    tool_messages = [msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "tool"]
    user_messages = [msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "user"]
    system_messages = [msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "system"]
    final_message = messages[-1] if messages and isinstance(messages[-1], dict) else {}
    final_role = final_message.get("role")
    final_content = _safe_text(final_message.get("content"))
    prompt_text = _safe_text(entry.get("prompt"))
    all_text = "\n".join(_safe_text(msg.get("content")) for msg in messages if isinstance(msg, dict))
    all_text_lower = all_text.lower()

    errors: List[str] = []
    warnings: List[str] = []
    info: List[str] = []

    ended_with_plain_assistant = formatter.final_assistant_is_plain(messages)
    structural_valid = formatter.validate_entry(entry)
    training_safe = formatter.is_training_safe_entry(entry)
    suspiciously_shallow_completion = formatter.is_suspiciously_shallow_completion(entry)
    ended_with_assistant = final_role == "assistant"
    metadata_error_present = bool(metadata.get("error"))
    completed = metadata.get("completed") is True

    if not structural_valid:
        errors.append("invalid_structure")
    if metadata_error_present:
        errors.append("metadata_error")
    if not ended_with_assistant:
        errors.append("last_message_not_assistant")
    if not completed:
        errors.append("metadata_not_completed")
    if not prompt_text.strip():
        errors.append("missing_prompt")
    if not user_messages:
        errors.append("missing_user_message")
    if not assistant_messages:
        errors.append("missing_assistant_message")
    if not messages:
        errors.append("missing_messages")
    if ended_with_assistant and not final_content.strip():
        errors.append("empty_final_assistant_content")
    if ended_with_assistant and "tool_calls" in final_message:
        errors.append("final_assistant_has_tool_calls")

    failed_tool_calls = 0
    unparsable_tool_messages = 0
    for msg in tool_messages:
        parsed = _parse_tool_content(msg.get("content"))
        if parsed is None:
            unparsable_tool_messages += 1
            continue
        if parsed.get("success") is False or parsed.get("isError") is True:
            failed_tool_calls += 1

    if failed_tool_calls:
        warnings.append("failed_tool_calls_present")
    if unparsable_tool_messages:
        warnings.append("unparsable_tool_message_content")
    if suspiciously_shallow_completion:
        warnings.append("suspiciously_shallow_completion")

    tool_calls_count = metadata.get("tool_calls_count")
    if isinstance(tool_calls_count, int) and tool_calls_count != len(tool_messages):
        warnings.append("tool_call_count_mismatch")

    has_reasoning_tags = "<think>" in all_text or "</think>" in all_text
    mentions_localhost = "localhost" in all_text_lower or "127.0.0.1" in all_text_lower
    mentions_port_conflict = "already in use" in all_text_lower

    if mentions_localhost:
        info.append("mentions_localhost")
    if mentions_port_conflict:
        warnings.append("mentions_port_conflict")

    quality_score = 100
    quality_score -= len(errors) * 20
    quality_score -= failed_tool_calls * 5
    quality_score -= unparsable_tool_messages * 3
    if mentions_port_conflict:
        quality_score -= 3
    if not ended_with_assistant:
        quality_score -= 10
    if not training_safe:
        quality_score -= 10
    quality_score = max(0, quality_score)

    return {
        "index": index,
        "prompt_id": metadata.get("prompt_id"),
        "run_id": metadata.get("run_id"),
        "prompt_preview": _preview(prompt_text),
        "final_preview": _preview(final_content),
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "cost": float(usage.get("cost") or 0.0),
        },
        "errors": errors,
        "warnings": warnings,
        "info": info,
        "quality": {
            "score": quality_score,
            "structural_valid": structural_valid,
            "training_safe": training_safe,
            "suspiciously_shallow_completion": suspiciously_shallow_completion,
            "ended_with_assistant": ended_with_assistant,
            "ended_with_plain_assistant": ended_with_plain_assistant,
            "metadata_error_present": metadata_error_present,
            "failed_tool_calls": failed_tool_calls,
            "unparsable_tool_messages": unparsable_tool_messages,
            "has_reasoning_tags": has_reasoning_tags,
            "mentions_localhost": mentions_localhost,
            "mentions_port_conflict": mentions_port_conflict,
            "turns": int(metadata.get("turns") or 0),
            "message_count": len(messages),
            "system_message_count": len(system_messages),
            "user_message_count": len(user_messages),
            "assistant_message_count": len(assistant_messages),
            "tool_message_count": len(tool_messages),
            "tool_calls_count": tool_calls_count,
        },
    }


def load_reports(dataset_path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    formatter = Formatter()
    reports: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if limit is not None and len(reports) >= limit:
                break
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                reports.append(
                    {
                        "index": line_number,
                        "prompt_id": None,
                        "run_id": None,
                        "prompt_preview": "",
                        "final_preview": "",
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "cost": 0.0,
                        },
                        "errors": ["invalid_json"],
                        "warnings": [],
                        "info": [],
                        "quality": {
                            "score": 0,
                            "structural_valid": False,
                            "training_safe": False,
                            "suspiciously_shallow_completion": False,
                            "ended_with_assistant": False,
                            "ended_with_plain_assistant": False,
                            "metadata_error_present": False,
                            "failed_tool_calls": 0,
                            "unparsable_tool_messages": 0,
                            "has_reasoning_tags": False,
                            "mentions_localhost": False,
                            "mentions_port_conflict": False,
                            "turns": 0,
                            "message_count": 0,
                            "system_message_count": 0,
                            "user_message_count": 0,
                            "assistant_message_count": 0,
                            "tool_message_count": 0,
                            "tool_calls_count": None,
                        },
                    }
                )
                continue
            reports.append(analyze_entry(entry, line_number, formatter=formatter))
    return reports


def summarize_reports(reports: Iterable[Dict[str, Any]], dataset_path: Optional[Path] = None) -> Dict[str, Any]:
    report_list = list(reports)
    error_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    info_counts: Counter[str] = Counter()
    total_failed_tool_calls = 0
    structural_valid = 0
    training_safe = 0
    ended_with_assistant = 0
    ended_with_plain_assistant = 0
    metadata_error_entries = 0
    reasoning_entries = 0
    localhost_entries = 0
    port_conflict_entries = 0
    shallow_completion_entries = 0
    total_score = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_cost = 0.0
    total_turns = 0
    total_tool_calls = 0

    for report in report_list:
        error_counts.update(report.get("errors") or [])
        warning_counts.update(report.get("warnings") or [])
        info_counts.update(report.get("info") or [])
        usage = report.get("usage") or {}
        quality = report.get("quality") or {}
        total_prompt_tokens += int(usage.get("prompt_tokens") or 0)
        total_completion_tokens += int(usage.get("completion_tokens") or 0)
        total_tokens += int(usage.get("total_tokens") or 0)
        total_cost += float(usage.get("cost") or 0.0)
        total_failed_tool_calls += int(quality.get("failed_tool_calls") or 0)
        total_score += int(quality.get("score") or 0)
        total_turns += int(quality.get("turns") or 0)
        total_tool_calls += int(quality.get("tool_calls_count") or 0)
        if quality.get("structural_valid"):
            structural_valid += 1
        if quality.get("training_safe"):
            training_safe += 1
        if quality.get("ended_with_assistant"):
            ended_with_assistant += 1
        if quality.get("ended_with_plain_assistant"):
            ended_with_plain_assistant += 1
        if quality.get("metadata_error_present"):
            metadata_error_entries += 1
        if quality.get("has_reasoning_tags"):
            reasoning_entries += 1
        if quality.get("mentions_localhost"):
            localhost_entries += 1
        if quality.get("mentions_port_conflict"):
            port_conflict_entries += 1
        if quality.get("suspiciously_shallow_completion"):
            shallow_completion_entries += 1

    flagged_entries = [r for r in report_list if r.get("errors") or r.get("warnings")]
    flagged_entries.sort(
        key=lambda report: (
            -len(report.get("errors") or []),
            -len(report.get("warnings") or []),
            report.get("quality", {}).get("score", 0),
            report.get("index", 0),
        )
    )

    total_entries = len(report_list)
    average_score = (total_score / total_entries) if total_entries else 0.0
    average_turns = (total_turns / total_entries) if total_entries else 0.0
    average_tool_calls = (total_tool_calls / total_entries) if total_entries else 0.0
    average_tokens = (total_tokens / total_entries) if total_entries else 0.0

    return {
        "dataset_path": str(dataset_path) if dataset_path else None,
        "totals": {
            "entries": total_entries,
            "final_rows": total_entries,
            "structural_valid": structural_valid,
            "training_safe": training_safe,
            "ended_with_assistant": ended_with_assistant,
            "ended_with_plain_assistant": ended_with_plain_assistant,
            "metadata_error_entries": metadata_error_entries,
            "entries_with_reasoning_tags": reasoning_entries,
            "entries_mentioning_localhost": localhost_entries,
            "entries_mentioning_port_conflict": port_conflict_entries,
            "entries_with_suspiciously_shallow_completion": shallow_completion_entries,
            "entries_with_failed_tool_calls": sum(
                1 for report in report_list if (report.get("quality") or {}).get("failed_tool_calls")
            ),
            "total_failed_tool_calls": total_failed_tool_calls,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 6),
            "average_turns": round(average_turns, 2),
            "average_tool_calls": round(average_tool_calls, 2),
            "average_tokens_per_row": round(average_tokens, 2),
            "average_quality_score": round(average_score, 2),
        },
        "issue_counts": {
            "errors": dict(error_counts),
            "warnings": dict(warning_counts),
            "info": dict(info_counts),
        },
        "flagged_entries": flagged_entries,
    }


def render_text_report(summary: Dict[str, Any], max_flagged: int = 20) -> str:
    totals = summary.get("totals") or {}
    issue_counts = summary.get("issue_counts") or {}
    lines = [
        "Dataset QA Summary",
        f"Dataset: {summary.get('dataset_path')}",
        f"Final rows: {totals.get('final_rows', 0)}",
        f"Entries analyzed: {totals.get('entries', 0)}",
        f"Structural valid: {totals.get('structural_valid', 0)}",
        f"Training safe: {totals.get('training_safe', 0)}",
        f"Ended with assistant: {totals.get('ended_with_assistant', 0)}",
        f"Ended with plain assistant: {totals.get('ended_with_plain_assistant', 0)}",
        f"Metadata error entries: {totals.get('metadata_error_entries', 0)}",
        f"Entries with suspiciously shallow completion: {totals.get('entries_with_suspiciously_shallow_completion', 0)}",
        f"Entries with failed tool calls: {totals.get('entries_with_failed_tool_calls', 0)}",
        f"Total failed tool calls: {totals.get('total_failed_tool_calls', 0)}",
        f"Total prompt tokens: {totals.get('total_prompt_tokens', 0)}",
        f"Total completion tokens: {totals.get('total_completion_tokens', 0)}",
        f"Total tokens: {totals.get('total_tokens', 0)}",
        f"Total cost: {totals.get('total_cost', 0.0)}",
        f"Average turns: {totals.get('average_turns', 0.0)}",
        f"Average tool calls: {totals.get('average_tool_calls', 0.0)}",
        f"Average tokens per row: {totals.get('average_tokens_per_row', 0.0)}",
        f"Entries with reasoning tags: {totals.get('entries_with_reasoning_tags', 0)}",
        f"Entries mentioning localhost: {totals.get('entries_mentioning_localhost', 0)}",
        f"Entries mentioning port conflict: {totals.get('entries_mentioning_port_conflict', 0)}",
        f"Average quality score: {totals.get('average_quality_score', 0.0)}",
        "",
        "Error counts:",
    ]

    error_counts = issue_counts.get("errors") or {}
    if error_counts:
        for name, count in sorted(error_counts.items()):
            lines.append(f"  - {name}: {count}")
    else:
        lines.append("  - none")

    lines.append("")
    lines.append("Warning counts:")
    warning_counts = issue_counts.get("warnings") or {}
    if warning_counts:
        for name, count in sorted(warning_counts.items()):
            lines.append(f"  - {name}: {count}")
    else:
        lines.append("  - none")

    lines.append("")
    lines.append("Info counts:")
    info_counts = issue_counts.get("info") or {}
    if info_counts:
        for name, count in sorted(info_counts.items()):
            lines.append(f"  - {name}: {count}")
    else:
        lines.append("  - none")

    flagged_entries = summary.get("flagged_entries") or []
    lines.append("")
    lines.append(f"Flagged entries (showing up to {max_flagged}):")
    if not flagged_entries:
        lines.append("  - none")
        return "\n".join(lines)

    for report in flagged_entries[:max_flagged]:
        quality = report.get("quality") or {}
        lines.append(
            f"  - #{report.get('index')} prompt_id={report.get('prompt_id')} score={quality.get('score')}"
        )
        lines.append(f"    errors={report.get('errors') or []}")
        lines.append(f"    warnings={report.get('warnings') or []}")
        lines.append(
            f"    failed_tool_calls={quality.get('failed_tool_calls', 0)} tool_messages={quality.get('tool_message_count', 0)}"
        )
        lines.append(f"    prompt={report.get('prompt_preview')}")
        final_preview = report.get("final_preview")
        if final_preview:
            lines.append(f"    final={final_preview}")

    return "\n".join(lines)


def render_rich_report(
    summary: Dict[str, Any], max_flagged: int = 20, console: Optional[Console] = None
) -> None:
    if not RICH_AVAILABLE:
        print(render_text_report(summary, max_flagged=max_flagged))
        return

    console = console or Console()
    totals = summary.get("totals") or {}
    issue_counts = summary.get("issue_counts") or {}
    total_entries = int(totals.get("entries", 0) or 0)

    console.print()
    console.print(
        Panel.fit(
            f"[bold]Dataset QA[/bold]\n[dim]{summary.get('dataset_path')}[/dim]",
            border_style="cyan",
        )
    )

    metrics = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
    metrics.add_column("Metric", style="bold")
    metrics.add_column("Count", justify="right")
    metrics.add_column("Rate", justify="right")
    metrics.add_row("Entries analyzed", str(total_entries), "100.0%" if total_entries else "0.0%")
    metrics.add_row("Final rows", str(totals.get("final_rows", 0)), "100.0%" if total_entries else "0.0%")
    metrics.add_row(
        "Training safe",
        str(totals.get("training_safe", 0)),
        _percent(int(totals.get("training_safe", 0) or 0), total_entries),
    )
    metrics.add_row(
        "Ended with plain assistant",
        str(totals.get("ended_with_plain_assistant", 0)),
        _percent(int(totals.get("ended_with_plain_assistant", 0) or 0), total_entries),
    )
    metrics.add_row(
        "Metadata error entries",
        str(totals.get("metadata_error_entries", 0)),
        _percent(int(totals.get("metadata_error_entries", 0) or 0), total_entries),
    )
    metrics.add_row(
        "Localhost mentions (info)",
        str(totals.get("entries_mentioning_localhost", 0)),
        _percent(int(totals.get("entries_mentioning_localhost", 0) or 0), total_entries),
    )
    metrics.add_row("Total prompt tokens", f"{int(totals.get('total_prompt_tokens', 0) or 0):,}", "")
    metrics.add_row("Total completion tokens", f"{int(totals.get('total_completion_tokens', 0) or 0):,}", "")
    metrics.add_row("Total tokens", f"{int(totals.get('total_tokens', 0) or 0):,}", "")
    metrics.add_row("Total cost", f"${float(totals.get('total_cost', 0.0) or 0.0):.6f}", "")
    metrics.add_row("Average turns", str(totals.get("average_turns", 0.0)), "")
    metrics.add_row("Average tool calls", str(totals.get("average_tool_calls", 0.0)), "")
    metrics.add_row("Average tokens per row", str(totals.get("average_tokens_per_row", 0.0)), "")
    metrics.add_row("Average quality score", str(totals.get("average_quality_score", 0.0)), "")
    console.print(metrics)

    def _issue_panel(title: str, counts: Dict[str, Any], color: str) -> None:
        table = Table(box=box.MINIMAL_DOUBLE_HEAD, header_style=f"bold {color}")
        table.add_column(title[:-1] if title.endswith("s") else title, style=color)
        table.add_column("Count", justify="right")
        if counts:
            for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
                table.add_row(name, str(count))
        else:
            table.add_row("none", "0")
        console.print(Panel(table, title=title, border_style=color))

    _issue_panel("Errors", issue_counts.get("errors") or {}, "red")
    _issue_panel("Warnings", issue_counts.get("warnings") or {}, "yellow")
    _issue_panel("Info", issue_counts.get("info") or {}, "blue")

    flagged_entries = summary.get("flagged_entries") or []
    console.rule(f"Flagged entries (showing up to {max_flagged})", style="cyan")
    if not flagged_entries:
        console.print("[green]No flagged entries.[/green]")
        return

    flagged = Table(box=box.SIMPLE_HEAVY, header_style="bold magenta")
    flagged.add_column("#", justify="right", style="bold")
    flagged.add_column("Score", justify="right")
    flagged.add_column("Prompt ID", style="cyan")
    flagged.add_column("Issues", style="yellow")
    flagged.add_column("Failed tools", justify="right")
    flagged.add_column("Prompt preview", overflow="fold")
    flagged.add_column("Final preview", overflow="fold")

    for report in flagged_entries[:max_flagged]:
        quality = report.get("quality") or {}
        issues = []
        issues.extend(report.get("errors") or [])
        issues.extend(report.get("warnings") or [])
        flagged.add_row(
            str(report.get("index")),
            str(quality.get("score", 0)),
            str(report.get("prompt_id") or "-"),
            "\n".join(issues) if issues else "-",
            str(quality.get("failed_tool_calls", 0)),
            str(report.get("prompt_preview") or ""),
            str(report.get("final_preview") or ""),
        )
    console.print(flagged)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--max-flagged", type=int, default=20)
    parser.add_argument("--fail-on-errors", action="store_true")
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset_path)
    reports = load_reports(dataset_path, limit=args.limit)
    summary = summarize_reports(reports, dataset_path=dataset_path)

    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif not args.plain and RICH_AVAILABLE:
        render_rich_report(summary, max_flagged=args.max_flagged)
    else:
        print(render_text_report(summary, max_flagged=args.max_flagged))

    if args.fail_on_errors and (summary.get("issue_counts", {}).get("errors") or {}):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
