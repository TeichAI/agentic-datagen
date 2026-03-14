from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .dataset_qa import RICH_AVAILABLE, Console, Panel, Table, box, load_reports

try:
    from rich.prompt import Confirm, IntPrompt
except ImportError:
    Confirm = None
    IntPrompt = None


@dataclass
class CleanupPolicy:
    remove_errors: bool = True
    remove_shallow: bool = True
    remove_port_conflicts: bool = True
    max_failed_tool_calls: Optional[int] = None
    min_quality_score: Optional[int] = None


def _policy_enabled(policy: CleanupPolicy) -> bool:
    return any(
        [
            policy.remove_errors,
            policy.remove_shallow,
            policy.remove_port_conflicts,
            policy.max_failed_tool_calls is not None,
            policy.min_quality_score is not None,
        ]
    )


def _load_nonempty_lines(dataset_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            rows.append({"index": line_number, "raw": stripped})
    return rows


def get_removal_reasons(report: Dict[str, Any], policy: CleanupPolicy) -> List[str]:
    reasons: List[str] = []
    quality = report.get("quality") or {}
    errors = report.get("errors") or []
    warnings = report.get("warnings") or []
    failed_tool_calls = int(quality.get("failed_tool_calls") or 0)
    score = int(quality.get("score") or 0)

    if policy.remove_errors and errors:
        reasons.extend(str(error) for error in errors)
    if policy.remove_shallow and quality.get("suspiciously_shallow_completion"):
        reasons.append("suspiciously_shallow_completion")
    if policy.remove_port_conflicts and quality.get("mentions_port_conflict"):
        reasons.append("mentions_port_conflict")
    if policy.max_failed_tool_calls is not None and failed_tool_calls > policy.max_failed_tool_calls:
        reasons.append(f"failed_tool_calls_gt_{policy.max_failed_tool_calls}")
    if policy.min_quality_score is not None and score < policy.min_quality_score:
        reasons.append(f"quality_below_{policy.min_quality_score}")
    if not reasons and "failed_tool_calls_present" in warnings and policy.max_failed_tool_calls == 0:
        reasons.append("failed_tool_calls_gt_0")

    deduped: List[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return deduped


def plan_cleanup(rows: List[Dict[str, Any]], reports: List[Dict[str, Any]], policy: CleanupPolicy) -> Dict[str, Any]:
    report_by_index = {int(report.get("index", -1)): report for report in reports}
    retained_rows: List[Dict[str, Any]] = []
    removed_rows: List[Dict[str, Any]] = []
    removal_reason_counts: Dict[str, int] = {}

    for row in rows:
        report = report_by_index.get(int(row["index"]))
        if report is None:
            retained_rows.append({"row": row, "report": None, "reasons": []})
            continue
        reasons = get_removal_reasons(report, policy)
        if reasons:
            removed_rows.append({"row": row, "report": report, "reasons": reasons})
            for reason in reasons:
                removal_reason_counts[reason] = removal_reason_counts.get(reason, 0) + 1
        else:
            retained_rows.append({"row": row, "report": report, "reasons": []})

    return {
        "policy": policy,
        "rows": rows,
        "reports": reports,
        "retained_rows": retained_rows,
        "removed_rows": removed_rows,
        "removal_reason_counts": removal_reason_counts,
        "total_rows": len(rows),
        "removed_count": len(removed_rows),
        "retained_count": len(retained_rows),
    }


def _default_backup_path(dataset_path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return dataset_path.with_name(f"{dataset_path.name}.{timestamp}.bak")


def apply_cleanup(dataset_path: Path, plan: Dict[str, Any], backup_path: Optional[Path] = None) -> Path:
    backup_target = backup_path or _default_backup_path(dataset_path)
    shutil.copy2(dataset_path, backup_target)
    retained_lines = [item["row"]["raw"] for item in plan.get("retained_rows") or []]
    with dataset_path.open("w", encoding="utf-8") as handle:
        if retained_lines:
            handle.write("\n".join(retained_lines) + "\n")
    return backup_target


def _render_policy_summary(policy: CleanupPolicy) -> List[str]:
    lines = [
        f"remove_errors={policy.remove_errors}",
        f"remove_shallow={policy.remove_shallow}",
        f"remove_port_conflicts={policy.remove_port_conflicts}",
        f"max_failed_tool_calls={policy.max_failed_tool_calls}",
        f"min_quality_score={policy.min_quality_score}",
    ]
    return lines


def render_plan_text(dataset_path: Path, plan: Dict[str, Any], max_examples: int = 10) -> str:
    lines = [
        "Dataset Cleanup Plan",
        f"Dataset: {dataset_path}",
        *[f"Policy: {line}" for line in _render_policy_summary(plan["policy"])],
        f"Rows scanned: {plan['total_rows']}",
        f"Rows to remove: {plan['removed_count']}",
        f"Rows to retain: {plan['retained_count']}",
        "",
        "Removal reasons:",
    ]
    reason_counts = plan.get("removal_reason_counts") or {}
    if reason_counts:
        for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"  - {reason}: {count}")
    else:
        lines.append("  - none")

    lines.append("")
    lines.append(f"Examples (up to {max_examples}):")
    examples = plan.get("removed_rows") or []
    if not examples:
        lines.append("  - none")
        return "\n".join(lines)

    for item in examples[:max_examples]:
        report = item.get("report") or {}
        quality = report.get("quality") or {}
        lines.append(
            f"  - #{report.get('index')} prompt_id={report.get('prompt_id')} score={quality.get('score')} reasons={item.get('reasons') or []}"
        )
        lines.append(f"    prompt={report.get('prompt_preview') or ''}")
        final_preview = report.get("final_preview")
        if final_preview:
            lines.append(f"    final={final_preview}")
    return "\n".join(lines)


def render_plan_rich(dataset_path: Path, plan: Dict[str, Any], max_examples: int = 10) -> None:
    if not RICH_AVAILABLE:
        print(render_plan_text(dataset_path, plan, max_examples=max_examples))
        return

    console = Console()
    console.print()
    console.print(
        Panel.fit(
            f"[bold]Dataset Cleanup[/bold]\n[dim]{dataset_path}[/dim]",
            border_style="cyan",
        )
    )

    summary = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
    summary.add_column("Setting", style="bold")
    summary.add_column("Value")
    for line in _render_policy_summary(plan["policy"]):
        setting, value = line.split("=", 1)
        summary.add_row(setting, value)
    summary.add_row("rows_scanned", str(plan["total_rows"]))
    summary.add_row("rows_to_remove", str(plan["removed_count"]))
    summary.add_row("rows_to_retain", str(plan["retained_count"]))
    console.print(summary)

    reasons = Table(box=box.MINIMAL_DOUBLE_HEAD, header_style="bold yellow")
    reasons.add_column("Removal reason", style="yellow")
    reasons.add_column("Count", justify="right")
    reason_counts = plan.get("removal_reason_counts") or {}
    if reason_counts:
        for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0])):
            reasons.add_row(reason, str(count))
    else:
        reasons.add_row("none", "0")
    console.print(Panel(reasons, title="Projected removals", border_style="yellow"))

    examples = Table(box=box.SIMPLE_HEAVY, header_style="bold magenta")
    examples.add_column("#", justify="right", style="bold")
    examples.add_column("Score", justify="right")
    examples.add_column("Reasons", style="yellow")
    examples.add_column("Prompt preview", overflow="fold")
    examples.add_column("Final preview", overflow="fold")
    for item in (plan.get("removed_rows") or [])[:max_examples]:
        report = item.get("report") or {}
        quality = report.get("quality") or {}
        examples.add_row(
            str(report.get("index")),
            str(quality.get("score", 0)),
            "\n".join(item.get("reasons") or []),
            str(report.get("prompt_preview") or ""),
            str(report.get("final_preview") or ""),
        )
    if not (plan.get("removed_rows") or []):
        examples.add_row("-", "-", "none", "", "")
    console.print(Panel(examples, title=f"Examples (up to {max_examples})", border_style="magenta"))


def _ask_bool(prompt: str, default: bool) -> bool:
    if Confirm is not None:
        return bool(Confirm.ask(prompt, default=default))
    suffix = "Y/n" if default else "y/N"
    response = input(f"{prompt} [{suffix}] ").strip().lower()
    if not response:
        return default
    return response in {"y", "yes"}


def _ask_optional_int(prompt: str) -> Optional[int]:
    if IntPrompt is not None:
        value = IntPrompt.ask(prompt, default=-1)
    else:
        raw = input(f"{prompt} [-1 to disable] ").strip()
        value = int(raw) if raw else -1
    return None if value < 0 else int(value)


def resolve_policy_from_args(args: argparse.Namespace) -> CleanupPolicy:
    explicit_policy = any(
        [
            args.remove_errors,
            args.remove_shallow,
            args.remove_port_conflicts,
            args.max_failed_tool_calls is not None,
            args.min_quality_score is not None,
        ]
    )
    if explicit_policy:
        return CleanupPolicy(
            remove_errors=bool(args.remove_errors),
            remove_shallow=bool(args.remove_shallow),
            remove_port_conflicts=bool(args.remove_port_conflicts),
            max_failed_tool_calls=args.max_failed_tool_calls,
            min_quality_score=args.min_quality_score,
        )
    if args.yes:
        return CleanupPolicy()
    return CleanupPolicy(
        remove_errors=_ask_bool("Remove rows with hard errors?", True),
        remove_shallow=_ask_bool("Remove suspiciously shallow completions?", True),
        remove_port_conflicts=_ask_bool("Remove rows mentioning port conflicts?", True),
        max_failed_tool_calls=_ask_optional_int("Remove rows with failed tool calls greater than"),
        min_quality_score=_ask_optional_int("Remove rows with quality score below"),
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--remove-errors", action="store_true")
    parser.add_argument("--remove-shallow", action="store_true")
    parser.add_argument("--remove-port-conflicts", action="store_true")
    parser.add_argument("--max-failed-tool-calls", type=int, default=None)
    parser.add_argument("--min-quality-score", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--plain", action="store_true")
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset_path)
    rows = _load_nonempty_lines(dataset_path)
    reports = load_reports(dataset_path)
    policy = resolve_policy_from_args(args)
    if not _policy_enabled(policy):
        print("No cleanup policy selected. Nothing to do.")
        return 0

    plan = plan_cleanup(rows, reports, policy)
    if not args.plain and RICH_AVAILABLE:
        render_plan_rich(dataset_path, plan, max_examples=args.max_examples)
    else:
        print(render_plan_text(dataset_path, plan, max_examples=args.max_examples))

    if args.dry_run or plan["removed_count"] == 0:
        return 0

    should_apply = args.yes or _ask_bool("Rewrite the dataset and create a backup?", False)
    if not should_apply:
        print("Aborted without changes.")
        return 0

    backup_path = apply_cleanup(dataset_path, plan)
    print(f"Rewrote {dataset_path} and created backup at {backup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
