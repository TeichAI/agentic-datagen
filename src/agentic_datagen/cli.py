from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .dataset_cleanup import main as cleanup_main
from .dataset_qa import main as qa_main
from .generator import main as generate_main


def _normalize_argv(argv: Optional[List[str]] = None) -> List[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return args
    if args[0] in {"-h", "--help", "help"}:
        return args
    if args[0] in {"generate", "qa", "cleanup"}:
        return args
    return ["generate", *args]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agentic dataset generation and dataset QA utilities"
    )
    subparsers = parser.add_subparsers(dest="command")

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate or resume an agentic dataset run",
    )
    generate_parser.add_argument(
        "-c", "--config", required=True, help="Path to configuration YAML file"
    )

    qa_parser = subparsers.add_parser(
        "qa",
        help="Run dataset QA checks and print a summary",
    )
    qa_parser.add_argument("dataset_path")
    qa_parser.add_argument("--limit", type=int, default=None)
    qa_parser.add_argument("--json", action="store_true", dest="as_json")
    qa_parser.add_argument("--plain", action="store_true")
    qa_parser.add_argument("--max-flagged", type=int, default=20)
    qa_parser.add_argument("--fail-on-errors", action="store_true")

    cleanup_parser = subparsers.add_parser(
        "cleanup",
        help="Plan or apply dataset cleanup rules",
    )
    cleanup_parser.add_argument("dataset_path")
    cleanup_parser.add_argument("--remove-errors", action="store_true")
    cleanup_parser.add_argument("--remove-shallow", action="store_true")
    cleanup_parser.add_argument("--remove-port-conflicts", action="store_true")
    cleanup_parser.add_argument("--max-failed-tool-calls", type=int, default=None)
    cleanup_parser.add_argument("--min-quality-score", type=int, default=None)
    cleanup_parser.add_argument("--max-examples", type=int, default=10)
    cleanup_parser.add_argument("--dry-run", action="store_true")
    cleanup_parser.add_argument("--yes", action="store_true")
    cleanup_parser.add_argument("--plain", action="store_true")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    normalized_argv = _normalize_argv(argv)
    args = parser.parse_args(normalized_argv)

    if args.command == "generate":
        return generate_main(["-c", args.config])
    if args.command == "qa":
        qa_args = [args.dataset_path]
        if args.limit is not None:
            qa_args.extend(["--limit", str(args.limit)])
        if args.as_json:
            qa_args.append("--json")
        if args.plain:
            qa_args.append("--plain")
        if args.max_flagged != 20:
            qa_args.extend(["--max-flagged", str(args.max_flagged)])
        if args.fail_on_errors:
            qa_args.append("--fail-on-errors")
        return qa_main(qa_args)
    if args.command == "cleanup":
        cleanup_args = [args.dataset_path]
        if args.remove_errors:
            cleanup_args.append("--remove-errors")
        if args.remove_shallow:
            cleanup_args.append("--remove-shallow")
        if args.remove_port_conflicts:
            cleanup_args.append("--remove-port-conflicts")
        if args.max_failed_tool_calls is not None:
            cleanup_args.extend(["--max-failed-tool-calls", str(args.max_failed_tool_calls)])
        if args.min_quality_score is not None:
            cleanup_args.extend(["--min-quality-score", str(args.min_quality_score)])
        if args.max_examples != 10:
            cleanup_args.extend(["--max-examples", str(args.max_examples)])
        if args.dry_run:
            cleanup_args.append("--dry-run")
        if args.yes:
            cleanup_args.append("--yes")
        if args.plain:
            cleanup_args.append("--plain")
        return cleanup_main(cleanup_args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
