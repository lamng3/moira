"""Command-line interface for experiment suites."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import tempfile
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from .models import ConfigurationError, ExperimentSuite
from .repository import JsonResultRepository
from .runner import ExperimentRunner
from .tracking import TrackingError, create_tracker


def load_suite(path: Path) -> ExperimentSuite:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError("The top-level configuration must be an object.")
    return ExperimentSuite.from_dict(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible AgentOI experiment suites."
    )
    parser.add_argument("config", type=Path, help="Path to a JSON suite config.")
    parser.add_argument(
        "--select",
        nargs="+",
        metavar="NAME",
        help="Run only the named experiments.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print commands without executing them.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (normally detected automatically).",
    )
    parser.add_argument(
        "--tracker",
        choices=("local", "wandb"),
        default=os.environ.get("AGENTOI_TRACKER", "local"),
        help="Result destination (default: AGENTOI_TRACKER or local).",
    )
    parser.add_argument(
        "--tracking-project",
        default=os.environ.get("WANDB_PROJECT", "agentoi"),
        help="W&B project name.",
    )
    parser.add_argument(
        "--tracking-entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="Optional W&B team or account.",
    )
    parser.add_argument(
        "--keep-local",
        action="store_true",
        help="Retain local artifacts when using a hosted tracker.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    tracker = create_tracker(
        args.tracker,
        project=args.tracking_project,
        entity=args.tracking_entity,
    )
    temporary: tempfile.TemporaryDirectory[str] | None = None
    try:
        suite = load_suite(args.config)
        if tracker.is_remote and not args.keep_local:
            temporary = tempfile.TemporaryDirectory(prefix="agentoi-results-")
            results_dir = Path(temporary.name) / "artifacts"
        else:
            results_dir = (
                suite.results_dir
                if suite.results_dir.is_absolute()
                else args.project_root / suite.results_dir
            )
        runner = ExperimentRunner(
            args.project_root,
            JsonResultRepository(results_dir),
            dry_run=args.dry_run,
        )
        results = runner.run_suite(suite, selected_names=args.select)
        tracking_url = None
        if not args.dry_run:
            tracking_url = tracker.publish_suite(
                args.config.stem, results, results_dir
            )
    except (ConfigurationError, TrackingError, ValueError) as exc:
        print(f"Configuration error: {exc}")
        return 2
    finally:
        if temporary is not None:
            temporary.cleanup()

    for result in results:
        command = shlex.join(result.command)
        print(f"[{result.status:7}] {result.name}: {command}")
        if result.output_file and (not tracker.is_remote or args.keep_local):
            print(f"          log: {result.output_file}")
    if tracking_url:
        print(f"Dashboard: {tracking_url}")

    return 1 if any(result.status == "failed" for result in results) else 0
