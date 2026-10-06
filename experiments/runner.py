"""Template runner for reproducible local experiments."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence

from .models import ExperimentResult, ExperimentSpec, ExperimentSuite
from .repository import ResultRepository
from .strategies import ExperimentStrategyFactory


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _spec_fingerprint(spec: ExperimentSpec) -> str:
    return _fingerprint(
        {
            "name": spec.name,
            "kind": spec.kind,
            "ontology": spec.ontology,
            "options": spec.options,
            "environment": spec.environment,
            "entrypoint": spec.entrypoint,
        }
    )


def _source_state(project_root: Path) -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _relative(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _portable_command(
    command: list[str],
    project_root: Path,
    artifact_root: Path,
) -> list[str]:
    portable: list[str] = []
    executable = Path(sys.executable).resolve()
    for value in command:
        path = Path(value)
        if path.is_absolute() and path.resolve() == executable:
            portable.append("{python}")
        elif path.is_absolute():
            project_value = _relative(path, project_root)
            if Path(project_value).is_absolute():
                artifact_value = _relative(path, artifact_root)
                portable.append(
                    f"{{artifacts}}/{artifact_value}"
                    if not Path(artifact_value).is_absolute()
                    else artifact_value
                )
            else:
                portable.append(project_value)
        else:
            portable.append(value)
    return portable


class ExperimentRunner:
    """Execute every experiment through the same prepare/run/record template."""

    def __init__(
        self,
        project_root: Path,
        repository: ResultRepository,
        *,
        dry_run: bool = False,
    ) -> None:
        self.project_root = project_root.resolve()
        self.repository = repository
        self.dry_run = dry_run

    def run_suite(
        self,
        suite: ExperimentSuite,
        *,
        selected_names: Optional[Sequence[str]] = None,
    ) -> List[ExperimentResult]:
        selected = set(selected_names or [])
        experiments = [
            spec
            for spec in suite.experiments
            if not selected or spec.name in selected
        ]
        missing = selected - {spec.name for spec in experiments}
        if missing:
            raise ValueError(
                f"Unknown selected experiment(s): {', '.join(sorted(missing))}"
            )

        results: List[ExperimentResult] = []
        for spec in experiments:
            result = self.run_experiment(spec)
            results.append(result)
            self.repository.save(result)
            if result.status == "failed" and not suite.continue_on_error:
                break

        self.repository.save_suite(results)
        return results

    def run_experiment(self, spec: ExperimentSpec) -> ExperimentResult:
        config_fingerprint = _spec_fingerprint(spec)
        run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"{run_stamp}-{config_fingerprint[:8]}"
        source_revision, source_dirty = _source_state(self.project_root)
        run_directory = (
            self.repository.output_directory / "runs" / spec.name / run_id
        )
        output_path = run_directory / "stdout.log"
        metrics_path = run_directory / "metrics.json"
        strategy = ExperimentStrategyFactory.create(spec.kind)
        command = strategy.build_command(spec, self.project_root)
        if spec.kind == "standard" and "--output" not in command:
            command.extend(["--output", str(run_directory / "pipeline.log")])
        started_at = _utc_now()
        started = time.monotonic()

        if self.dry_run:
            return ExperimentResult(
                name=spec.name,
                kind=spec.kind,
                command=_portable_command(
                    command, self.project_root, self.repository.output_directory
                ),
                status="dry-run",
                exit_code=None,
                started_at=started_at,
                finished_at=_utc_now(),
                duration_seconds=time.monotonic() - started,
                run_id=run_id,
                config_fingerprint=config_fingerprint,
                source_revision=source_revision,
                source_dirty=source_dirty,
                artifact_directory=self._artifact_reference(run_directory),
            )

        run_directory.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(spec.environment)
        environment.setdefault("PYTHONPATH", str(self.project_root / "src"))
        environment.setdefault("PYTHONUNBUFFERED", "1")
        environment["MOIRA_METRICS_PATH"] = str(metrics_path)

        error = None
        exit_code = None
        try:
            with output_path.open("w", encoding="utf-8") as output:
                process = subprocess.run(
                    command,
                    cwd=self.project_root,
                    env=environment,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            exit_code = process.returncode
            status = "passed" if process.returncode == 0 else "failed"
        except OSError as exc:
            status = "failed"
            error = str(exc)

        metrics: dict[str, Any] = {}
        if metrics_path.is_file():
            try:
                value = json.loads(metrics_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    metrics = value
                else:
                    error = "metrics.json must contain a JSON object"
                    status = "failed"
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                error = f"Unable to read metrics: {exc}"
                status = "failed"

        result_fingerprint = _fingerprint(metrics) if metrics else None
        reproducible = self._compare_previous(
            spec.name,
            config_fingerprint=config_fingerprint,
            source_revision=source_revision,
            result_fingerprint=result_fingerprint,
        )

        return ExperimentResult(
            name=spec.name,
            kind=spec.kind,
            command=_portable_command(
                command, self.project_root, self.repository.output_directory
            ),
            status=status,
            exit_code=exit_code,
            started_at=started_at,
            finished_at=_utc_now(),
            duration_seconds=time.monotonic() - started,
            run_id=run_id,
            config_fingerprint=config_fingerprint,
            source_revision=source_revision,
            source_dirty=source_dirty,
            output_file=self._artifact_reference(output_path),
            artifact_directory=self._artifact_reference(run_directory),
            metrics_file=(
                self._artifact_reference(metrics_path) if metrics else None
            ),
            metrics=metrics,
            result_fingerprint=result_fingerprint,
            reproducible=reproducible,
            error=error,
        )

    def _artifact_reference(self, path: Path) -> str:
        project_value = _relative(path, self.project_root)
        if not Path(project_value).is_absolute():
            return project_value
        return _relative(path, self.repository.output_directory)

    def _compare_previous(
        self,
        name: str,
        *,
        config_fingerprint: str,
        source_revision: str,
        result_fingerprint: str | None,
    ) -> bool | None:
        if result_fingerprint is None:
            return None
        root = self.repository.output_directory / "runs" / name
        for path in sorted(root.glob("*/manifest.json"), reverse=True):
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if (
                previous.get("status") == "passed"
                and previous.get("config_fingerprint") == config_fingerprint
                and previous.get("source_revision") == source_revision
                and previous.get("result_fingerprint")
            ):
                return previous["result_fingerprint"] == result_fingerprint
        return None
