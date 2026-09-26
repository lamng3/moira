"""Optional hosted experiment tracking adapters."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

from .models import ExperimentResult
from .reporting import flatten_metrics


class TrackingError(RuntimeError):
    """Raised when a configured tracker cannot publish a suite."""


class ExperimentTracker(ABC):
    """Adapter interface for publishing completed experiment suites."""

    is_remote = False

    @abstractmethod
    def publish_suite(
        self,
        suite_name: str,
        results: Iterable[ExperimentResult],
        artifact_directory: Path,
    ) -> str | None:
        """Publish a suite and return its dashboard URL when available."""


class LocalTracker(ExperimentTracker):
    """Keep artifacts in the repository's configured results directory."""

    def publish_suite(
        self,
        suite_name: str,
        results: Iterable[ExperimentResult],
        artifact_directory: Path,
    ) -> str | None:
        return None


class WandbTracker(ExperimentTracker):
    """Publish metrics, comparisons, and artifacts to Weights & Biases."""

    is_remote = True

    def __init__(self, project: str = "agentoi", entity: str | None = None) -> None:
        self.project = project
        self.entity = entity

    def publish_suite(
        self,
        suite_name: str,
        results: Iterable[ExperimentResult],
        artifact_directory: Path,
    ) -> str | None:
        try:
            import wandb
        except ImportError as exc:
            raise TrackingError(
                "W&B tracking requires the 'tracking' extra: "
                "uv sync --extra tracking"
            ) from exc

        materialized = list(results)
        columns, rows = _table(materialized)
        run: Any = None
        try:
            run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=suite_name,
                job_type="experiment-suite",
                dir=str(artifact_directory.parent),
                config={"suite": suite_name, "experiments": len(materialized)},
            )
            run.log({"results": wandb.Table(columns=columns, data=rows)})
            for result in materialized:
                for metric, value in flatten_metrics(result.metrics).items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        run.summary[f"{result.name}/{metric}"] = value
                run.summary[f"{result.name}/status"] = result.status
                run.summary[f"{result.name}/reproducible"] = result.reproducible

            artifact = wandb.Artifact(
                name=_artifact_name(suite_name),
                type="experiment-results",
                metadata={
                    "suite": suite_name,
                    "experiments": [result.name for result in materialized],
                },
            )
            artifact.add_dir(str(artifact_directory))
            run.log_artifact(artifact)
            return str(run.url) if run.url else None
        except Exception as exc:
            raise TrackingError(f"Unable to publish results to W&B: {exc}") from exc
        finally:
            if run is not None:
                run.finish()


def create_tracker(
    name: str,
    *,
    project: str = "agentoi",
    entity: str | None = None,
) -> ExperimentTracker:
    """Create a tracker without importing optional SDKs prematurely."""
    normalized = name.strip().lower()
    if normalized == "local":
        return LocalTracker()
    if normalized == "wandb":
        return WandbTracker(project=project, entity=entity)
    raise TrackingError("Tracker must be 'local' or 'wandb'.")


def _table(
    results: list[ExperimentResult],
) -> tuple[list[str], list[list[object]]]:
    metric_names = sorted(
        {
            name
            for result in results
            for name in flatten_metrics(result.metrics)
        }
    )
    columns = [
        "experiment",
        "status",
        "reproducible",
        "duration_seconds",
        *metric_names,
    ]
    rows: list[list[object]] = []
    for result in results:
        metrics = flatten_metrics(result.metrics)
        rows.append(
            [
                result.name,
                result.status,
                result.reproducible,
                result.duration_seconds,
                *(metrics.get(name) for name in metric_names),
            ]
        )
    return columns, rows


def _artifact_name(suite_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", suite_name).strip("-")
    return f"{safe or 'suite'}-results"
