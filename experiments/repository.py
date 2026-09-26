"""Result persistence abstractions."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from .models import ExperimentResult
from .reporting import SuiteReporter


class ResultRepository(ABC):
    """Repository interface for storing experiment metadata."""

    @property
    @abstractmethod
    def output_directory(self) -> Path:
        """Directory where execution logs may be written."""

    @abstractmethod
    def save(self, result: ExperimentResult) -> Path:
        """Persist one experiment result and return its path."""

    @abstractmethod
    def save_suite(self, results: Iterable[ExperimentResult]) -> Path:
        """Persist the suite summary and return its path."""


class JsonResultRepository(ResultRepository):
    """Store versioned manifests beside concise Markdown summaries."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def output_directory(self) -> Path:
        return self.root

    def save(self, result: ExperimentResult) -> Path:
        run_directory = self.root / "runs" / result.name / result.run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        path = run_directory / "manifest.json"
        self._write(path, result.to_dict())
        summary = self._summary(result)
        (run_directory / "summary.md").write_text(summary, encoding="utf-8")
        self._write(self.root / f"{result.name}.latest.json", result.to_dict())
        (self.root / f"{result.name}.latest.md").write_text(
            summary, encoding="utf-8"
        )
        return path

    def save_suite(self, results: Iterable[ExperimentResult]) -> Path:
        materialized = list(results)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "suite.manifest.json"
        self._write(
            path,
            {
                "schema_version": 1,
                "experiments": [result.to_dict() for result in materialized],
            },
        )
        SuiteReporter().write(self.root, materialized)
        return path

    @staticmethod
    def _write(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)

    @classmethod
    def _summary(cls, result: ExperimentResult) -> str:
        reproducible = (
            "Yes—metrics match the previous run."
            if result.reproducible is True
            else "No—metrics differ from the previous run."
            if result.reproducible is False
            else "First comparable run."
        )
        lines = [
            f"# {result.name}",
            "",
            f"- **Status:** {result.status}",
            f"- **Run ID:** `{result.run_id}`",
            f"- **Configuration:** `{result.config_fingerprint[:12]}`",
            f"- **Source:** `{result.source_revision[:12]}`"
            + (" (dirty working tree)" if result.source_dirty else ""),
            f"- **Reproducible:** {reproducible}",
            f"- **Duration:** {result.duration_seconds:.3f} seconds",
        ]
        if result.result_fingerprint:
            lines.append(f"- **Result fingerprint:** `{result.result_fingerprint}`")
        if result.error:
            lines.append(f"- **Error:** {result.error}")
        if result.metrics:
            lines.extend(
                [
                    "",
                    "## Metrics",
                    "",
                    "| Metric | Value |",
                    "| --- | ---: |",
                ]
            )
            for key, value in cls._flatten(result.metrics):
                display = json.dumps(value, sort_keys=True)
                lines.append(f"| `{key}` | {display} |")
        lines.extend(
            [
                "",
                "## Artifacts",
                "",
                f"- Output: `{result.output_file or 'none'}`",
                f"- Metrics: `{result.metrics_file or 'none'}`",
                f"- Manifest: `{result.artifact_directory or '.'}/manifest.json`",
            ]
        )
        return "\n".join(lines) + "\n"

    @classmethod
    def _flatten(
        cls, value: object, prefix: str = ""
    ) -> list[tuple[str, object]]:
        rows: list[tuple[str, object]] = []
        if isinstance(value, dict):
            for key in sorted(value):
                nested = f"{prefix}.{key}" if prefix else str(key)
                rows.extend(cls._flatten(value[key], nested))
        else:
            rows.append((prefix, value))
        return rows
