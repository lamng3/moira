"""Typed configuration and result models for reproducible experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


class ConfigurationError(ValueError):
    """Raised when an experiment configuration is invalid."""


@dataclass(frozen=True)
class ExperimentSpec:
    """A single experiment described independently of its execution strategy."""

    name: str
    kind: str = "standard"
    ontology: List[str] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    entrypoint: Optional[str] = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentSpec":
        name = str(value.get("name", "")).strip()
        if not name:
            raise ConfigurationError("Each experiment requires a non-empty 'name'.")

        kind = str(value.get("kind", "standard")).strip().lower()
        ontology = [str(item) for item in value.get("ontology", [])]
        options = dict(value.get("options", {}))
        environment = {
            str(key): str(item)
            for key, item in dict(value.get("environment", {})).items()
        }
        entrypoint = value.get("entrypoint")

        if kind == "standard" and not ontology:
            raise ConfigurationError(
                f"Standard experiment '{name}' requires at least one ontology."
            )
        if entrypoint is not None:
            entrypoint = str(entrypoint)

        return cls(
            name=name,
            kind=kind,
            ontology=ontology,
            options=options,
            environment=environment,
            entrypoint=entrypoint,
        )


@dataclass(frozen=True)
class ExperimentSuite:
    """Top-level experiment configuration."""

    experiments: List[ExperimentSpec]
    results_dir: Path = Path("results/experiments")
    continue_on_error: bool = False

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentSuite":
        version = value.get("schema_version", 1)
        if version != 1:
            raise ConfigurationError(
                f"Unsupported schema_version {version!r}; expected 1."
            )

        raw_experiments = value.get("experiments", [])
        if not isinstance(raw_experiments, list) or not raw_experiments:
            raise ConfigurationError("'experiments' must be a non-empty list.")

        experiments = [ExperimentSpec.from_dict(item) for item in raw_experiments]
        names = [experiment.name for experiment in experiments]
        if len(names) != len(set(names)):
            raise ConfigurationError("Experiment names must be unique.")

        return cls(
            experiments=experiments,
            results_dir=Path(str(value.get("results_dir", "results/experiments"))),
            continue_on_error=bool(value.get("continue_on_error", False)),
        )


@dataclass(frozen=True)
class ExperimentResult:
    """Serializable execution metadata for a single experiment."""

    name: str
    kind: str
    command: List[str]
    status: str
    exit_code: Optional[int]
    started_at: str
    finished_at: str
    duration_seconds: float
    run_id: str
    config_fingerprint: str
    source_revision: str
    source_dirty: bool
    output_file: Optional[str] = None
    artifact_directory: Optional[str] = None
    metrics_file: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    result_fingerprint: Optional[str] = None
    reproducible: Optional[bool] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "command": self.command,
            "status": self.status,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "run_id": self.run_id,
            "config_fingerprint": self.config_fingerprint,
            "source_revision": self.source_revision,
            "source_dirty": self.source_dirty,
            "output_file": self.output_file,
            "artifact_directory": self.artifact_directory,
            "metrics_file": self.metrics_file,
            "metrics": self.metrics,
            "result_fingerprint": self.result_fingerprint,
            "reproducible": self.reproducible,
            "error": self.error,
        }
