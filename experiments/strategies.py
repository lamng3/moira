"""Command-building strategies for different experiment families."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Type

from .models import ConfigurationError, ExperimentSpec


def _option_arguments(options: Dict[str, Any]) -> Iterable[str]:
    """Convert JSON options into deterministic command-line arguments."""
    for key in sorted(options):
        value = options[key]
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                yield flag
        elif value is None:
            continue
        elif isinstance(value, list):
            yield flag
            yield from (str(item) for item in value)
        elif isinstance(value, (str, int, float)):
            yield flag
            yield str(value)
        else:
            raise ConfigurationError(
                f"Option '{key}' must be a scalar, list, null, or boolean; "
                f"received {type(value).__name__}."
            )


class ExperimentStrategy(ABC):
    """Strategy interface for translating a specification into a command."""

    @abstractmethod
    def build_command(self, spec: ExperimentSpec, project_root: Path) -> List[str]:
        """Build the command used to execute ``spec``."""


class StandardExperimentStrategy(ExperimentStrategy):
    """Run the primary ontology integration pipeline."""

    def build_command(self, spec: ExperimentSpec, project_root: Path) -> List[str]:
        if spec.entrypoint:
            command = [sys.executable, str(project_root / spec.entrypoint)]
        else:
            command = [sys.executable, "-m", "agentoi.cli", "run"]
        command.extend(["--ontology", *spec.ontology])
        command.extend(_option_arguments(spec.options))
        return command


class CommandExperimentStrategy(ExperimentStrategy):
    """Run an explicit argv list for one-off or external baselines."""

    def build_command(self, spec: ExperimentSpec, project_root: Path) -> List[str]:
        command = spec.options.get("command")
        if not isinstance(command, list) or not command:
            raise ConfigurationError(
                f"Command experiment '{spec.name}' requires a non-empty "
                "options.command list."
            )
        values = [str(item) for item in command]
        if values[0] == "{python}":
            values[0] = sys.executable
        return values


class ExperimentStrategyFactory:
    """Factory and extension point for experiment execution strategies."""

    _strategies: Dict[str, Type[ExperimentStrategy]] = {
        "standard": StandardExperimentStrategy,
        "command": CommandExperimentStrategy,
    }

    @classmethod
    def register(cls, kind: str, strategy: Type[ExperimentStrategy]) -> None:
        if not kind.strip():
            raise ValueError("Strategy kind cannot be empty.")
        cls._strategies[kind.strip().lower()] = strategy

    @classmethod
    def create(cls, kind: str) -> ExperimentStrategy:
        try:
            strategy = cls._strategies[kind.lower()]
        except KeyError as exc:
            available = ", ".join(sorted(cls._strategies))
            raise ConfigurationError(
                f"Unknown experiment kind '{kind}'. Available kinds: {available}."
            ) from exc
        return strategy()
