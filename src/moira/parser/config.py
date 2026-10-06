"""Configuration for path-based parser selection."""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ParserConfigError
from .formats import InputFormat, detect_format, parse_format


@dataclass(frozen=True)
class ParserSelection:
    format: InputFormat
    namespaces: dict[str, str] = field(default_factory=dict)
    name: str | None = None
    version: str | None = None


@dataclass(frozen=True)
class FileRule:
    pattern: str
    format: InputFormat | None = None
    namespaces: dict[str, str] = field(default_factory=dict)
    name: str | None = None
    version: str | None = None

    def matches(self, path: Path) -> bool:
        return fnmatch.fnmatch(path.as_posix(), self.pattern) or fnmatch.fnmatch(
            path.name, self.pattern
        )


@dataclass(frozen=True)
class ParserConfig:
    namespaces: dict[str, str] = field(default_factory=dict)
    files: tuple[FileRule, ...] = ()

    @classmethod
    def from_file(cls, path: str | Path) -> "ParserConfig":
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ParserConfigError(f"Unable to load parser config {source}: {exc}") from exc
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ParserConfig":
        if not isinstance(raw, dict):
            raise ParserConfigError("Parser config must be a JSON object")
        global_config = raw.get("global", {})
        rules_config = raw.get("files", [])
        if not isinstance(global_config, dict) or not isinstance(rules_config, list):
            raise ParserConfigError("'global' must be an object and 'files' must be a list")

        namespaces = _namespaces(global_config.get("namespaces", {}), "global")
        rules: list[FileRule] = []
        for index, value in enumerate(rules_config):
            if not isinstance(value, dict) or not isinstance(value.get("pattern"), str):
                raise ParserConfigError(f"files[{index}] must contain a string 'pattern'")
            configured_format = value.get("format")
            rules.append(
                FileRule(
                    pattern=value["pattern"],
                    format=parse_format(configured_format) if configured_format else None,
                    namespaces=_namespaces(
                        value.get("namespaces", {}), f"files[{index}]"
                    ),
                    name=_optional_string(value, "name", index),
                    version=_optional_string(value, "version", index),
                )
            )
        return cls(namespaces=namespaces, files=tuple(rules))

    def select(
        self,
        path: str | Path,
        format: str | InputFormat | None = None,
    ) -> ParserSelection:
        source = Path(path)
        fallback_format = parse_format(format) if format is not None else None
        for rule in self.files:
            if rule.matches(source):
                return ParserSelection(
                    format=rule.format or fallback_format or detect_format(source),
                    namespaces={**self.namespaces, **rule.namespaces},
                    name=rule.name,
                    version=rule.version,
                )
        return ParserSelection(
            format=fallback_format or detect_format(source),
            namespaces=dict(self.namespaces),
        )


def _namespaces(value: object, location: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(uri, str) for key, uri in value.items()
    ):
        raise ParserConfigError(f"{location}.namespaces must map strings to strings")
    return dict(value)


def _optional_string(value: dict[str, Any], key: str, index: int) -> str | None:
    result = value.get(key)
    if result is not None and not isinstance(result, str):
        raise ParserConfigError(f"files[{index}].{key} must be a string")
    return result
