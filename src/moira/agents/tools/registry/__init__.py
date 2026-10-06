from __future__ import annotations

import json
import re
from os import PathLike
from pathlib import Path
from typing import Any, Union

from moira.agents.tools.common.resources import PackageResource, package_resource

RegistrySource = Union[str, PathLike[str], PackageResource]


def _load_object(raw: str) -> dict[str, Any] | None:
    def decode(candidate: str) -> dict[str, Any] | None:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            candidate = re.sub(r"//[^\n]*|/\*[\s\S]*?\*/", "", candidate)
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                return None
        return value if isinstance(value, dict) else None

    direct = decode(raw)
    if direct is not None:
        return direct
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    quoted = escaped = False
    for index in range(start, len(raw)):
        character = raw[index]
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return decode(raw[start : index + 1])
    return None


def default_registry_resource() -> PackageResource:
    return package_resource(__package__, "ontology_tools.json")


def sanitize_module_path(module_path: str | None) -> str | None:
    if not module_path:
        return None
    value = re.sub(r"\s+", "", module_path.strip().replace("/", "."))
    value = re.sub(r"\.{2,}", ".", value).strip(".")
    return value or None


def load_registry(
    source: RegistrySource | None = None,
) -> dict[str, dict[str, Any]]:
    resource = source or default_registry_resource()
    if isinstance(resource, (str, PathLike)):
        resource = Path(resource)
    try:
        if not resource.is_file():
            return {}
        data = _load_object(resource.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}

    registry: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        name = str(value.get("tool_name") or key or "").strip()
        if name:
            registry[name] = {
                "tool_name": name,
                "module_path": sanitize_module_path(value.get("module_path")),
                "tool_type": str(value.get("tool_type") or "module").lower(),
                "arguments": dict(value.get("arguments") or {}),
                "aliases": list(value.get("aliases") or []),
            }
    return registry


__all__ = [
    "default_registry_resource",
    "load_registry",
    "sanitize_module_path",
]
