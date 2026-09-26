from __future__ import annotations

import importlib
import json
import re
import unicodedata
from typing import Any

from agentoi.agents.tools.registry import (
    RegistrySource,
    load_registry,
    sanitize_module_path,
)

FALLBACK_MODULES = {
    "ontology_term_info": "agentoi.agents.tools.ontology.ontology_lookup",
    "search_knowledge_graph": "agentoi.agents.tools.knowledge_graph.knowledge_graph_lookup",
}

_CODEBLOCK_RE = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_-]*)\s*(?P<body>[\s\S]*?)\s*```", re.MULTILINE
)
_SECRET_KEYS = {
    "api_key", "apikey", "api-key", "authorization", "auth", "token",
    "password", "passwd", "pwd", "secret", "client_secret", "access_token",
    "private_key", "key",
}


def safe_slug(value: str, limit: int = 50) -> str:
    value = (value or "").strip()
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii") or value
    value = re.sub(r"\s+", " ", value).strip()[:limit]
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    return re.sub(r"_+", "_", value).strip("._") or "query"


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "******" if isinstance(key, str) and key.lower() in _SECRET_KEYS else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return type(value)(redact(item) for item in value)
    if isinstance(value, str):
        value = re.sub(r"(?i)\bBearer\s+\S+", "Bearer ******", value)
        value = re.sub(r"(?i)\bsk-[A-Za-z0-9]{16,}\b", "sk-******", value)
        value = re.sub(r"(?i)\bghp_[A-Za-z0-9]{20,}\b", "ghp_******", value)
    return value


def preview(value: Any, limit: int | None = 240) -> str:
    try:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    text = re.sub(r"[\r\n\t]+", " ", text).strip()
    if limit is None or len(text) <= limit:
        return text
    return text[:limit] + "…"


def _json_object(text: str) -> dict[str, Any] | None:
    def load(candidate: str) -> dict[str, Any] | None:
        try:
            result = json.loads(candidate)
            return result if isinstance(result, dict) else None
        except Exception:
            try:
                candidate = re.sub(r"//[^\n]*|/\*[\s\S]*?\*/", "", candidate)
                candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
                result = json.loads(candidate)
                return result if isinstance(result, dict) else None
            except Exception:
                return None

    direct = load(text)
    if direct is not None:
        return direct
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    quoted = escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return load(text[start : index + 1])
    return None


def extract_tool_json(content: str) -> dict[str, Any] | None:
    text = (content or "").strip()
    direct = _json_object(text)
    if direct is not None:
        return direct
    for match in _CODEBLOCK_RE.finditer(text):
        parsed = _json_object((match.group("body") or "").strip())
        if parsed is not None:
            return parsed
    return None


def load_tools_json(
    tools_path: RegistrySource | None,
) -> dict[str, dict[str, Any]]:
    """Backward-compatible entry point for loading a tool registry."""
    return load_registry(tools_path)


def resolve_spec_for(registry: dict[str, Any], tool_name: str) -> dict[str, Any] | None:
    if not tool_name:
        return None
    direct = registry.get(tool_name)
    if isinstance(direct, dict):
        return direct
    name = tool_name.lower()
    for spec in registry.values():
        if not isinstance(spec, dict):
            continue
        aliases = [str(alias).lower() for alias in spec.get("aliases") or []]
        if str(spec.get("tool_name") or "").lower() == name or name in aliases:
            return spec
    return None


def coerce_tool_call(
    tool_call: dict[str, Any],
    registry: dict[str, Any],
    fallbacks: dict[str, str] | None = None,
) -> dict[str, Any]:
    name = str(tool_call.get("tool_name") or "").strip()
    spec = resolve_spec_for(registry, name)
    arguments = dict(spec.get("arguments") or {}) if spec else {}
    arguments.update(dict(tool_call.get("arguments") or {}))
    if "ontologies" not in arguments and "sources" in arguments:
        arguments["ontologies"] = arguments.pop("sources")
    if isinstance(arguments.get("max_results"), str):
        try:
            arguments["max_results"] = int(arguments["max_results"])
        except ValueError:
            pass
    if isinstance(arguments.get("exact"), str):
        arguments["exact"] = arguments["exact"].strip().lower() in {"true", "1", "yes"}

    module_path = sanitize_module_path(tool_call.get("module_path"))
    if module_path is None and spec:
        module_path = sanitize_module_path(spec.get("module_path"))
    if module_path is None:
        module_path = sanitize_module_path((fallbacks or {}).get(name))
    return {
        "tool_name": name,
        "tool_type": str(
            tool_call.get("tool_type") or (spec.get("tool_type") if spec else None) or "module"
        ).lower(),
        "arguments": arguments,
        "module_path": module_path,
    }


def execute_tool_call(
    tool_call: dict[str, Any],
    registry: dict[str, Any],
    *,
    fallbacks: dict[str, str] | None = None,
) -> Any:
    """Execute the first resolvable module function for a normalized tool call."""

    name = tool_call["tool_name"]
    spec = resolve_spec_for(registry, name)
    candidates = [
        tool_call.get("module_path"),
        spec.get("module_path") if spec else None,
        (fallbacks or {}).get(name),
    ]
    last_error: Exception | None = None
    attempted: list[str] = []
    for module_path in dict.fromkeys(path for path in candidates if path):
        attempted.append(module_path)
        try:
            function = getattr(importlib.import_module(module_path), name)
            return function(**(tool_call.get("arguments") or {}))
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ModuleNotFoundError(f"Could not import module for {name}; tried: {attempted}")
