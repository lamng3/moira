from __future__ import annotations

import json
from importlib import import_module
from importlib.resources import files
from pathlib import Path
from typing import Any, Protocol


class PackageResource(Protocol):
    def is_file(self) -> bool: ...

    def read_text(self, encoding: str = "utf-8") -> str: ...


def package_resource(package: str, name: str) -> PackageResource:
    """Return a package resource without assuming a source-tree layout."""
    return files(package).joinpath(name)


def read_json_resource(package: str, name: str) -> Any:
    text = package_resource(package, name).read_text(encoding="utf-8")
    if not text.strip():
        module_file = getattr(import_module(package), "__file__", None)
        if module_file:
            text = Path(module_file).with_name(name).read_text(encoding="utf-8")
    return json.loads(text)
