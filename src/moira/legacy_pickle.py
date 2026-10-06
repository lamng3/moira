"""Load pickles written before the package was renamed from agentoi to moira."""

from __future__ import annotations

import pickle
from typing import Any, BinaryIO

_OLD_PACKAGE = "agentoi"


class _RenamingUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == _OLD_PACKAGE or module.startswith(_OLD_PACKAGE + "."):
            module = "moira" + module[len(_OLD_PACKAGE):]
        return super().find_class(module, name)


def load(file: BinaryIO) -> Any:
    """``pickle.load`` that also reads graphs cached under the old package name."""
    return _RenamingUnpickler(file).load()
