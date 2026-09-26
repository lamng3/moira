"""Exceptions raised by the parser package."""

from __future__ import annotations

from pathlib import Path


class ParserError(Exception):
    """Base class for parser failures."""


class ParserConfigError(ParserError):
    """Raised when parser configuration is missing or invalid."""


class UnsupportedFormatError(ParserError):
    """Raised when an input format cannot be selected."""


class ParseError(ParserError):
    """Raised when an input cannot be decoded or parsed."""

    def __init__(
        self,
        message: str,
        *,
        source: str | Path | None = None,
        format: str | None = None,
    ) -> None:
        details = []
        if source is not None:
            details.append(f"source={source}")
        if format is not None:
            details.append(f"format={format}")
        suffix = f" ({', '.join(details)})" if details else ""
        super().__init__(f"{message}{suffix}")
        self.source = Path(source) if source is not None else None
        self.format = format
