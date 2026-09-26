"""Local input helpers shared by parser adapters."""

from __future__ import annotations

import bz2
import gzip
from pathlib import Path

from .errors import ParseError


def read_text(path: str | Path) -> str:
    """Read plain, gzip, or bzip2 text with deterministic decoding."""
    source = Path(path)
    if not source.exists():
        raise ParseError("Input does not exist", source=source)
    if not source.is_file():
        raise ParseError("Input is not a file", source=source)

    try:
        if source.suffix.lower() == ".gz":
            data = gzip.open(source, "rb").read()
        elif source.suffix.lower() == ".bz2":
            data = bz2.open(source, "rb").read()
        else:
            data = source.read_bytes()
    except OSError as exc:
        raise ParseError("Unable to read input", source=source) from exc

    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return data.decode(encoding).replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError:
            continue
    raise ParseError("Unable to decode input text", source=source)
