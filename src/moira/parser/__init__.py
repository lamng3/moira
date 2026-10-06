"""Public parsing API."""

from .config import FileRule, ParserConfig, ParserSelection
from .errors import ParseError, ParserConfigError, ParserError, UnsupportedFormatError
from .formats import InputFormat, detect_format, parse_format
from .json import JSONParser
from .parser import Parser
from .rdf import RDFParser
from .text import TextParser

__all__ = [
    "FileRule",
    "InputFormat",
    "JSONParser",
    "ParseError",
    "Parser",
    "ParserConfig",
    "ParserConfigError",
    "ParserError",
    "ParserSelection",
    "RDFParser",
    "TextParser",
    "UnsupportedFormatError",
    "detect_format",
    "parse_format",
]
