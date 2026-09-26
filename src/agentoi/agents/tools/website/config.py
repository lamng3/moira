from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, ClassVar

@dataclass
class Config:
    TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")

    # HTTP defaults
    DEFAULT_TIMEOUT: float = float(os.getenv("HTTP_TIMEOUT", 15))
    USER_AGENT: str = os.getenv("USER_AGENT", "website-lookup/1.0 (+https://example.org)")

    # headers
    HEADERS_JSON: Dict[str, str] = field(init=False)

    # constants
    JSON_ACCEPT: ClassVar[str] = "application/json"

    def __post_init__(self):
        self.HEADERS_JSON = {"Accept": self.JSON_ACCEPT, "User-Agent": self.USER_AGENT}
