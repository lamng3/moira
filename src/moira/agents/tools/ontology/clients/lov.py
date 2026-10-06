from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import requests
from moira.agents.tools.ontology.http_client import Http
from moira.agents.tools.ontology.config import Config

class LOVClient:
    def __init__(self, http: Http, cfg: Config):
        self.http, self.cfg = http, cfg

    def ontology_info(self, vocab: str) -> Optional[Dict[str, Any]]:
        info = self.http.get(
            f"{self.cfg.LOV_API}/vocabulary/info", 
            params={"vocab": vocab}, 
            headers=self.cfg.HEADERS_JSON
        )
        if not info or not isinstance(info, dict):
            return None
        return {
            "source": "lov",
            "acronym": info.get("prefix") or vocab,
            "title": info.get("nsp") or info.get("title") or info.get("prefix"),
            "description": info.get("description"),
            "homepage": info.get("homepage"),
            "version": info.get("version"),
            "license": info.get("license"),
            "created": info.get("issued"),
            "updated": info.get("modified"),
            "metrics": {},
            "raw": info,
        }
