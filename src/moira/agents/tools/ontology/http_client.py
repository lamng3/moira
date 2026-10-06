from __future__ import annotations

from typing import Any, Dict, Optional

from moira.agents.tools.common.http import SessionHttp
from moira.agents.tools.ontology.config import Config

class Http:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = SessionHttp(default_timeout=cfg.DEFAULT_TIMEOUT)

    def get(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Any]:
        return self._client.get(url, params=params, headers=headers)

    def close(self) -> None:
        self._client.close()