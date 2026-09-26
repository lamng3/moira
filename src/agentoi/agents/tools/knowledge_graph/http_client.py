from __future__ import annotations
from agentoi.agents.tools.common.http import SessionHttp
from agentoi.agents.tools.knowledge_graph.config import Config

class Http(SessionHttp):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        super().__init__(
            default_timeout=cfg.DEFAULT_TIMEOUT,
            headers=cfg.HEADERS_JSON,
        )