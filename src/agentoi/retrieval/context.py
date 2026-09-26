"""Context selection between retrieval and prompt generation."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Protocol

from agentoi.algorithms.graph import ConceptGraph, EquivalentClass

from .neighborhood import sample_neighborhood

ONTOLOGY_RELEVANCE_GUIDANCE = (
    "Prefer concepts and relations that answer the query. "
    "Keep hierarchy needed to interpret a selected concept. "
    "Reject matches that are only topically related."
)
_MAX_DOCUMENTS = 256


@dataclass(frozen=True, slots=True)
class ContextCandidate:
    """One retrieved concept, neighbor, or web passage."""

    id: str
    text: str
    source: str
    score: float | None = None


@dataclass(frozen=True, slots=True)
class ContextDecision:
    """A harness judgment for one candidate."""

    id: str
    relevance: float | None
    selected: bool
    returned: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ContextSelection:
    """Documents returned to the prompt and the proposal behind them."""

    status: str
    documents: tuple[ContextCandidate, ...]
    proposed_ids: tuple[str, ...]
    decisions: tuple[ContextDecision, ...]
    usage_tokens: int | None = None
    error: str | None = None

    def metrics(self) -> dict[str, float | int | str | bool | None]:
        """Return compact comparison metrics for experiment reports."""
        total = len(self.decisions)
        selected = sum(decision.selected for decision in self.decisions)
        return {
            "status": self.status,
            "candidate_count": total,
            "selected_count": selected,
            "returned_count": len(self.documents),
            "rejection_rate": (total - selected) / total if total else 0.0,
            "usage_tokens": self.usage_tokens,
            "fail_open": self.status == "bypassed",
        }


class ContextHarness(Protocol):
    """Select which retrieved candidates become prompt context."""

    expands_context: bool

    def select(
        self,
        query: str,
        candidates: Sequence[ContextCandidate],
    ) -> ContextSelection: ...


class PassthroughHarness:
    """Return the retrieved shortlist without an external decision model."""

    expands_context = False

    def select(
        self,
        query: str,
        candidates: Sequence[ContextCandidate],
    ) -> ContextSelection:
        del query
        documents = tuple(candidates)
        return ContextSelection(
            status="passthrough",
            documents=documents,
            proposed_ids=tuple(candidate.id for candidate in documents),
            decisions=tuple(
                ContextDecision(
                    id=candidate.id,
                    relevance=candidate.score,
                    selected=True,
                    returned=True,
                    reason="retained",
                )
                for candidate in documents
            ),
        )


class JevContextHarness:
    """Score a retrieved shortlist with Jev before prompt generation.

    Shadow mode records the proposed selection while returning the original
    shortlist. Missing credentials, import errors, and provider failures return
    that same shortlist with ``status="bypassed"``.
    """

    expands_context = True

    def __init__(
        self,
        *,
        selector: Callable[..., object] | None = None,
        shadow: bool = True,
        min_relevance: float = 0.2,
        max_context_tokens: int = 3000,
        relevance_guidance: str = ONTOLOGY_RELEVANCE_GUIDANCE,
    ) -> None:
        if not 0.0 <= min_relevance <= 1.0:
            raise ValueError("min_relevance must be between 0 and 1")
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        self._selector = selector
        self.shadow = shadow
        self.min_relevance = min_relevance
        self.max_context_tokens = max_context_tokens
        self.relevance_guidance = relevance_guidance

    def select(
        self,
        query: str,
        candidates: Sequence[ContextCandidate],
    ) -> ContextSelection:
        documents = tuple(candidates)
        if not documents:
            return PassthroughHarness().select(query, documents)
        if not query.strip():
            return _bypassed(documents, "query must not be empty")
        try:
            result = self._invoke(query, documents)
        except Exception as exc:
            return _bypassed(documents, str(exc))
        return _selection_from_result(documents, result)

    def _invoke(self, query: str, candidates: Sequence[ContextCandidate]) -> object:
        options = {
            "scoring_strategy": "contextual",
            "mode": "filter_and_rerank",
            "min_relevance": self.min_relevance,
            "max_context_tokens": self.max_context_tokens,
            "shadow": self.shadow,
            "relevance_guidance": self.relevance_guidance,
        }
        if self._selector is not None:
            return self._selector(query, candidates, **options)
        return asyncio.run(self._invoke_sdk(query, candidates, options))

    async def _invoke_sdk(
        self,
        query: str,
        candidates: Sequence[ContextCandidate],
        options: Mapping[str, object],
    ) -> object:
        _export_api_key()
        try:
            from rag_jev import ContextSelector, Document, Jev
        except ImportError as exc:
            raise RuntimeError(
                "Jev context selection requires the 'jev' extra: uv sync --extra jev"
            ) from exc

        async with Jev() as provider:
            selector = ContextSelector(provider)
            return await selector.select(
                query=query,
                documents=[
                    Document(id=candidate.id, text=candidate.text)
                    for candidate in candidates
                ],
                **options,
            )


def create_context_harness(name: str | None = None) -> ContextHarness:
    """Create the configured harness. The default preserves current prompts."""
    selected = (name or os.environ.get("AGENTOI_CONTEXT_HARNESS", "passthrough")).strip()
    normalized = selected.lower()
    if normalized == "passthrough":
        return PassthroughHarness()
    if normalized == "jev":
        live = os.environ.get("AGENTOI_CONTEXT_MODE", "shadow").strip().lower() == "live"
        return JevContextHarness(shadow=not live)
    raise ValueError("Context harness must be 'passthrough' or 'jev'.")


def build_context_candidates(
    graph: ConceptGraph,
    query: str,
    *,
    top_k: int = 10,
    include_neighborhood: bool = False,
    max_hops: int = 1,
    web_passages: Sequence[Mapping[str, object]] | None = None,
) -> list[ContextCandidate]:
    """Collect concept matches, optional neighbors, and supplied web passages."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    candidates: list[ContextCandidate] = []
    seen: set[str] = set()
    matches = graph.nearest_nodes([query], k=top_k)
    for node, score in matches:
        _append_candidate(
            candidates,
            seen,
            node.id,
            _node_text(node),
            "concept",
            float(score),
        )
        if include_neighborhood:
            for neighbor in sample_neighborhood(graph, node, max_hops=max_hops):
                _append_candidate(
                    candidates,
                    seen,
                    neighbor.id,
                    _node_text(neighbor),
                    "neighborhood",
                    None,
                )
    for index, passage in enumerate(web_passages or (), start=1):
        text = str(passage.get("text") or passage.get("snippet") or "").strip()
        identifier = str(passage.get("id") or f"web-{index}")
        _append_candidate(candidates, seen, identifier, text, "web", None)
    return candidates[:_MAX_DOCUMENTS]


def _append_candidate(
    candidates: list[ContextCandidate],
    seen: set[str],
    identifier: str,
    text: str,
    source: str,
    score: float | None,
) -> None:
    identifier = identifier.strip()
    text = text.strip()
    if not identifier or not text or identifier in seen or len(candidates) >= _MAX_DOCUMENTS:
        return
    seen.add(identifier)
    candidates.append(
        ContextCandidate(id=identifier, text=text, source=source, score=score)
    )


def _node_text(node: EquivalentClass) -> str:
    return str(node.members(return_label=True))


def _export_api_key() -> None:
    if os.environ.get("TYPESAFE_API_KEY"):
        return
    key = os.environ.get("JEV_API_KEY", "").strip()
    if not key:
        raise RuntimeError("JEV_API_KEY or TYPESAFE_API_KEY is required")
    os.environ["TYPESAFE_API_KEY"] = key


def _bypassed(
    candidates: Sequence[ContextCandidate], error: str
) -> ContextSelection:
    documents = tuple(candidates)
    return ContextSelection(
        status="bypassed",
        documents=documents,
        proposed_ids=tuple(candidate.id for candidate in documents),
        decisions=tuple(
            ContextDecision(
                id=candidate.id,
                relevance=candidate.score,
                selected=True,
                returned=True,
                reason="bypassed",
            )
            for candidate in documents
        ),
        error=error,
    )


def _selection_from_result(
    candidates: Sequence[ContextCandidate],
    result: object,
) -> ContextSelection:
    payload = _mapping(result)
    by_id = {candidate.id: candidate for candidate in candidates}
    status = str(payload.get("status") or "applied")
    decisions = tuple(_decision(item) for item in payload.get("decisions", []))
    proposed = payload.get("selected_ids")
    proposed_ids = (
        tuple(str(item) for item in proposed)
        if isinstance(proposed, Sequence) and not isinstance(proposed, str)
        else tuple(decision.id for decision in decisions if decision.selected)
    )
    returned_ids = _returned_ids(payload, decisions, candidates, status)
    documents = tuple(by_id[identifier] for identifier in returned_ids if identifier in by_id)
    if not documents and status == "bypassed":
        documents = tuple(candidates)
    return ContextSelection(
        status=status,
        documents=documents,
        proposed_ids=proposed_ids,
        decisions=decisions,
        usage_tokens=_usage_tokens(payload.get("usage")),
        error=_optional_text(payload.get("error_code")),
    )


def _returned_ids(
    payload: Mapping[str, object],
    decisions: Sequence[ContextDecision],
    candidates: Sequence[ContextCandidate],
    status: str,
) -> list[str]:
    if status in {"shadow", "bypassed"}:
        return [candidate.id for candidate in candidates]
    raw_documents = payload.get("documents", [])
    if isinstance(raw_documents, Sequence) and not isinstance(raw_documents, str):
        identifiers = []
        for item in raw_documents:
            item_payload = _mapping(item)
            identifier = item_payload.get("id")
            if identifier is not None:
                identifiers.append(str(identifier))
        if identifiers:
            return identifiers
    returned = [decision.id for decision in decisions if decision.returned]
    return returned or [candidate.id for candidate in candidates]


def _decision(value: object) -> ContextDecision:
    payload = _mapping(value)
    relevance = payload.get("probability", payload.get("relevance"))
    return ContextDecision(
        id=str(payload.get("id", "")),
        relevance=float(relevance) if isinstance(relevance, (int, float)) else None,
        selected=bool(payload.get("selected", False)),
        returned=bool(payload.get("returned", False)),
        reason=str(payload.get("reason") or "retained"),
    )


def _usage_tokens(value: object) -> int | None:
    payload = _mapping(value)
    for key in ("total_tokens", "tokens", "observed_tokens"):
        item = payload.get(key)
        if isinstance(item, int) and not isinstance(item, bool):
            return item
    return None


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        dumped = dump()
        if isinstance(dumped, Mapping):
            return dumped
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return {
        name: getattr(value, name)
        for name in dir(value)
        if not name.startswith("_") and not callable(getattr(value, name))
    }


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
