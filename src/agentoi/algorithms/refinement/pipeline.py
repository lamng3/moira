"""Relation refinement pipeline."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .rules import RuleSet, Triple, load_rules

FactualCheck = Callable[[str, str, str], bool | None]


@dataclass(frozen=True)
class RefinementDiagnostic:
    stage: str
    code: str
    message: str
    triple: Triple | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RefinementOptions:
    rules_path: str | None = None
    transitive_closure: bool = True
    transitive_reduction: bool = True
    apply_case3: bool = True
    apply_case32: bool = True
    max_new_edges: int = 10_000
    reduction_size_guard: int = 150_000
    node_domain: Mapping[str, set[str]] | None = None
    factual_check: FactualCheck | None = None
    on_escalate: Callable[[str, str, str], None] | None = None
    llm: Any | None = None
    batch_size: int = 10


@dataclass
class RefinementResult:
    triples: list[Triple]
    diagnostics: list[RefinementDiagnostic] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def conflicts(self) -> list[RefinementDiagnostic]:
        return [
            item
            for item in self.diagnostics
            if item.code in {"domain_mismatch", "factual_denied"}
        ]


class RelationRefiner:
    """Normalize, infer, reduce, and validate ontology relations."""

    def __init__(
        self,
        rules: RuleSet | None = None,
        options: RefinementOptions | None = None,
    ) -> None:
        self.rules = rules
        self.options = options or RefinementOptions()

    def refine(
        self,
        triples: Iterable[Triple],
        options: RefinementOptions | None = None,
    ) -> RefinementResult:
        opts = options or self.options
        rules = self.rules or load_rules(opts.rules_path)
        diagnostics: list[RefinementDiagnostic] = []
        actions: list[str] = []
        stages = ["normalize"]

        raw = list(triples)
        normalized = rules.normalize_triples(raw)
        removed = len(raw) - len(normalized)
        if removed:
            actions.append(f"Removed {removed} invalid, dropped, or duplicate triples.")
            diagnostics.append(
                RefinementDiagnostic(
                    "normalize",
                    "triples_removed",
                    "Invalid, dropped, or duplicate triples were removed.",
                    details={"count": removed},
                )
            )

        refined = normalized
        if opts.transitive_closure:
            stages.append("transitive_closure")
            before = len(refined)
            refined, closure_diagnostics = self._transitive_closure(
                refined, rules, opts.max_new_edges
            )
            if added := len(refined) - before:
                actions.append(f"Added {added} transitive relations.")
            diagnostics.extend(closure_diagnostics)
        if opts.transitive_reduction:
            stages.append("transitive_reduction")
            before = len(refined)
            refined, reduction_diagnostics = self._transitive_reduction(
                refined, rules, opts.reduction_size_guard
            )
            if reduced := before - len(refined):
                actions.append(f"Removed {reduced} redundant relations.")
            diagnostics.extend(reduction_diagnostics)
        if opts.apply_case3:
            stages.append("validation")
            before = len(refined)
            refined, validation_diagnostics = self._validate(refined, rules, opts)
            if rejected := before - len(refined):
                actions.append(f"Rejected {rejected} invalid relations.")
            diagnostics.extend(validation_diagnostics)

        return RefinementResult(
            triples=refined,
            diagnostics=diagnostics,
            actions=actions,
            metadata={
                "input_count": len(raw),
                "output_count": len(refined),
                "stages": stages,
            },
        )

    @staticmethod
    def _transitive_closure(
        triples: list[Triple],
        rules: RuleSet,
        max_new_edges: int,
    ) -> tuple[list[Triple], list[RefinementDiagnostic]]:
        existing = set(triples)
        additions: list[Triple] = []
        by_predicate: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for subject, predicate, object_ in triples:
            if predicate in rules.transitive and subject != object_:
                by_predicate[predicate][subject].add(object_)

        if max_new_edges <= 0:
            return triples, [
                RefinementDiagnostic(
                    "transitive_closure",
                    "closure_truncated",
                    "Transitive closure stopped at the configured edge limit.",
                    details={"max_new_edges": max_new_edges},
                )
            ]

        truncated = False
        for predicate in sorted(by_predicate):
            adjacency = by_predicate[predicate]
            for source in sorted(adjacency):
                reached = set(adjacency[source])
                queue = deque(sorted(reached))
                while queue:
                    middle = queue.popleft()
                    for target in sorted(adjacency.get(middle, ())):
                        if target == source or target in reached:
                            continue
                        reached.add(target)
                        queue.append(target)
                        triple = (source, predicate, target)
                        if triple not in existing:
                            existing.add(triple)
                            additions.append(triple)
                            if len(additions) >= max_new_edges:
                                truncated = True
                                break
                    if truncated:
                        break
                if truncated:
                    break
            if truncated:
                break

        diagnostics = (
            [
                RefinementDiagnostic(
                    "transitive_closure",
                    "closure_truncated",
                    "Transitive closure stopped at the configured edge limit.",
                    details={"max_new_edges": max_new_edges},
                )
            ]
            if truncated
            else []
        )
        return triples + additions, diagnostics

    @staticmethod
    def _transitive_reduction(
        triples: list[Triple],
        rules: RuleSet,
        size_guard: int,
    ) -> tuple[list[Triple], list[RefinementDiagnostic]]:
        if len(triples) > size_guard:
            return triples, [
                RefinementDiagnostic(
                    "transitive_reduction",
                    "size_guard",
                    "Transitive reduction was skipped because the graph is too large.",
                    details={"triples": len(triples), "size_guard": size_guard},
                )
            ]

        by_predicate: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for subject, predicate, object_ in triples:
            if predicate in rules.transitive and subject != object_:
                by_predicate[predicate].add((subject, object_))

        remove: set[Triple] = set()
        diagnostics: list[RefinementDiagnostic] = []
        for predicate in sorted(by_predicate):
            edges = by_predicate[predicate]
            adjacency: dict[str, set[str]] = defaultdict(set)
            indegree: dict[str, int] = defaultdict(int)
            nodes: set[str] = set()
            for source, target in edges:
                adjacency[source].add(target)
                indegree[target] += 1
                nodes.update((source, target))

            queue = deque(sorted(node for node in nodes if indegree[node] == 0))
            topological: list[str] = []
            while queue:
                node = queue.popleft()
                topological.append(node)
                for target in sorted(adjacency.get(node, ())):
                    indegree[target] -= 1
                    if indegree[target] == 0:
                        queue.append(target)
            if len(topological) != len(nodes):
                diagnostics.append(
                    RefinementDiagnostic(
                        "transitive_reduction",
                        "cycle_detected",
                        f"Reduction was skipped for cyclic predicate {predicate!r}.",
                        details={"predicate": predicate},
                    )
                )
                continue

            reachable: dict[str, set[str]] = {}
            for source in reversed(topological):
                descendants: set[str] = set()
                for target in adjacency.get(source, ()):
                    descendants.add(target)
                    descendants.update(reachable.get(target, ()))
                reachable[source] = descendants

            for source, target in edges:
                if any(
                    neighbor != target and target in reachable.get(neighbor, ())
                    for neighbor in adjacency[source]
                ):
                    remove.add((source, predicate, target))

        return [triple for triple in triples if triple not in remove], diagnostics

    def _validate(
        self,
        triples: list[Triple],
        rules: RuleSet,
        options: RefinementOptions,
    ) -> tuple[list[Triple], list[RefinementDiagnostic]]:
        output: list[Triple] = []
        pending: list[Triple] = []
        diagnostics: list[RefinementDiagnostic] = []

        for triple in triples:
            subject, predicate, object_ = triple
            expected = rules.domain_range.get(predicate)
            if expected and options.node_domain:
                domain, range_ = expected
                actual_domain = options.node_domain.get(subject)
                actual_range = options.node_domain.get(object_)
                mismatch = (
                    actual_domain is not None
                    and domain
                    and actual_domain.isdisjoint(domain)
                ) or (
                    actual_range is not None
                    and range_
                    and actual_range.isdisjoint(range_)
                )
                if mismatch:
                    diagnostics.append(
                        RefinementDiagnostic(
                            "validation",
                            "domain_mismatch",
                            "Relation does not satisfy its domain/range rule.",
                            triple,
                        )
                    )
                    continue

            reversible = (
                predicate in {"subclassof", "superclassof"}
                or predicate in rules.inverse
                or predicate in rules.domain_range
                or predicate in rules.grammar_flip
            )
            output.append(triple)
            if not reversible and options.apply_case32:
                pending.append(triple)
                diagnostics.append(
                    RefinementDiagnostic(
                        "validation",
                        "non_reversible",
                        "Non-reversible relation queued for factual checking.",
                        triple,
                    )
                )

        denied: set[Triple] = set()
        for batch_start in range(0, len(pending), max(1, options.batch_size)):
            batch = pending[batch_start : batch_start + max(1, options.batch_size)]
            verdicts = self._fact_check(batch, options)
            for triple, verdict in zip(batch, verdicts):
                if verdict is False:
                    denied.add(triple)
                    diagnostics.append(
                        RefinementDiagnostic(
                            "validation",
                            "factual_denied",
                            "Factual check denied the relation.",
                            triple,
                        )
                    )
                    if options.on_escalate:
                        try:
                            options.on_escalate(*triple)
                        except Exception as error:  # noqa: BLE001
                            diagnostics.append(
                                RefinementDiagnostic(
                                    "validation",
                                    "escalation_failed",
                                    "Escalation callback failed.",
                                    triple,
                                    {"error": type(error).__name__},
                                )
                            )
        return [triple for triple in output if triple not in denied], diagnostics

    @staticmethod
    def _fact_check(
        batch: list[Triple], options: RefinementOptions
    ) -> list[bool | None]:
        if options.llm is not None:
            prompt = (
                "Return one line per triple: TRUE, FALSE, or UNKNOWN.\n"
                + "\n".join(
                    f"{index}. {subject} | {predicate} | {object_}"
                    for index, (subject, predicate, object_) in enumerate(batch, 1)
                )
            )
            try:
                response = options.llm.invoke(prompt)
                text = getattr(response, "content", None) or str(response)
                tokens = re.findall(r"\b(true|false|unknown)\b", text, re.IGNORECASE)
                return [
                    (
                        tokens[index].lower() == "true"
                        if index < len(tokens) and tokens[index].lower() != "unknown"
                        else None
                    )
                    for index in range(len(batch))
                ]
            except Exception:  # noqa: BLE001
                return [None] * len(batch)
        if options.factual_check:
            verdicts: list[bool | None] = []
            for triple in batch:
                try:
                    verdicts.append(options.factual_check(*triple))
                except Exception:  # noqa: BLE001
                    verdicts.append(None)
            return verdicts
        return [None] * len(batch)


def refine_relations(
    triples: Iterable[Triple],
    *,
    rules_json: str | None = None,
    transitive_closure: bool = True,
    apply_case3: bool = True,
    apply_case32: bool = True,
    node_domain: Mapping[str, set[str]] | None = None,
    llm: Any | None = None,
) -> list[Triple]:
    """Compatibility wrapper returning only refined triples."""
    options = RefinementOptions(
        rules_path=rules_json,
        transitive_closure=transitive_closure,
        apply_case3=apply_case3,
        apply_case32=apply_case32,
        node_domain=node_domain,
        llm=llm,
    )
    return RelationRefiner(options=options).refine(triples).triples
