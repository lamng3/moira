import json

from moira.algorithms.graph import (
    Concept,
    ConceptGraph,
    EquivalentClass,
    EquivalentClassRelation,
    Ontology,
)
from moira.algorithms.refinement import (
    DeferredReason,
    GraphUpdate,
    OfflineGraphRefiner,
    OnlineOntologyRefiner,
    OntologyRefiner,
    RefinementOptions,
    RelationRefiner,
    load_rules,
)


def test_rule_loading_merges_all_fields_and_normalizes(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(
        json.dumps(
            {
                "normalize": {"contains": "has_part"},
                "inverse": {"has_part": "part_of"},
                "transitive": ["contains"],
                "drop": ["ignore_me"],
                "grammar": {"mentors": "is_mentored_by"},
                "grammar_flip": ["is_mentored_by"],
                "domain_range": {
                    "teaches": {"domain": ["Person"], "range": ["Course"]}
                },
            }
        )
    )

    rules = load_rules(path)

    assert rules.normalize_triple(("whole", "contains", "piece")) == (
        "piece",
        "part_of",
        "whole",
    )
    assert rules.normalize_triple(("teacher", "mentors", "student")) == (
        "student",
        "is_mentored_by",
        "teacher",
    )
    assert rules.normalize_triple(("a", "ignore_me", "b")) is None
    assert rules.domain_range["teaches"] == ({"Person"}, {"Course"})
    assert "subclassof" in rules.transitive
    assert "part_of" in rules.transitive


def test_pipeline_deduplicates_and_computes_transitive_reduction():
    result = RelationRefiner().refine(
        [
            ("A", "is_a", "B"),
            ("A", "is_a", "B"),
            ("B", "is_a", "C"),
            ("A", "is_a", "C"),
        ],
        RefinementOptions(apply_case3=False),
    )

    assert result.triples == [
        ("A", "subclassof", "B"),
        ("B", "subclassof", "C"),
    ]
    assert result.diagnostics[0].code == "triples_removed"
    assert result.metadata["stages"] == [
        "normalize",
        "transitive_closure",
        "transitive_reduction",
    ]
    assert result.actions


def test_pipeline_reports_cycles_and_domain_conflicts_structurally():
    refiner = RelationRefiner()
    cycle = refiner.refine(
        [("A", "is_a", "B"), ("B", "is_a", "A")],
        RefinementOptions(apply_case3=False),
    )
    assert {item.code for item in cycle.diagnostics} == {"cycle_detected"}
    assert len(cycle.triples) == 2

    rules = load_rules()
    rules.domain_range["teaches"] = ({"Person"}, {"Course"})
    mismatch = RelationRefiner(rules).refine(
        [("rock", "teaches", "math")],
        RefinementOptions(
            transitive_closure=False,
            node_domain={"rock": {"Mineral"}, "math": {"Course"}},
        ),
    )
    assert mismatch.triples == []
    assert mismatch.conflicts[0].code == "domain_mismatch"
    assert mismatch.conflicts[0].triple == ("rock", "teaches", "math")


def test_pipeline_bounds_closure_and_reports_factual_denials():
    bounded = RelationRefiner().refine(
        [
            ("A", "is_a", "B"),
            ("B", "is_a", "C"),
            ("C", "is_a", "D"),
        ],
        RefinementOptions(
            apply_case3=False,
            transitive_reduction=False,
            max_new_edges=1,
        ),
    )
    assert len(bounded.triples) == 4
    assert any(item.code == "closure_truncated" for item in bounded.diagnostics)

    denied = RelationRefiner().refine(
        [("A", "unverified_relation", "B")],
        RefinementOptions(
            transitive_closure=False,
            factual_check=lambda *_: False,
        ),
    )
    assert denied.triples == []
    assert {item.code for item in denied.conflicts} == {"factual_denied"}
    assert any(item.code == "non_reversible" for item in denied.diagnostics)

    unchecked = RelationRefiner().refine(
        [("A", "unverified_relation", "B")],
        RefinementOptions(
            transitive_closure=False,
            apply_case32=False,
        ),
    )
    assert unchecked.triples == [("A", "unverified_relation", "B")]


def test_ontology_refiner_preserves_identity_and_ground_set():
    ontology = Ontology(name="source", version="2").build_ontology_from_triples(
        [("A", "is_a", "B")]
    )
    ontology.get_node("A").ground_set["labels"] = ["Alpha"]

    refined = OntologyRefiner().refine(
        ontology,
        RefinementOptions(transitive_closure=False),
    )

    assert refined.name == "source"
    assert refined.version == "2"
    assert refined.get_node("A").ground_set["labels"] == ["Alpha"]
    assert refined.edges[0].pred == "subclassof"


def _node(name):
    return EquivalentClass([Concept(name, nodeid=name)])


def test_graph_collapse_uses_stable_representative_and_cleans_edges():
    a, b, c = _node("A"), _node("B"), _node("C")
    graph = ConceptGraph(
        [a, b, c],
        [
            EquivalentClassRelation(a, b, "no", 2.0),
            EquivalentClassRelation(a, c, "no", 1.0),
            EquivalentClassRelation(b, c, "no", 2.0),
        ],
    )

    representative = OfflineGraphRefiner.collapse(graph, {"B", "A"})

    assert representative == "A"
    assert set(graph.nodes) == {"A", "C"}
    assert [(edge.src.id, edge.tgt.id, edge.score) for edge in graph.edges] == [
        ("A", "C", 2.0)
    ]
    assert [node.id for node in graph.children["A"]] == ["C"]
    assert [node.id for node in graph.parents["C"]] == ["A"]


def test_offline_refiner_collapses_only_semantic_bisimulation_blocks():
    parent, left, right = _node("parent"), _node("left"), _node("right")
    graph = ConceptGraph(
        [parent, left, right],
        [
            EquivalentClassRelation(parent, left, "no", 2.0),
            EquivalentClassRelation(parent, right, "no", 2.0),
        ],
    )
    refiner = OfflineGraphRefiner(
        concept_similar=lambda first, second: {
            first.id,
            second.id,
        }
        == {"left", "right"}
    )

    refined = refiner.refine(graph)

    assert set(refined.nodes) == {"left", "parent"}
    assert {
        concept.name for concept in refined.nodes["left"].equiv_concepts
    } == {"left", "right"}
    assert set(graph.nodes) == {"parent", "left", "right"}


def test_offline_refiner_splits_semantic_peers_with_different_structure():
    first_parent, second_parent = _node("p1"), _node("p2")
    left, right = _node("left"), _node("right")
    graph = ConceptGraph(
        [first_parent, second_parent, left, right],
        [
            EquivalentClassRelation(first_parent, left, "no", 2.0),
            EquivalentClassRelation(second_parent, right, "no", 2.0),
        ],
    )
    refined = OfflineGraphRefiner(
        concept_similar=lambda first, second: {
            first.id,
            second.id,
        }
        == {"left", "right"}
    ).refine(graph)

    assert set(refined.nodes) == {"p1", "p2", "left", "right"}


def test_online_refiner_applies_streamed_updates_and_preserves_input():
    root, leaf, new_leaf = _node("root"), _node("leaf"), _node("new")
    graph = ConceptGraph(
        [root, leaf],
        [EquivalentClassRelation(root, leaf, "no", 2.0)],
    )

    result = OnlineOntologyRefiner().refine(
        graph,
        [GraphUpdate(root, new_leaf, confidence=9.0)],
    )

    assert "new" not in graph.nodes
    assert set(result.graph.nodes) == {"root", "leaf", "new"}
    assert [(edge.src.id, edge.tgt.id) for edge in result.graph.edges] == [
        ("root", "leaf"),
        ("root", "new"),
    ]
    assert result.graph.nodes["root"].rank == 1
    assert result.graph.nodes["new"].rank == 0


def test_online_refiner_defers_low_confidence_rank_conflicts_and_cycles():
    root, middle, leaf, unknown = (
        _node("root"),
        _node("middle"),
        _node("leaf"),
        _node("unknown"),
    )
    graph = ConceptGraph(
        [root, middle, leaf],
        [
            EquivalentClassRelation(root, middle, "no", 2.0),
            EquivalentClassRelation(middle, leaf, "no", 2.0),
        ],
    )
    result = OnlineOntologyRefiner().refine(
        graph,
        [
            GraphUpdate(unknown, leaf, confidence=5.0),
            GraphUpdate(leaf, root, confidence=9.0),
            GraphUpdate(leaf, middle, confidence=9.0),
        ],
    )

    assert result.applied == []
    assert [item.reason for item in result.deferred].count(
        DeferredReason.LOW_CONFIDENCE
    ) == 1
    assert [item.reason for item in result.deferred].count(
        DeferredReason.CYCLE
    ) == 2
    assert "unknown" not in result.graph.nodes


def test_online_refiner_merges_semantically_and_structurally_equivalent_nodes():
    root, left, right = (
        _node("root"),
        _node("left"),
        _node("right"),
    )
    graph = ConceptGraph(
        [root, left],
        [EquivalentClassRelation(root, left, "no", 2.0)],
    )
    result = OnlineOntologyRefiner(
        concept_similar=lambda first, second: {
            first.id,
            second.id,
        }
        == {"left", "right"}
    ).refine(graph, [GraphUpdate(root, right, confidence=9.0)])

    assert ("left", "right") in result.merged_classes
    assert "right" not in result.graph.nodes


def test_online_refiner_splits_non_similar_members_and_defers_review():
    left_concept = Concept("left", nodeid="left")
    right_concept = Concept("right", nodeid="right")
    merged = EquivalentClass([left_concept, right_concept])
    merged.id = "left"
    graph = ConceptGraph([merged], [])

    result = OnlineOntologyRefiner().refine(
        graph,
        [
            GraphUpdate(
                EquivalentClass([Concept("left", nodeid="left")]),
                EquivalentClass([Concept("right", nodeid="right")]),
                confidence=9.0,
                semantically_similar=False,
            )
        ],
    )

    assert set(result.graph.nodes) == {"left", "right"}
    assert result.split_classes == [("left", "right")]
    assert result.deferred[0].reason is DeferredReason.SEMANTIC_CONFLICT
