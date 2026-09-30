"""The memory view keeps hit counts and leaves answer text out of the chat outline."""

import json

from agentoi.memory.view import memory_snapshot


def test_snapshot_shows_hits_and_skips_answer_text() -> None:
    view = memory_snapshot(
        hot=[
            {"question": "heart", "frequency": 1},
            {"question": "liver", "frequency": 4},
        ],
        concepts={
            "concept_id": None,
            "frequency": 2,
            "children": {
                "http://example.org/Heart": {
                    "concept_id": "http://example.org/Heart",
                    "frequency": 3,
                    "children": {},
                }
            },
        },
        long_term=[
            {"concept_ids": ["http://example.org/Heart"], "frequency": 5, "answer": "secret"}
        ],
        chat={
            "kind": "root",
            "children": [
                {
                    "kind": "question",
                    "text": "Where is the heart?",
                    "children": [
                        {"kind": "route", "text": "retrieve"},
                        {"kind": "answer", "text": "A very long answer that stays in the transcript."},
                    ],
                }
            ],
        },
    )

    assert view["hot"][0] == {"label": "liver", "hits": 4}
    assert view["concepts"]["children"][0]["label"] == "Heart"
    assert view["concepts"]["children"][0]["hits"] == 3
    assert view["long_term"][0]["label"] == "Heart"
    assert view["long_term"][0]["hits"] == 5
    assert view["chat"]["children"][0]["label"] == "Where is the heart? · retrieve"
    assert "very long answer" not in json.dumps(view)


def test_the_trie_uses_ontology_labels_for_equivalence_ids() -> None:
    concept_id = "eq_Nf956502f12724437e22bc380e0cfd99e"
    view = memory_snapshot(
        hot=[],
        concepts={
            "concept_id": None,
            "frequency": 1,
            "children": {
                concept_id: {
                    "concept_id": concept_id,
                    "frequency": 4,
                    "children": {},
                }
            },
        },
        long_term=[{"concept_ids": [concept_id], "frequency": 4}],
        names=lambda concept: "visceral organ system" if concept == concept_id else "",
    )

    assert view["concepts"]["children"][0]["label"] == "visceral organ system"
    assert view["long_term"][0]["label"] == "visceral organ system"
    assert concept_id not in json.dumps(view)
