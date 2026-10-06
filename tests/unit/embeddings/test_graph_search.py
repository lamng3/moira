import pytest
import torch

from moira.algorithms.graph import (
    Concept,
    ConceptGraph,
    ConceptHypergraph,
    EquivalentClass,
)
from moira.embeddings import (
    MultiQuerySearchService,
    TorchExactVectorIndex,
    register_vector_index,
    unregister_vector_index,
)


class FakeTextEncoder:
    embed_dim = 2

    def __init__(self, vectors):
        self.vectors = vectors

    def to_embedding(self, text, max_length=512):
        return self.vectors[text].clone()


def equivalent_class(name, vector, *iris):
    concepts = [
        Concept(name=f"{name}-{position}", iri=iri)
        for position, iri in enumerate(iris or (f"https://all/{name}",))
    ]
    node = EquivalentClass(concepts)
    node.id = name
    node.embedding = torch.tensor(vector, dtype=torch.float32)
    return node


@pytest.fixture(params=["graph", "hypergraph"])
def graphlike(request):
    nodes = [
        equivalent_class("east", [1.0, 0.0], "https://a/east", "https://b/east"),
        equivalent_class("north", [0.0, 1.0], "https://b/north"),
        equivalent_class("diagonal", [1.0, 1.0], "https://a/diagonal"),
        equivalent_class("west", [-1.0, 0.0], "https://a/west"),
    ]
    if request.param == "graph":
        return ConceptGraph(nodes, [])
    return ConceptHypergraph(nodes, [])


@pytest.mark.parametrize(
    ("agg", "expected"),
    [
        ("max", ["east", "north", "diagonal", "west"]),
        ("mean", ["diagonal", "east", "north", "west"]),
        ("sum", ["diagonal", "east", "north", "west"]),
        ("weighted", ["east", "diagonal", "north", "west"]),
    ],
)
def test_graph_search_aggregations_are_exact_and_ordered(graphlike, agg, expected):
    encoder = FakeTextEncoder(
        {
            "horizontal": torch.tensor([1.0, 0.0]),
            "vertical": torch.tensor([0.2, 1.0]),
        }
    )
    results = graphlike.nearest_nodes(
        ["horizontal", "vertical"],
        k=4,
        text_model=encoder,
        agg=agg,
        weights=[3.0, 1.0],
    )

    assert [node.id for node, _ in results] == expected


def test_prebuilt_index_is_queried_without_mutation(graphlike):
    nodes = list(graphlike.nodes.values())
    index = TorchExactVectorIndex(dimension=2)
    MultiQuerySearchService.index_nodes(index, nodes)
    index.add(["unrelated"], torch.tensor([[1.0, 0.0]]))
    before = len(index)

    results = graphlike.nearest_nodes(
        "horizontal",
        k=2,
        text_model=FakeTextEncoder(
            {"horizontal": torch.tensor([1.0, 0.0])}
        ),
        index=index,
    )

    assert len(index) == before
    assert [node.id for node, _ in results] == ["east", "diagonal"]


def test_graph_search_selects_registered_backend(graphlike):
    created = []

    def factory(**options):
        created.append(options)
        return TorchExactVectorIndex(**options)

    register_vector_index("test-index", factory)
    try:
        results = graphlike.nearest_nodes(
            "horizontal",
            k=1,
            text_model=FakeTextEncoder(
                {"horizontal": torch.tensor([1.0, 0.0])}
            ),
            index_backend="test-index",
        )
    finally:
        unregister_vector_index("test-index")

    assert created == [{"dimension": 2}]
    assert results[0][0].id == "east"


def test_graph_search_filters_and_tensor_output(graphlike):
    encoder = FakeTextEncoder({"horizontal": torch.tensor([1.0, 0.0])})

    inclusive = graphlike.nearest_nodes(
        "horizontal",
        k=4,
        text_model=encoder,
        iri_prefix="https://a/",
    )
    assert [node.id for node, _ in inclusive] == ["east", "diagonal", "west"]

    exclusive = graphlike.nearest_nodes(
        "horizontal",
        k=4,
        text_model=encoder,
        iri_prefix="https://a/",
        exclusive=True,
        return_out=False,
    )
    values, positions = exclusive
    # Indices retain the original nearest_nodes meaning: positions within the
    # filtered candidate list, not within all graph nodes.
    assert positions.tolist() == [0, 1]
    assert values.tolist() == pytest.approx([2**-0.5, -1.0])


def test_hypergraph_prompt_matches_cli_contract():
    node = equivalent_class("east", [1.0, 0.0])
    graph = ConceptHypergraph([node], [])

    # The CLI uses the default model; replace the wrapper call only to keep this
    # contract test deterministic and independent of model downloads.
    graph.nearest_nodes = lambda terms, k=10: [(node, 1.0)]
    prompt = graph.make_prompt_for_query("Where?", ["east"])

    assert "QUERY: Where?" in prompt
    assert "east-0" in prompt
