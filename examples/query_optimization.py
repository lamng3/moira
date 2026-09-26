"""Select high-value questions from a thought graph."""

from agentoi.agents.ontology_of_thought import Thought, ThoughtGraph
from agentoi.agents.ontology_of_thought.optimizer import QueryOptimizer


def main() -> None:
    graph = ThoughtGraph()
    root = graph.add_thought(Thought.prompt("Are these concepts equivalent?"))
    lexical = graph.add_thought(Thought.reasoning("Compare labels.", reasoning_depth=1))
    structural = graph.add_thought(Thought.reasoning("Compare neighbors.", reasoning_depth=1))
    graph.connect(root, lexical, "flow")
    graph.connect(root, structural, "flow")

    optimizer = QueryOptimizer(graph, min_cluster_size=1)
    queries = optimizer.select_queries(budget=2)
    answers = {query: index == 0 for index, query in enumerate(queries)}
    candidates = optimizer.update_candidate_set(queries, answers)

    print("Queries:", queries)
    print("Remaining candidates:", candidates)


if __name__ == "__main__":
    main()
