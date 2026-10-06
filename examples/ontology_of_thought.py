"""Build an Ontology-of-Thought trace."""

from moira.agents.ontology_of_thought import ConversationTrace, ThoughtGraph


def main() -> None:
    conversation = ConversationTrace()
    conversation.add_prompt("What is myocardium?")
    conversation.add_reasoning("Search an anatomy ontology.")
    conversation.add_tool_call(
        "ontology_term_info",
        {"term": "myocardium", "ontologies": ["UBERON"]},
    )
    conversation.add_tool_result(
        "ontology_term_info",
        {"label": "myocardium", "definition": "Cardiac muscle tissue."},
    )
    conversation.add_response("Myocardium is the muscle tissue of the heart.")

    trace = conversation.get_trace()
    graph = ThoughtGraph.from_memory_trace(trace)

    print(conversation.get_conversation_summary())
    print(graph.get_statistics())


if __name__ == "__main__":
    main()
