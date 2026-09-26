"""Select or extend AgentOI retrieval backends."""

from agentoi.retrieval import (
    StaticSearchProvider,
    WebSearchPlugin,
    available_vector_indexes,
    available_web_searches,
    create_web_search,
    register_web_search,
)


def main() -> None:
    print("Web search:", available_web_searches())
    print("Vector index:", available_vector_indexes())

    offline = create_web_search(
        "static",
        counts={"ontology": 100, "knowledge graph": 80},
    )
    print("Ontology results:", offline.search_count("ontology"))

    register_web_search(
        WebSearchPlugin(
            name="project-index",
            description="Project-specific search counts.",
            network_access=False,
        ),
        StaticSearchProvider,
    )
    project_index = create_web_search(
        "project-index",
        counts={"agent ontology": 25},
    )
    print("Project results:", project_index.search_count("agent ontology"))


if __name__ == "__main__":
    main()
