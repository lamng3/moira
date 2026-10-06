"""Search configured knowledge graphs."""

import argparse
import json

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("term")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--public-only", action="store_true")
    args = parser.parse_args()

    from moira.agents.tools.knowledge_graph.knowledge_graph_lookup import (
        KnowledgeGraphLookup,
        SearchOpts,
    )

    options = SearchOpts(
        top_k=args.top_k,
        include_public=True,
        only_kgs=[] if args.public_only else None,
    )
    with KnowledgeGraphLookup() as lookup:
        result = lookup.entity_search(args.term, opts=options)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
