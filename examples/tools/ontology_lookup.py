"""Search ontology terms."""

import argparse
import json

from moira.agents.tools.ontology.ontology_lookup import OntologyLookup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("term")
    parser.add_argument("--ontology", action="append", dest="ontologies")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    result = OntologyLookup().term_info(
        args.term,
        ontologies=args.ontologies,
        max_results=args.top_k,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
