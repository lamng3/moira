"""Collect web context for a term."""

import argparse
import json

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("term")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--domain", action="append", dest="domains")
    args = parser.parse_args()

    from agentoi.agents.tools.website.website_lookup import WebsiteLookup

    with WebsiteLookup() as lookup:
        result = lookup.term_context(
            args.term,
            top_k=args.top_k,
            site_filters=args.domains,
        )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
