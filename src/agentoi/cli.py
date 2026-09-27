"""Public command-line interface for AgentOI."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any
from pathlib import Path

from dotenv import load_dotenv

from agentoi.memory_store import clear_ontology_memory
from agentoi.progress import StderrProgress


def _add_ontology_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("ontology", type=Path, help="Path to an OWL or RDF ontology.")
    parser.add_argument(
        "--format",
        help="Input format override, such as xml, turtle, nt, or json-ld.",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Apply relation refinement before building the concept graph.",
    )


def _add_memory_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--memory-dir",
        type=Path,
        help="Directory for saved concept embeddings. Defaults to results/memory.",
    )
    parser.add_argument(
        "--no-save-memory",
        action="store_true",
        help="Do not write concept embeddings after building them.",
    )


def _add_query_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--top-k", type=int, default=10, help="Context size.")
    parser.add_argument("--model", help="LLM name; defaults to LLM_MODEL.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--save-traces",
        action="store_true",
        help="Write Ontology-of-Thought traces under results/traces/.",
    )
    _add_memory_options(parser)
    parser.add_argument(
        "--log",
        action="store_true",
        help="Show the detailed agent log while answering. Otherwise type 'log' after an answer.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentoi",
        description="Inspect, search, and query ontologies from the terminal.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Show ontology metadata without loading a model."
    )
    _add_ontology_options(inspect_parser)
    inspect_parser.add_argument("--json", action="store_true", dest="as_json")

    search_parser = subparsers.add_parser(
        "search", help="Retrieve relevant concepts without using an LLM."
    )
    _add_ontology_options(search_parser)
    search_parser.add_argument("query", help="Natural-language search text.")
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.add_argument("--json", action="store_true", dest="as_json")
    _add_memory_options(search_parser)

    query_parser = subparsers.add_parser(
        "query", help="Answer one question using ontology context."
    )
    _add_ontology_options(query_parser)
    query_parser.add_argument("query", help="Natural-language question.")
    _add_query_options(query_parser)

    chat_parser = subparsers.add_parser(
        "chat", help="Open an interactive ontology question session."
    )
    _add_ontology_options(chat_parser)
    _add_query_options(chat_parser)

    subparsers.add_parser(
        "run",
        add_help=False,
        help="Run the catalog-based evaluation pipeline; use 'agentoi run --help'.",
    )
    memory_parser = subparsers.add_parser(
        "memory",
        help="Save or delete concept-embedding memory.",
    )
    memory_commands = memory_parser.add_subparsers(
        dest="memory_command",
        required=True,
    )
    clear_parser = memory_commands.add_parser(
        "clear",
        help="Delete saved embeddings for one ontology.",
    )
    clear_parser.add_argument("ontology", type=Path, help="Ontology whose memory to delete.")
    clear_parser.add_argument(
        "--memory-dir",
        type=Path,
        help="Directory that holds saved concept embeddings.",
    )
    return parser


def _workspace(args: argparse.Namespace):
    from agentoi.workspace import OntologyWorkspace

    return OntologyWorkspace(
        args.ontology,
        format=args.format,
        refine=args.refine,
        memory_dir=getattr(args, "memory_dir", None),
        save_memory=not getattr(args, "no_save_memory", False),
    )


def _inspect(args: argparse.Namespace) -> int:
    summary = _workspace(args).summary
    values = {
        "path": str(summary.path),
        "format": summary.format,
        "name": summary.name,
        "version": summary.version,
        "concepts": summary.concepts,
        "relations": summary.relations,
    }
    if args.as_json:
        print(json.dumps(values, indent=2))
    else:
        for key, value in values.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
    return 0


def _search(args: argparse.Namespace) -> int:
    results = _workspace(args).search(
        args.query,
        top_k=args.top_k,
        progress=StderrProgress(),
    )
    if args.as_json:
        print(json.dumps(results, indent=2))
    else:
        for index, result in enumerate(results, start=1):
            print(f"{index:>2}. {result['label']} ({result['score']:.4f})")
    return 0


def _query_options(args: argparse.Namespace) -> dict[str, object]:
    return {
        "top_k": args.top_k,
        "model": args.model,
        "temperature": args.temperature,
        "save_traces": args.save_traces,
        "progress": StderrProgress(),
        "show_log": bool(getattr(args, "log", False)),
    }


def _query(args: argparse.Namespace) -> int:
    print(_workspace(args).ask(args.query, **_query_options(args)))
    return 0


def _chat(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    print(f"Loading {args.ontology}.", flush=True)
    summary = workspace.summary
    print(
        f"Loaded {summary.name}: {summary.concepts} concepts, "
        f"{summary.relations} relations.",
        flush=True,
    )
    workspace.prepare(progress=StderrProgress())
    print("Ontology and embeddings are loaded.", flush=True)
    if not args.no_save_memory:
        print(
            f"Delete saved embeddings with: agentoi memory clear {args.ontology}",
            flush=True,
        )
    print("Type 'exit' to stop.", flush=True)
    if args.log:
        print("Detailed logs are enabled.", flush=True)
    else:
        print("Type 'log' after an answer to see the detailed run.", flush=True)
    while True:
        try:
            query = input("agentoi> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        command = query.lower()
        if command in {"exit", "quit", ":q"}:
            return 0
        if command == "log":
            _print_last_log(workspace)
            continue
        if query:
            print(workspace.ask(query, **_query_options(args)))
            if not args.log:
                print("Type 'log' to see the details of this answer.", flush=True)


def _print_last_log(workspace: Any) -> None:
    agent = getattr(workspace, "_agent", None)
    run_log = getattr(agent, "run_log", None)
    text = run_log.text() if run_log is not None else ""
    note = getattr(workspace, "last_route_note", None)
    if isinstance(note, str) and note:
        text = f"{note}\n{text}".strip() if text else note
    if text:
        print(text, flush=True)
    else:
        print("No answer log yet. Ask a question first.", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    arguments = list(argv if argv is not None else sys.argv[1:])
    if arguments and arguments[0] == "run":
        from agentoi.main import main as run_pipeline

        return run_pipeline(arguments[1:])

    parser = build_parser()
    args = parser.parse_args(arguments)
    handlers = {
        "inspect": _inspect,
        "search": _search,
        "query": _query,
        "chat": _chat,
        "memory": _memory,
    }
    try:
        return handlers[args.command](args)
    except ImportError as error:
        print(f"agentoi: {error}", file=sys.stderr)
        return 2
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


def _memory(args: argparse.Namespace) -> int:
    if args.memory_command != "clear":
        raise ValueError(f"Unknown memory command: {args.memory_command}")
    ontology = args.ontology.expanduser().resolve()
    if not ontology.is_file():
        raise FileNotFoundError(f"Ontology not found: {ontology}")
    kwargs = {}
    if args.memory_dir is not None:
        kwargs["directory"] = args.memory_dir
    removed = clear_ontology_memory(ontology, **kwargs)
    if removed:
        for folder in removed:
            print(f"Deleted embedding memory {folder}")
    else:
        print(f"No embedding memory found for {ontology}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
