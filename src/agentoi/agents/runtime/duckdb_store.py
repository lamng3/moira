"""One local DuckDB file for thought-graph runs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentoi.agents.ontology_of_thought.serialization import to_jsonable

DEFAULT_PATH = Path("results/traces/agentoi.duckdb")


def trace_store_name() -> str:
    return (os.getenv("AGENTOI_TRACE_STORE") or "").strip().lower()


def trace_db_path() -> Path:
    override = os.getenv("AGENTOI_TRACE_DB")
    return Path(override) if override else DEFAULT_PATH


def save_run(
    path: Path,
    *,
    run_id: str,
    slug: str,
    query: str | None,
    graph: dict[str, Any],
    memory_trace: dict[str, Any] | None,
    start_id: str,
    end_id: str,
    created_at: datetime | None = None,
) -> Path:
    """Insert one run. A repeated run id replaces the previous row."""
    stamp = created_at or datetime.now(timezone.utc)
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(timezone.utc).replace(tzinfo=None)
    connection = _connect(path)
    try:
        connection.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, created_at, slug, query, start_id, end_id, graph, memory_trace
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                stamp,
                slug,
                query,
                start_id,
                end_id,
                json.dumps(to_jsonable(graph or {"thoughts": [], "edges": []})),
                json.dumps(to_jsonable(memory_trace or {})),
            ],
        )
    finally:
        connection.close()
    return path


def list_runs(path: Path, *, limit: int | None = None) -> list[str]:
    """Return run ids, newest first."""
    connection = _connect(path)
    try:
        sql = "SELECT run_id FROM runs ORDER BY created_at DESC"
        if limit is None:
            rows = connection.execute(sql).fetchall()
        else:
            rows = connection.execute(f"{sql} LIMIT ?", [limit]).fetchall()
    finally:
        connection.close()
    return [str(row[0]) for row in rows]


def load_run(path: Path, run_id: str) -> dict[str, Any]:
    """Return the same run and graph shape as the DynamoDB loader."""
    connection = _connect(path)
    try:
        row = connection.execute(
            """
            SELECT run_id, created_at, slug, start_id, end_id, graph, memory_trace
            FROM runs WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        graph = {"thoughts": [], "edges": []}
        memory: dict[str, Any] = {}
        created = None
        slug = None
        start_id = None
        end_id = None
    else:
        graph = _json_object(row[5]) or {"thoughts": [], "edges": []}
        memory = _json_object(row[6]) or {}
        created = row[1]
        slug = row[2]
        start_id = row[3]
        end_id = row[4]
    thoughts = list(graph.get("thoughts") or [])
    edges = list(graph.get("edges") or [])
    return {
        "run": {
            "RunId": run_id,
            "Slug": slug,
            "CreatedAt": _stamp(created),
            "StartId": start_id,
            "EndId": end_id,
            "NodeCount": len(thoughts),
            "EdgeCount": len(edges),
            "MemoryTrace": memory,
        },
        "graph": {"thoughts": thoughts, "edges": edges},
    }


def _connect(path: Path):
    import duckdb

    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id VARCHAR PRIMARY KEY,
            created_at TIMESTAMP,
            slug VARCHAR,
            query VARCHAR,
            start_id VARCHAR,
            end_id VARCHAR,
            graph JSON,
            memory_trace JSON
        )
        """
    )
    return connection


def _json_object(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return json.loads(str(value))


def _stamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return str(value)
