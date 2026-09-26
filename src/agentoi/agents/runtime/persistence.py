from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from agentoi.agents.ontology_of_thought.serialization import (
    to_dynamodb_safe,
    to_jsonable,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_slug(value: str, limit: int = 50) -> str:
    # Keep persistence independent from the legacy utility package.
    import re
    import unicodedata

    value = (value or "").strip()
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii") or value
    value = re.sub(r"\s+", " ", value).strip()[:limit]
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    return re.sub(r"_+", "_", value).strip("._") or "query"


def _preview(value: Any, limit: int = 240) -> str:
    try:
        text = value if isinstance(value, str) else json.dumps(to_jsonable(value), ensure_ascii=False)
    except Exception:
        text = str(value)
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")


def save_trace_to_file(
    traces_dir: Path,
    *,
    run_id: str,
    model_name: str,
    query_text: str | None,
    messages_preview: str | None,
    tool_call: dict[str, Any] | None,
    result: object,
    trace_payload: dict[str, Any],
    graph: dict[str, Any] | None = None,
    memory_trace: dict[str, Any] | None = None,
    start_id: str | None = None,
    end_id: str | None = None,
    write_latest: bool = True,
) -> Path:
    """Persist a JSON-safe agent trace and return its path."""

    traces_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "created_at": _now_iso(),
        "model": model_name,
        "query": query_text,
        "messages_preview": messages_preview,
        "tool_call": tool_call,
        "result_preview": _preview(result),
        "trace": trace_payload,
    }
    optional = {
        "graph": graph,
        "memory_trace": memory_trace,
        "start_id": start_id,
        "end_id": end_id,
    }
    payload.update({key: value for key, value in optional.items() if value is not None})

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    path = traces_dir / f"{stamp}_{run_id}_{_safe_slug(query_text or 'run')}.json"
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")

    if write_latest:
        latest = json.dumps({"path": str(path)}, ensure_ascii=False, indent=2)
        (traces_dir / "latest.json").write_text(latest, encoding="utf-8")
    return path


def _table(table_name: str):
    import boto3

    profile = os.getenv("AWS_PROFILE")
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-2"
    session = boto3.session.Session(region_name=region, profile_name=profile)
    return session.resource(
        "dynamodb", endpoint_url=os.getenv("AWS_DYNAMODB_ENDPOINT")
    ).Table(table_name)


def _key_schema(table_name: str) -> tuple[str, Optional[str]]:
    if os.getenv("DYNAMO_PK"):
        return os.environ["DYNAMO_PK"], os.getenv("DYNAMO_SK")
    try:
        schema = _table(table_name).meta.client.describe_table(
            TableName=table_name
        )["Table"]["KeySchema"]
        partition = next(
            item["AttributeName"] for item in schema if item["KeyType"] == "HASH"
        )
        sort = next(
            (item["AttributeName"] for item in schema if item["KeyType"] == "RANGE"),
            None,
        )
        return partition, sort
    except Exception:
        return "PK", "SK"


def _clean(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _node_hash(thought: dict[str, Any]) -> str:
    fields = (
        thought.get("question"), thought.get("answer"),
        thought.get("content"), thought.get("type"),
    )
    return hashlib.sha256(
        "\n|\n".join(str(value or "").strip() for value in fields).encode()
    ).hexdigest()


def _decode_node(item: dict[str, Any]) -> dict[str, Any]:
    if "Payload" in item:
        return item["Payload"]
    return {
        "id": item.get("NodeId"),
        "type": item.get("Type"),
        "question": item.get("Question"),
        "answer": item.get("Answer"),
        "content": item.get("Content"),
        "tools_used": item.get("ToolsUsed", []),
        "reasoning_depth": item.get("ReasoningDepth"),
        "confidence": item.get("Confidence"),
        "event": item.get("Event"),
        "parent_id": item.get("ParentId"),
        "children_ids": item.get("ChildrenIds", []),
        "tags": item.get("Tags", []),
        "refs": item.get("Refs", []),
        "meta": item.get("Meta", {}),
        "created_at": item.get("CreatedAt"),
        "updated_at": item.get("UpdatedAt"),
    }


def _decode_edge(item: dict[str, Any]) -> dict[str, Any]:
    if "Payload" in item:
        return item["Payload"]
    return {
        "id": item.get("EdgeId"),
        "src": item.get("Src"),
        "dst": item.get("Dst"),
        "decision": item.get("Decision"),
        "kind": item.get("Kind"),
        "src_type": item.get("SrcType"),
        "dst_type": item.get("DstType"),
        "order_index": item.get("OrderIndex"),
        "reasoning_delta": item.get("ReasoningDelta"),
        "note": item.get("Note"),
        "confidence": item.get("Confidence"),
        "event": item.get("Event"),
        "meta": item.get("Meta", {}),
        "created_at": item.get("CreatedAt"),
        "updated_at": item.get("UpdatedAt"),
    }


def save_run_graph(
    *,
    table_name: str,
    run_id: str,
    slug: str,
    graph: dict[str, Any],
    memory_trace: Optional[dict[str, Any]],
    start_id: str,
    end_id: str,
    ttl_days: Optional[int] = None,
) -> tuple[int, int, str]:
    table = _table(table_name)
    partition_key, sort_key = _key_schema(table_name)
    timestamp = _now_iso()
    ttl = (
        int((datetime.now(timezone.utc) + timedelta(days=ttl_days)).timestamp())
        if ttl_days and ttl_days > 0 else None
    )
    ttl_key = os.getenv("DYNAMO_TTL_ATTR", "ttl")
    thoughts = list(graph.get("thoughts") or [])
    edges = list(graph.get("edges") or [])

    if sort_key is None:
        item = {
            partition_key: run_id, "slug": slug, "created_at": timestamp,
            "start_id": start_id, "end_id": end_id,
            "node_count": len(thoughts), "edge_count": len(edges),
            "graph": to_dynamodb_safe(graph),
            "memory_trace": to_dynamodb_safe(memory_trace or {}),
        }
        if ttl is not None:
            item[ttl_key] = ttl
        table.put_item(Item=_clean(item))
        return len(thoughts), len(edges), timestamp

    partition_value = (
        f"RUN#{run_id}"
        if partition_key.upper() in {"PK", "PARTITION", "HASH"} else run_id
    )
    metadata = {
        partition_key: partition_value, sort_key: "META", "ItemType": "RUN",
        "RunId": run_id, "Slug": slug, "CreatedAt": timestamp,
        "StartId": start_id, "EndId": end_id,
        "MemoryTrace": to_dynamodb_safe(memory_trace or {}),
        "NodeCount": len(thoughts), "EdgeCount": len(edges),
    }
    if ttl is not None:
        metadata[ttl_key] = ttl
    table.put_item(Item=_clean(metadata))
    with table.batch_writer(
        overwrite_by_pkeys=[partition_key, sort_key]
    ) as writer:
        for thought in thoughts:
            item = {
                partition_key: partition_value,
                sort_key: f"NODE#{thought['id']}",
                "ItemType": "NODE", "RunId": run_id, "NodeId": thought["id"],
                "NodeHash": _node_hash(thought),
                "Payload": to_dynamodb_safe(thought),
            }
            if ttl is not None:
                item[ttl_key] = ttl
            writer.put_item(Item=item)
        for edge in edges:
            item = {
                partition_key: partition_value,
                sort_key: f"EDGE#{edge['id']}",
                "ItemType": "EDGE", "RunId": run_id, "EdgeId": edge["id"],
                "Payload": to_dynamodb_safe(edge),
            }
            if ttl is not None:
                item[ttl_key] = ttl
            writer.put_item(Item=item)
    return len(thoughts), len(edges), timestamp


def load_run_graph(*, table_name: str, run_id: str) -> dict[str, Any]:
    from boto3.dynamodb.conditions import Key

    table = _table(table_name)
    partition_key, sort_key = _key_schema(table_name)
    if sort_key is None:
        item = table.get_item(Key={partition_key: run_id}).get("Item", {})
        return {
            "run": {
                "RunId": run_id, "Slug": item.get("slug"),
                "CreatedAt": item.get("created_at"),
                "StartId": item.get("start_id"), "EndId": item.get("end_id"),
                "NodeCount": item.get("node_count"),
                "EdgeCount": item.get("edge_count"),
                "MemoryTrace": item.get("memory_trace") or {},
            },
            "graph": item.get("graph") or {"thoughts": [], "edges": []},
        }
    partition_value = (
        f"RUN#{run_id}"
        if partition_key.upper() in {"PK", "PARTITION", "HASH"} else run_id
    )
    response = table.query(
        KeyConditionExpression=Key(partition_key).eq(partition_value)
    )
    items = list(response.get("Items", []))
    while response.get("LastEvaluatedKey"):
        response = table.query(
            KeyConditionExpression=Key(partition_key).eq(partition_value),
            ExclusiveStartKey=response["LastEvaluatedKey"],
        )
        items.extend(response.get("Items", []))
    metadata = next(
        (item for item in items if item.get("ItemType") == "RUN"), {}
    )
    thoughts = [_decode_node(item) for item in items if item.get("ItemType") == "NODE"]
    edges = [_decode_edge(item) for item in items if item.get("ItemType") == "EDGE"]
    return {
        "run": {
            "RunId": run_id, "Slug": metadata.get("Slug"),
            "CreatedAt": metadata.get("CreatedAt"),
            "StartId": metadata.get("StartId"), "EndId": metadata.get("EndId"),
            "NodeCount": metadata.get("NodeCount", len(thoughts)),
            "EdgeCount": metadata.get("EdgeCount", len(edges)),
            "MemoryTrace": metadata.get("MemoryTrace") or {},
        },
        "graph": {"thoughts": thoughts, "edges": edges},
    }
