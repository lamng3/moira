from __future__ import annotations
import os
from typing import Optional, Tuple, Dict, Any, List
import boto3
from boto3.dynamodb.conditions import Attr


def session(*, region: Optional[str] = None, profile: Optional[str] = None) -> boto3.session.Session:
    """
    Create an isolated boto3 Session (no global side effects).
    Region is resolved from args, env(AWS_REGION/AWS_DEFAULT_REGION), fallback 'us-east-2'.
    """
    region = region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-2"
    if profile:
        return boto3.session.Session(region_name=region, profile_name=profile)
    return boto3.session.Session(region_name=region)


def table(table_name: str, *, region: Optional[str] = None, profile: Optional[str] = None):
    """
    Return a DynamoDB Table handle.
    Honors AWS_DYNAMODB_ENDPOINT for LocalStack/dev overrides.
    """
    sess = session(region=region, profile=profile)
    endpoint = os.getenv("AWS_DYNAMODB_ENDPOINT")  # e.g., http://localhost:4566
    return sess.resource("dynamodb", endpoint_url=endpoint).Table(table_name)


def key_schema_safe(tbl) -> Tuple[str, Optional[str]]:
    """
    Detect PK/SK from DescribeTable. If blocked/unavailable, fall back to env hints:
      DYNAMO_PK / DYNAMO_SK
    Finally default to ("PK", "SK") which is common in single-table designs.
    """
    pk_env = os.getenv("DYNAMO_PK")
    sk_env = os.getenv("DYNAMO_SK")
    if pk_env:
        return pk_env, (sk_env or None)
    try:
        desc = tbl.meta.client.describe_table(TableName=tbl.name)
        ks = desc["Table"]["KeySchema"]
        pk = next(k["AttributeName"] for k in ks if k["KeyType"] == "HASH")
        sk = next((k["AttributeName"] for k in ks if k["KeyType"] == "RANGE"), None)
        return pk, sk
    except Exception:
        return "PK", "SK"


def _best_ts_attr(item: Dict[str, Any]) -> Optional[str]:
    """choose the best timestamp attribute present on an item."""
    for k in ("created_at", "CreatedAt", "timestamp", "Timestamp"):
        if k in item and item[k]:
            return k
    return None


def _best_run_id(item: Dict[str, Any], pk_name: str) -> Optional[str]:
    """choose the best run id attribute present on an item, fallback to PK if needed."""
    for k in ("run_id", "RunId"):
        v = item.get(k)
        if v:
            return v
    return item.get(pk_name)


def find_latest_run_id(tbl) -> Optional[str]:
    """return the newest single run id in the table (if any)."""
    ids = list_latest_run_ids(tbl, limit=1)
    return ids[0] if ids else None


def list_latest_run_ids(tbl, *, limit: int = 5) -> List[str]:
    """List latest run IDs using simple scan (more reliable)"""
    # Use simple scan to avoid DynamoDB projection conflicts
    return _list_latest_run_ids_simple_scan(tbl, limit=limit)


def _list_latest_run_ids_with_projection(tbl, *, limit: int = 5) -> List[str]:
    """Original implementation with projection"""
    pk_name, sk_name = key_schema_safe(tbl)

    # Very simple projection - just get the primary key and sort key
    expr_names: Dict[str, str] = {
        "#pk": pk_name,
    }
    proj = ["#pk"]
    
    if sk_name:
        expr_names["#sk"] = sk_name
        proj.append("#sk")

    # Don't project specific fields to avoid conflicts - just get all attributes
    scan_kwargs: Dict[str, Any] = {"ProjectionExpression": ", ".join(proj), "ExpressionAttributeNames": expr_names}

    # build a flexible FilterExpression
    fe = None
    if sk_name:
        meta_vals = os.getenv("DYNAMO_META_VALUES", "META,meta").split(",")
        # (SK == any meta value)
        for v in meta_vals:
            cond = Attr(sk_name).eq(v.strip())
            fe = cond if fe is None else (fe | cond)

        # …or any item that explicitly carries a run id
        rid_exists = Attr("run_id").exists() | Attr("RunId").exists()
        fe = rid_exists if fe is None else (fe | rid_exists)
    else:
        # No SK: accept anything that has a sensible timestamp (or a run id)
        fe = (Attr("created_at").exists() | Attr("CreatedAt").exists() |
              Attr("run_id").exists() | Attr("RunId").exists())

    scan_kwargs["FilterExpression"] = fe

    # full scan with pagination
    items: List[Dict[str, Any]] = []
    while True:
        resp = tbl.scan(**scan_kwargs)
        items.extend(resp.get("Items", []))
        lek = resp.get("LastEvaluatedKey")
        if not lek:
            break
        scan_kwargs["ExclusiveStartKey"] = lek

    # filter to those with some timestamp, then sort descending by that ts
    def ts_value(it: Dict[str, Any]) -> str:
        k = _best_ts_attr(it)
        return it.get(k, "") if k else ""

    items = [it for it in items if ts_value(it)]
    items.sort(key=ts_value, reverse=True)

    # collect distinct run ids (preserving order)
    out: List[str] = []
    for it in items:
        rid = _best_run_id(it, pk_name)
        if rid and rid not in out:
            out.append(rid)
            if len(out) >= limit:
                break
    return out


def _list_latest_run_ids_simple_scan(tbl, *, limit: int = 5) -> List[str]:
    """Fallback implementation with simple scan (no projection)"""
    pk_name, sk_name = key_schema_safe(tbl)
    
    # Simple scan without projection to avoid conflicts
    scan_kwargs: Dict[str, Any] = {}
    
    # Build a simple filter expression
    fe = None
    if sk_name:
        meta_vals = os.getenv("DYNAMO_META_VALUES", "META,meta").split(",")
        # (SK == any meta value)
        for v in meta_vals:
            cond = Attr(sk_name).eq(v.strip())
            fe = cond if fe is None else (fe | cond)

        # …or any item that explicitly carries a run id
        rid_exists = Attr("run_id").exists() | Attr("RunId").exists()
        fe = rid_exists if fe is None else (fe | rid_exists)
    else:
        # No SK: accept anything that has a sensible timestamp (or a run id)
        fe = (Attr("created_at").exists() | Attr("CreatedAt").exists() |
              Attr("run_id").exists() | Attr("RunId").exists())

    if fe:
        scan_kwargs["FilterExpression"] = fe

    # full scan with pagination
    items: List[Dict[str, Any]] = []
    while True:
        resp = tbl.scan(**scan_kwargs)
        items.extend(resp.get("Items", []))
        lek = resp.get("LastEvaluatedKey")
        if not lek:
            break
        scan_kwargs["ExclusiveStartKey"] = lek

    # filter to those with some timestamp, then sort descending by that ts
    def ts_value(it: Dict[str, Any]) -> str:
        k = _best_ts_attr(it)
        return it.get(k, "") if k else ""

    items = [it for it in items if ts_value(it)]
    items.sort(key=ts_value, reverse=True)

    # collect distinct run ids (preserving order)
    out: List[str] = []
    for it in items:
        rid = _best_run_id(it, pk_name)
        if rid and rid not in out:
            out.append(rid)
            if len(out) >= limit:
                break
    return out
