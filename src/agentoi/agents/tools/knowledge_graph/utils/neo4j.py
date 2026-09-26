from __future__ import annotations
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase, Driver

def build_driver(uri: str, user: Optional[str] = None, password: Optional[str] = None) -> Optional[Driver]:
    try:
        if user and password:
            return GraphDatabase.driver(uri, auth=(user, password))
        return GraphDatabase.driver(uri)
    except Exception:
        return None

def pick_first(props: Dict[str, Any], keys: List[str], default: Optional[str] = None) -> Optional[str]:
    for k in keys:
        if k in props and props[k] not in (None, ""):
            v = props[k]
            if isinstance(v, (list, tuple)):
                return str(v[0]) if v else default
            return str(v)
    return default

def record_to_item(record, *, kg_name: str, name_props: List[str], desc_props: List[str], iri_props: List[str]) -> Dict[str, Any]:
    props: Dict[str, Any] = record["props"]
    nm = pick_first(props, name_props, default=f"node/{record['id']}")
    desc = pick_first(props, desc_props, default="") or ""
    iri  = pick_first(props, iri_props)
    return {
        "source": "neo4j",
        "kg": kg_name,
        "id": int(record["id"]),
        "labels": list(record["labels"]),
        "name": nm,
        "description": desc,
        "iri": iri,
        "props": props,
    }
