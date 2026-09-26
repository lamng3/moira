from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class Config:
    # service endpoints
    OLS4_BASE: str = os.getenv("OLS4_BASE", "https://www.ebi.ac.uk/ols4")
    BIOPORTAL_BASE: str = os.getenv("BIOPORTAL_BASE", "https://data.bioontology.org")
    LOV_API: str = os.getenv("LOV_API", "https://lov.linkeddata.es/dataset/lov/api/v2")
    DBPEDIA_SPARQL: str = os.getenv("DBPEDIA_SPARQL", "http://dbpedia.org/sparql")
    YAGO_SPARQL: str = os.getenv("YAGO_SPARQL", "https://yago-knowledge.org/sparql/query")
    WIKIDATA_SPARQL: str = os.getenv("WIKIDATA_SPARQL", "https://query.wikidata.org/sparql")

    # credentials
    BIOPORTAL_API_KEY: Optional[str] = os.getenv("BIOPORTAL_API_KEY")

    # timeouts / headers
    DEFAULT_TIMEOUT: float = float(os.getenv("HTTP_TIMEOUT", 15))
    HEADERS_JSON: Dict[str, str] = None  # set in __post_init__
    HEADERS_SPARQL: Dict[str, str] = None

    # todo: MI-MatOnto
    MATERIALINFORMATION_URL: Optional[str] = os.getenv("MATERIALINFORMATION_URL")
    MATERIALINFORMATION_PARTS: Optional[str] = os.getenv("MATERIALINFORMATION_PARTS")

    DEFAULT_TESTBEDS: Dict[str, List[str]] = field(default_factory=lambda: {
        "Mouse-Human": ["MA", "NCIT"],
        "NCIT-DOID": ["NCIT", "DOID"],
        "Nell-DBpedia": ["DBpedia"],
        "YAGO-WikiData": ["YAGO", "Wikidata"],
        "GeoOutageOnto": ["GeoOutageOnto"],
        # todo: MI-MatOnto
    })

    def __post_init__(self):
        self.HEADERS_JSON = {"Accept": "application/json"}
        self.HEADERS_SPARQL = {
            "Accept": "application/sparql-results+json",
            "User-Agent": "ontology-lookup-plus/1.0 (+https://example.org)",
        }