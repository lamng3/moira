from typing import Dict, Iterable, List, Optional, Tuple, Union
from rdflib import Namespace

class NamespaceRegistry:
    """holding namespaces"""
    def __init__(self):
        self._map: Dict[str, Namespace] = {}

    def add(self, prefix: str, ns: Union[str, Namespace]):
        ns_obj = ns if isinstance(ns, Namespace) else Namespace(str(ns))
        self._map[prefix] = ns_obj
        # attribute access: rdf / RDF
        setattr(self, prefix, ns_obj)
        setattr(self, prefix.upper(), ns_obj)

    def update(self, mapping: Dict[str, Union[str, Namespace]]):
        for p, n in mapping.items():
            self.add(p, n)

    def get(self, prefix: str) -> Optional[Namespace]:
        return self._map.get(prefix)

    def resolve(self, prefix: str, fallback_uri: Optional[str] = None) -> Namespace:
        ns = self.get(prefix)
        if ns is not None:
            return ns
        if fallback_uri:
            self.add(prefix, fallback_uri)
            return self._map[prefix]
        return Namespace("")
