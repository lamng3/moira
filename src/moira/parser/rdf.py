from __future__ import annotations
import logging
import uuid
from pathlib import Path
from typing import TypeAlias

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from moira.algorithms.graph import Ontology
from moira.algorithms.graph.identifiers import name_from_uuid

from .errors import ParseError
from .formats import InputFormat, parse_format
from .io import read_text
from .namespaces.constants import NAMESPACES
from .namespaces.registry import NamespaceRegistry

NodeLike: TypeAlias = str | URIRef | BNode

class RDFParser:
    """Parse any RDF serialization supported by the package."""

    def __init__(
        self, 
        name: str | None = None,
        version: str | None = None,
        namespaces: dict[str, str | Namespace] | None = None,
    ):
        self.graph = Graph()
        self.name: str = name if name is not None else name_from_uuid(uuid.uuid4(), "codename")
        self.version: str = version if version is not None else "0.1.0"
        self.namespaces = NamespaceRegistry()
        self._configured_namespaces = dict(namespaces or {})
        self._bind_namespaces(NAMESPACES)
        if namespaces:
            self._bind_namespaces(namespaces)

    def to_ontology(self) -> Ontology:
        """convert rdf to ontology"""
        extracted = self.extract_triples(
            as_n3=False, # use iri
            exclude_literal_objects=True,
            exclude_blank_nodes=True, # skip anonymous nodes
        )
        # RDF type declarations describe the serialization schema; they are not
        # ontology relations and would otherwise create nodes such as owl:Class.
        triples = [
            triple for triple in extracted if triple[1] != str(RDF.type)
        ]
        ont = Ontology(name=self.name, version=self.version)
        ont = ont.build_ontology_from_triples(triples)
        declared_classes = {
            str(subject)
            for class_type in (OWL.Class, RDFS.Class)
            for subject in self.graph.subjects(RDF.type, class_type)
            if not isinstance(subject, BNode)
        }
        for class_iri in declared_classes:
            ont.get_node(class_iri)
        ont.nodes = list(ont.node_map.values())
        ont._rebuild_local_index()

        # enrich node with name
        name_to_concept = {c.name: c for c in ont.nodes}

        node_ids = set(declared_classes)
        for s, _, o in triples:
            node_ids.add(s)
            node_ids.add(o)

        for node_id in node_ids:
            try:
                # extract metadata
                info = self.extract_node_info(node_id)
            except Exception as e:
                logging.debug(f"extract_node_info failed for {node_id}: {e}")
                continue

            # concepts keyed by n3 strings
            c = name_to_concept.get(info.get("n3")) or name_to_concept.get(node_id)
            if not c:
                continue

            # enrich node textual information
            labels: list[str] = []
            alt_labels: list[str] = []
            related_synonyms: list[str] = []
            exact_synonyms: list[str] = []
            definitions: list[str] = []

            vals = info.get("labels") or []
            labels.extend(vals)

            vals = (info.get("data_properties") or {}).get("skos:prefLabel") or []
            labels.extend(vals)

            vals = info.get("data_properties").get("skos:altLabel") or []
            alt_labels.extend(vals)

            vals = info.get("data_properties").get("oboInOwl:hasRelatedSynonym") or []
            related_synonyms.extend(vals)

            vals = info.get("data_properties").get("oboInOwl:hasExactSynonym") or []
            exact_synonyms.extend(vals)

            vals = info.get("definitions") or []
            definitions.extend(vals)
            for key in info.get("data_properties").keys():
                if "definition" in self.extract_node_info(key).get("labels"):
                    vals = info.get("data_properties").get(key) or []
                    definitions.extend(vals)

            # dedup and update ground text
            seen = set()
            for l in labels:
                if l in seen:
                    continue
                seen.add(l)
                if l not in c.ground_set["labels"]:
                    c.ground_set["labels"].append(l)

            seen = set()
            for al in alt_labels:
                if al in seen:
                    continue
                seen.add(al)
                if al not in c.ground_set["alt_labels"]:
                    c.ground_set["alt_labels"].append(al)

            seen = set()
            for rs in related_synonyms:
                if rs in seen:
                    continue
                seen.add(rs)
                if rs not in c.ground_set["related_synonyms"]:
                    c.ground_set["related_synonyms"].append(rs)

            seen = set()
            for es in exact_synonyms:
                if es in seen:
                    continue
                seen.add(es)
                if es not in c.ground_set["exact_synonyms"]:
                    c.ground_set["exact_synonyms"].append(es)

            seen = set()
            for d in definitions:
                if d in seen:
                    continue
                seen.add(d)
                if d not in c.ground_set["definitions"]:
                    c.ground_set["definitions"].append(d)

        return ont

    def _bind_namespaces(self, mapping: dict[str, str | Namespace]) -> None:
        for prefix, ns in mapping.items():
            ns_obj = ns if isinstance(ns, Namespace) else Namespace(str(ns))
            self.graph.bind(prefix, ns_obj, override=True)
            self.namespaces.add(prefix, ns_obj)

    def add_namespaces(self, mapping: dict[str, str | Namespace]) -> "RDFParser":
        """add or override namespaces"""
        self._configured_namespaces.update(mapping)
        self._bind_namespaces(mapping)
        return self

    def clear(self) -> "RDFParser":
        """clear parser"""
        self.graph = Graph()
        self.namespaces = NamespaceRegistry()
        self._bind_namespaces(NAMESPACES)
        self._bind_namespaces(self._configured_namespaces)
        return self

    def parse_text(
        self,
        text: str,
        format: str | InputFormat = InputFormat.TURTLE,
        *,
        source: str | Path | None = None,
    ) -> "RDFParser":
        """Parse serialized RDF text into the graph."""
        selected = parse_format(format)
        if selected.adapter != "rdf":
            raise ParseError(
                "Format is not an RDF serialization",
                source=source,
                format=selected.value,
            )
        try:
            self.graph.parse(data=text, format=selected.value)
        except Exception as exc:
            raise ParseError(
                "Invalid RDF input", source=source, format=selected.value
            ) from exc
        return self

    def parse_file(
        self, path: str | Path, format: str | InputFormat
    ) -> "RDFParser":
        """Parse a local RDF file, including gzip and bzip2 inputs."""
        return self.parse_text(read_text(path), format=format, source=path)

    def _to_node(self, s: NodeLike) -> URIRef | BNode:
        """
            accepts:
                * rdflib node (URIRef/BNode)
                * CURIE: 'mds:CellNumberY'
                * IRI: '<http://...>' or 'http://...'
                * blank node: '_:b1'
        """
        if isinstance(s, (URIRef, BNode)):
            return s
        nm = self.graph.namespace_manager
        s = s.strip()
        if s.startswith("_:"):
            return BNode(s[2:])
        if s.startswith("<") and s.endswith(">"):
            return URIRef(s[1:-1])
        if "://" in s and not s.startswith("<"):
            return URIRef(s)
        if ":" in s:  # CURIE
            if hasattr(nm, "expand_curie"):
                return nm.expand_curie(s)
            prefix, local = s.split(":", 1)
            base = nm.store.namespace(prefix)
            if base is None:
                raise ValueError(f"Unknown prefix: {prefix}")
            return URIRef(base + local)
        return URIRef(s)

    def _is_blankish(self, node):
        from rdflib import URIRef, BNode
        if isinstance(node, BNode):
            return True
        if isinstance(node, URIRef):
            u = str(node)
            return ("://" not in u) and (not u.startswith(("urn:", "tag:")))
        return False

    def qname(self, node: NodeLike) -> str:
        """compact name for display"""
        n = self._to_node(node)
        return self.graph.namespace_manager.normalizeUri(n)

    def _display(self, node):
        if isinstance(node, Literal):
            return str(node)
        try:
            return self.qname(node)  # pretty CURIE if possible
        except Exception:
            try:
                nm = self.graph.namespace_manager
                return node.n3(nm)  # e.g., <http://...> or _:b0
            except Exception:
                return str(node)

    def _textify(self, node):
        g = self.graph
        ns = self.namespaces
        RDFS = ns.resolve("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
        LABEL    = RDFS.label
        if isinstance(node, Literal): return str(node)
        labels = [str(l) for l in g.objects(node, LABEL)]
        return labels[0] if labels else self._display(node)

    def _parenthesize(self, x, depth):
        """parenthesize compound expressions when nested"""
        if depth > 0 and (" and " in x or " or " in x):
            return f"({x})"
        return x

    def _iter_rdf_list(self, head):
        g = self.graph
        ns = self.namespaces

        RDF  = ns.resolve("rdf",  "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        NIL = RDF.nil
        FIRST, REST = RDF.first, RDF.rest

        cur = head
        seen = set()
        while cur and cur != NIL and cur not in seen:
            seen.add(cur)
            first = g.value(cur, FIRST)
            if first is not None:
                # return a value and pause execution
                yield first
            cur = g.value(cur, REST)

    def _render_class_expr(self, node, depth=0):
        """
        render OWL class expressions (restrictions, intersections, unions, complements, oneOf) into a Manchester-like string. 
        return CURIE/IRI for named classes.
        """
        from rdflib import Literal, URIRef, BNode
        g = self.graph
        ns = self.namespaces
        RDF  = ns.resolve("rdf",  "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        RDFS = ns.resolve("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
        OWL  = ns.resolve("owl",  "http://www.w3.org/2002/07/owl#")

        if isinstance(node, Literal):
            return str(node)
        if isinstance(node, URIRef) and not self._is_blankish(node):
            return self._display(node)

        if (node, RDF.type, OWL.Restriction) in g:
            p = g.value(node, OWL.onProperty)
            p_str = self._display(p) if p else "<?>"

            # has self
            has_self = g.value(node, OWL.hasSelf)
            if isinstance(has_self, Literal) and str(has_self).lower() in ("true", "1"):
                return f"{p_str} Self"

            # qualified candidates
            for pred, kw in [(OWL.qualifiedCardinality, "exactly"),
                         (OWL.minQualifiedCardinality, "min"),
                         (OWL.maxQualifiedCardinality, "max")]:
                n = g.value(node, pred)
                if n is not None:
                    on_cls = g.value(node, OWL.onClass)
                    n_str = str(int(n)) if isinstance(n, Literal) else self._display(n)
                    if on_cls:
                        return f"{p_str} {kw} {n_str} {self._render_class_expr(on_cls, depth+1)}"
                    return f"{p_str} {kw} {n_str}"

            # unqualified candidates
            for pred, kw in [(OWL.cardinality, "exactly"),
                         (OWL.minCardinality, "min"),
                         (OWL.maxCardinality, "max")]:
                n = g.value(node, pred)
                if n is not None:
                    n_str = str(int(n)) if isinstance(n, Literal) else self._display(n)
                    return f"{p_str} {kw} {n_str}"

            # value / some / only
            v = g.value(node, OWL.hasValue)
            if v is not None:
                v_str = self._display(v) if not isinstance(v, Literal) else str(v)
                return f"{p_str} value {v_str}"

            svf = g.value(node, OWL.someValuesFrom)
            if svf is not None:
                return f"{p_str} some {self._render_class_expr(svf, depth+1)}"

            avf = g.value(node, OWL.allValuesFrom)
            if avf is not None:
                return f"{p_str} only {self._render_class_expr(avf, depth+1)}"

            return self._display(node)

        # boolean class constructors
        inter = g.value(node, OWL.intersectionOf)
        if inter is not None:
            parts = [self._render_class_expr(x, depth+1) for x in self._iter_rdf_list(inter)]
            return self._parenthesize(" and ".join(parts), depth)

        union = g.value(node, OWL.unionOf)
        if union is not None:
            parts = [self._render_class_expr(x, depth+1) for x in self._iter_rdf_list(union)]
            return self._parenthesize(" or ".join(parts), depth)

        compl = g.value(node, OWL.complementOf)
        if compl is not None:
            return f"not {self._parenthesize(self._render_class_expr(compl, depth+1), depth)}"

        oneof = g.value(node, OWL.oneOf)
        if oneof is not None:
            items = [self._display(x) for x in self._iter_rdf_list(oneof)]
            return "{" + ", ".join(items) + "}"

        # unknown anonymous expression
        return self._display(node)

    def extract_node_info(self, subject: NodeLike) -> dict[str, object]:
        """extract info about a node"""
        g = self.graph
        nm = g.namespace_manager
        s = self._to_node(subject)

        ns = self.namespaces
        RDF  = ns.resolve("rdf",  "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        RDFS = ns.resolve("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
        OWL  = ns.resolve("owl",  "http://www.w3.org/2002/07/owl#")
        SKOS = ns.resolve("skos", "http://www.w3.org/2004/02/skos/core#")
        OBO  = ns.resolve("oboInOwl", "http://www.geneontology.org/formats/oboInOwl#")

        TYPE     = RDF.type
        LABEL    = RDFS.label
        COMMENT  = RDFS.comment
        SUBCLASS = RDFS.subClassOf
        EQUIV    = OWL.equivalentClass
        SAMEAS   = OWL.sameAs

        def uniq(seq):
            seen = set(); out = []
            for x in seq:
                if x not in seen:
                    seen.add(x); out.append(x)
            return out

        uri = str(s) if isinstance(s, URIRef) else None
        node_id = s.n3(nm)

        labels   = [str(o) for o in g.objects(s, LABEL)]
        comments = [str(o) for o in g.objects(s, COMMENT)]
        types    = uniq([self._display(o) for o in g.objects(s, TYPE)])

        # synonyms
        synonym_props = [
            SKOS.altLabel,
            OBO.hasExactSynonym,
            OBO.hasRelatedSynonym,
            OBO.hasBroadSynonym,
            OBO.hasNarrowSynonym,
        ]
        synonyms = uniq([self._textify(o) for p in synonym_props for o in g.objects(s, p)])

        # definitions
        def_props   = [SKOS.definition, OBO.hasDefinition, COMMENT]
        definitions = uniq([self._textify(o) for p in def_props for o in g.objects(s, p)])

        # hierarchy
        parents  = uniq([self._render_class_expr(o)    for o    in g.objects(s, SUBCLASS)])
        children = uniq([self._render_class_expr(subj) for subj in g.subjects(SUBCLASS, s)])

        # equivalences
        equivalent_to = uniq([self._render_class_expr(o) for o in g.objects(s, EQUIV)])
        same_as       = uniq([self._render_class_expr(o) for o in g.objects(s, SAMEAS)])

        # capture all other predicates
        skip = {LABEL, COMMENT, TYPE, SUBCLASS, EQUIV, SAMEAS}
        properties, data_properties = {}, {}
        for p, o in g.predicate_objects(s):
            if p in skip: 
                continue
            p_name = self._display(p)
            if isinstance(o, Literal):
                data_properties.setdefault(p_name, []).append(str(o))
            else:
                properties.setdefault(p_name, []).append(self._render_class_expr(o))

        # dedup
        properties = {k: sorted(set(v)) for k, v in properties.items()}
        data_properties = {k: sorted(set(v)) for k, v in data_properties.items()}

        return {
            "uri": uri,
            "n3": node_id,
            "qname": None if uri is None else self.qname(s),
            "labels": uniq(labels),
            "comments": uniq(comments),
            "types": types,
            "synonyms": synonyms,
            "definitions": definitions,
            "parents": parents,
            "children": children,
            "equivalent_to": equivalent_to,
            "same_as": same_as,
            "properties": properties,           
            "data_properties": data_properties,
        }

    def extract_triples(
        self,
        subject: NodeLike | None = None,
        as_n3: bool = False,
        exclude_literal_objects: bool = False,
        exclude_blank_nodes: bool = True,
    ) -> list[tuple[str, str, str]]:
        nm = self.graph.namespace_manager
        
        def fmt(node): return node.n3(nm) if as_n3 else str(node)
        
        triples: list[tuple[str, str, str]] = []
        if subject is not None:
            s_node = self._to_node(subject)
            for p, o in self.graph.predicate_objects(s_node):
                if exclude_literal_objects and isinstance(o, Literal): 
                    continue
                if exclude_blank_nodes and isinstance(o, BNode):
                    continue
                triples.append((fmt(s_node), fmt(p), fmt(o)))
        else:
            for s, p, o in self.graph:
                if exclude_literal_objects and isinstance(o, Literal): 
                    continue
                if exclude_blank_nodes and (isinstance(s, BNode) or isinstance(o, BNode)):
                    continue
                triples.append((fmt(s), fmt(p), fmt(o)))
        
        triples.sort()
        return triples
