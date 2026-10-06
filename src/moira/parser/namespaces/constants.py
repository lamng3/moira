from typing import Dict
from rdflib import Namespace
from rdflib.namespace import RDF, RDFS, OWL, XSD, SKOS, DCTERMS

NAMESPACES: Dict[str, Namespace] = {
    "rdf": RDF,
    "rdfs": RDFS,
    "owl": OWL,
    "xsd": XSD,
    "skos": SKOS,
    "dcterms": DCTERMS,  # http://purl.org/dc/terms/
    "dc": Namespace("http://purl.org/dc/elements/1.1/"),
    "foaf": Namespace("http://xmlns.com/foaf/0.1/"),

    # OBO stack
    "obo": Namespace("http://purl.obolibrary.org/obo/"),
    "IAO": Namespace("http://purl.obolibrary.org/obo/IAO_"),  # e.g. IAO["0000115"]
    "oboInOwl": Namespace("http://www.geneontology.org/formats/oboInOwl#"),
    "oboRel":   Namespace("http://www.obofoundry.org/ro/ro.owl#"),
}