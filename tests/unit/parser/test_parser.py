from __future__ import annotations

import bz2
import gzip
import json

import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS

import moira.parser as public_api
from moira.parser import (
    InputFormat,
    JSONParser,
    ParseError,
    Parser,
    ParserConfig,
    ParserConfigError,
    RDFParser,
    TextParser,
    UnsupportedFormatError,
    detect_format,
)


TURTLE = """
@prefix ex: <https://example.test/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .

ex:ontology a owl:Ontology .
ex:Parent a owl:Class ; rdfs:label "Parent" .
ex:Child a owl:Class ;
    rdfs:subClassOf ex:Parent ;
    rdfs:label "Child" ;
    rdfs:comment "A child class" ;
    skos:altLabel "Kid" ;
    skos:definition "Child definition" ;
    ex:code "C1" .
"""


def test_public_api_is_explicit() -> None:
    assert public_api.__all__ == [
        "FileRule",
        "InputFormat",
        "JSONParser",
        "ParseError",
        "Parser",
        "ParserConfig",
        "ParserConfigError",
        "ParserError",
        "ParserSelection",
        "RDFParser",
        "TextParser",
        "UnsupportedFormatError",
        "detect_format",
        "parse_format",
    ]


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("ontology.ttl", InputFormat.TURTLE),
        ("ontology.ttl.gz", InputFormat.TURTLE),
        ("ontology.owl.bz2", InputFormat.RDF_XML),
        ("ontology.jsonld", InputFormat.JSON_LD),
        ("dataset.json", InputFormat.JSON),
        ("types.txt", InputFormat.TEXT),
    ],
)
def test_format_detection(filename: str, expected: InputFormat) -> None:
    assert detect_format(filename) is expected


def test_unknown_format_is_explicit() -> None:
    with pytest.raises(UnsupportedFormatError, match="Cannot detect"):
        detect_format("ontology.unknown")


def test_config_selects_first_rule_and_merges_namespaces() -> None:
    config = ParserConfig.from_dict(
        {
            "global": {"namespaces": {"global": "https://global.test/"}},
            "files": [
                {
                    "pattern": "*.data",
                    "format": "ttl",
                    "name": "configured",
                    "version": "2.0",
                    "namespaces": {"ex": "https://example.test/"},
                },
                {"pattern": "*.data", "format": "json"},
            ],
        }
    )
    selection = config.select("anything.data")
    assert selection.format is InputFormat.TURTLE
    assert selection.name == "configured"
    assert selection.version == "2.0"
    assert selection.namespaces == {
        "global": "https://global.test/",
        "ex": "https://example.test/",
    }


@pytest.mark.parametrize(
    "config",
    [
        [],
        {"global": []},
        {"files": {}},
        {"files": [{}]},
        {"global": {"namespaces": {"ex": 1}}},
    ],
)
def test_invalid_config_is_explicit(config: object) -> None:
    with pytest.raises(ParserConfigError):
        ParserConfig.from_dict(config)  # type: ignore[arg-type]


def test_facade_rejects_conflicting_config_sources(tmp_path) -> None:
    with pytest.raises(ParserConfigError, match="either config or config_path"):
        Parser(
            tmp_path / "ontology.ttl",
            config=ParserConfig(),
            config_path=tmp_path / "config.json",
        )


def test_turtle_parser_extracts_triples_and_node_metadata() -> None:
    parser = RDFParser(namespaces={"ex": "https://example.test/"}).parse_text(TURTLE)
    triples = parser.extract_triples(exclude_literal_objects=True)
    assert (
        "https://example.test/Child",
        str(RDFS.subClassOf),
        "https://example.test/Parent",
    ) in triples

    info = parser.extract_node_info("ex:Child")
    assert info["labels"] == ["Child"]
    assert info["comments"] == ["A child class"]
    assert info["synonyms"] == ["Kid"]
    assert info["definitions"] == ["Child definition", "A child class"]
    assert info["parents"] == ["ex:Parent"]
    assert info["data_properties"]["ex:code"] == ["C1"]


def test_rdf_xml_uses_same_parser() -> None:
    graph = Graph()
    child = URIRef("https://example.test/Child")
    graph.add((child, RDF.type, OWL.Class))
    graph.add((child, RDFS.label, Literal("Child")))
    serialized = graph.serialize(format="xml")

    parser = RDFParser().parse_text(serialized, format=InputFormat.RDF_XML)
    assert parser.extract_node_info(child)["labels"] == ["Child"]


def test_rdf_parser_rejects_dataset_format_and_bad_syntax() -> None:
    with pytest.raises(ParseError, match="not an RDF serialization"):
        RDFParser().parse_text("[]", format="json")
    with pytest.raises(ParseError, match=r"Invalid RDF input.*format=turtle"):
        RDFParser().parse_text("@prefix broken", source="bad.ttl")


def test_rdf_to_ontology_enriches_node_grounding() -> None:
    ontology = RDFParser(namespaces={"ex": "https://example.test/"}).parse_text(
        TURTLE
    ).to_ontology()
    child = ontology.get_node("https://example.test/Child")
    assert child is not None
    assert "Child" in child.ground_set["labels"]
    assert "Kid" in child.ground_set["alt_labels"]
    assert "Child definition" in child.ground_set["definitions"]


def test_json_adapter_extracts_dataset_and_metadata() -> None:
    parser = JSONParser(name="terms").parse_text(
        json.dumps(
            [
                {
                    "ID": "1",
                    "term": "alpha",
                    "type": ["noun", "x >= y"],
                    "sentence": "Alpha occurs.",
                }
            ]
        )
    )
    triples = parser.extract_triples()
    assert ("terms:1", "kroma:hasType", "types:x_gte_y") in triples
    assert parser.extract_node_info("terms:1")["labels"] == ["alpha"]
    assert parser.extract_node_info("1")["data_properties"]["kroma:hasContext"] == [
        "Alpha occurs."
    ]


@pytest.mark.parametrize("text", ["{}", "[1]", "{bad json"])
def test_json_adapter_reports_invalid_datasets(text: str) -> None:
    with pytest.raises(ParseError, match="JSON"):
        JSONParser().parse_text(text, source="dataset.json")


def test_text_adapter_ignores_blank_lines_and_encodes_types() -> None:
    parser = TextParser().parse_text("\n noun \n\n x >= y \n")
    assert parser.lines == ["noun", "x >= y"]
    assert ("types:x_gte_y", "rdf:type", "kroma:TermType") in parser.extract_triples()
    assert parser.extract_node_info("types:noun")["labels"] == ["noun"]


def test_facade_selects_configured_adapter_and_metadata(tmp_path) -> None:
    source = tmp_path / "ontology.data"
    source.write_text(TURTLE, encoding="utf-8")
    config = ParserConfig.from_dict(
        {
            "files": [
                {
                    "pattern": "*.data",
                    "format": "turtle",
                    "name": "configured",
                    "version": "3.0",
                    "namespaces": {"ex": "https://example.test/"},
                }
            ]
        }
    )
    parser = Parser(source, config=config)
    assert parser.format == "turtle"
    assert isinstance(parser.parser, RDFParser)
    assert parser.parser.name == "configured"
    assert parser.parser.version == "3.0"
    assert parser.qname("https://example.test/Child") == "ex:Child"
    assert dict(parser.parser.graph.namespaces())["ontology"] == URIRef(
        "https://example.test/ontology#"
    )


def test_explicit_format_handles_unknown_extension(tmp_path) -> None:
    source = tmp_path / "ontology.data"
    source.write_text(TURTLE, encoding="utf-8")
    assert Parser(source, format="ttl").format == "turtle"


def test_compressed_rdf_json_and_text_inputs(tmp_path) -> None:
    turtle_path = tmp_path / "ontology.ttl.gz"
    with gzip.open(turtle_path, "wt", encoding="utf-8") as stream:
        stream.write(TURTLE)
    assert len(Parser(turtle_path).extract_triples()) > 0

    json_path = tmp_path / "dataset.json.bz2"
    with bz2.open(json_path, "wt", encoding="utf-8") as stream:
        json.dump([{"ID": "1", "term": "alpha"}], stream)
    assert isinstance(Parser(json_path).parser, JSONParser)

    text_path = tmp_path / "types.txt.gz"
    with gzip.open(text_path, "wt", encoding="utf-8") as stream:
        stream.write("noun\nverb\n")
    assert Parser(text_path).parser.lines == ["noun", "verb"]


def test_missing_input_has_contextual_parse_error(tmp_path) -> None:
    missing = tmp_path / "missing.ttl"
    with pytest.raises(ParseError, match=r"Input does not exist.*missing.ttl"):
        Parser(missing).extract_triples()
