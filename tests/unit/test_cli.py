from __future__ import annotations

import json

from moira.cli import main


ONTOLOGY = """
@prefix ex: <https://example.test/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:ontology a owl:Ontology .
ex:Parent a owl:Class ; rdfs:label "Parent" .
ex:Child a owl:Class ;
    rdfs:subClassOf ex:Parent ;
    rdfs:label "Child" .
"""


def test_inspect_accepts_an_ontology_path(tmp_path, capsys) -> None:
    ontology = tmp_path / "example.ttl"
    ontology.write_text(ONTOLOGY, encoding="utf-8")

    assert main(["inspect", str(ontology), "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["path"] == str(ontology)
    assert output["format"] == "turtle"
    assert output["concepts"] == 2
    assert output["relations"] == 1


def test_missing_ontology_returns_a_cli_error(tmp_path) -> None:
    missing = tmp_path / "missing.owl"

    try:
        main(["inspect", str(missing)])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("Expected argparse to report the missing ontology")
