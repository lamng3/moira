"""The installable package exposes the integration import."""

from importlib.metadata import version

import moira
from moira import OntologySummary, OntologyWorkspace


def test_public_import_matches_the_installed_version() -> None:
    assert version("moira") == moira.__version__
    assert OntologyWorkspace is moira.OntologyWorkspace
    assert OntologySummary is moira.OntologySummary
