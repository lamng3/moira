"""The installable package exposes the integration import."""

from importlib.metadata import version

import agentoi
from agentoi import OntologySummary, OntologyWorkspace


def test_public_import_matches_the_installed_version() -> None:
    assert version("agentoi") == agentoi.__version__
    assert OntologyWorkspace is agentoi.OntologyWorkspace
    assert OntologySummary is agentoi.OntologySummary
