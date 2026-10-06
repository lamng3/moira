import io
import pickle

from moira import legacy_pickle
from moira.routing.harness import Noul


def test_loads_pickle_written_under_old_package_name():
    current = pickle.dumps(Noul(instructions="same concept?"), protocol=0)
    old = current.replace(b"moira.routing.harness", b"agentoi.routing.harness")
    assert old != current

    loaded = legacy_pickle.load(io.BytesIO(old))

    assert loaded == Noul(instructions="same concept?")


def test_loads_current_pickles_unchanged():
    data = pickle.dumps(Noul(instructions="same concept?"))

    assert legacy_pickle.load(io.BytesIO(data)) == Noul(instructions="same concept?")
