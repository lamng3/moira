from datetime import datetime, timezone

from moira.agents.runtime.duckdb_store import list_runs, load_run, save_run


def test_duckdb_round_trip_lists_the_newer_run_first(tmp_path) -> None:
    path = tmp_path / "traces" / "moira.duckdb"
    older = {
        "thoughts": [{"id": "t1", "question": "once"}],
        "edges": [],
    }
    newer = {
        "thoughts": [{"id": "t2", "question": "again"}, {"id": "t3", "question": "next"}],
        "edges": [{"id": "e1", "src": "t2", "dst": "t3"}],
    }

    save_run(
        path,
        run_id="run-old",
        slug="old",
        query="first question",
        graph=older,
        memory_trace={"id": "run-old"},
        start_id="t1",
        end_id="t1",
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )
    save_run(
        path,
        run_id="run-new",
        slug="new",
        query="second question",
        graph=newer,
        memory_trace={"id": "run-new"},
        start_id="t2",
        end_id="t3",
        created_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
    )

    assert list_runs(path) == ["run-new", "run-old"]

    loaded = load_run(path, "run-new")

    assert loaded["run"]["RunId"] == "run-new"
    assert loaded["run"]["Slug"] == "new"
    assert loaded["run"]["StartId"] == "t2"
    assert loaded["run"]["EndId"] == "t3"
    assert loaded["run"]["NodeCount"] == 2
    assert loaded["run"]["EdgeCount"] == 1
    assert loaded["run"]["MemoryTrace"] == {"id": "run-new"}
    assert loaded["graph"] == newer
