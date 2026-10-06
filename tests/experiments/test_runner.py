import json
import sys
from pathlib import Path

import pytest

from moira.application import _write_experiment_metrics
from experiments.cli import main
from experiments.models import ConfigurationError, ExperimentSpec, ExperimentSuite
from experiments.repository import JsonResultRepository
from experiments.runner import ExperimentRunner
from experiments.strategies import ExperimentStrategyFactory


def test_suite_rejects_duplicate_names():
    with pytest.raises(ConfigurationError, match="unique"):
        ExperimentSuite.from_dict(
            {
                "experiments": [
                    {"name": "same", "ontology": ["a"]},
                    {"name": "same", "ontology": ["b"]},
                ]
            }
        )


def test_standard_strategy_builds_deterministic_command(tmp_path):
    spec = ExperimentSpec(
        name="baseline",
        ontology=["envo", "sweet"],
        options={"top_k": 10, "refine": True, "similarity": False},
    )

    command = ExperimentStrategyFactory.create("standard").build_command(
        spec, tmp_path
    )

    assert command == [
        sys.executable,
        "-m",
        "moira.cli",
        "run",
        "--ontology",
        "envo",
        "sweet",
        "--refine",
        "--top_k",
        "10",
    ]


def test_dry_run_writes_manifests_without_executing(tmp_path):
    suite = ExperimentSuite.from_dict(
        {
            "results_dir": "results",
            "experiments": [
                {
                    "name": "smoke",
                    "kind": "command",
                    "options": {"command": ["{python}", "-c", "raise SystemExit(9)"]},
                }
            ],
        }
    )
    repository = JsonResultRepository(tmp_path / "results")
    runner = ExperimentRunner(tmp_path, repository, dry_run=True)

    results = runner.run_suite(suite)

    assert results[0].status == "dry-run"
    manifest = json.loads((tmp_path / "results/suite.manifest.json").read_text())
    assert manifest["experiments"][0]["name"] == "smoke"


def test_command_experiment_records_output(tmp_path):
    suite = ExperimentSuite.from_dict(
        {
            "experiments": [
                {
                    "name": "smoke",
                    "kind": "command",
                    "options": {"command": ["{python}", "-c", "print('ok')"]},
                }
            ]
        }
    )
    repository = JsonResultRepository(tmp_path / "results")

    result = ExperimentRunner(tmp_path, repository).run_suite(suite)[0]

    assert result.status == "passed"
    assert result.command[0] == "{python}"
    assert (tmp_path / result.output_file).read_text().strip() == "ok"
    assert (tmp_path / result.artifact_directory / "manifest.json").is_file()
    assert (tmp_path / result.artifact_directory / "summary.md").is_file()
    assert (tmp_path / "results/smoke.latest.json").is_file()
    assert (tmp_path / "results/summary.md").is_file()


def test_metrics_are_fingerprinted_and_compared(tmp_path):
    metrics_code = (
        "import json, os; "
        "json.dump({'alignment': {'precision': 0.8, 'recall': 0.5, "
        "'f1': 0.6154}}, open(os.environ['MOIRA_METRICS_PATH'], 'w'))"
    )
    suite = ExperimentSuite.from_dict(
        {
            "experiments": [
                {
                    "name": "repeatable",
                    "kind": "command",
                    "options": {"command": ["{python}", "-c", metrics_code]},
                }
            ]
        }
    )
    repository = JsonResultRepository(tmp_path / "results")
    runner = ExperimentRunner(tmp_path, repository)

    first = runner.run_suite(suite)[0]
    second = runner.run_suite(suite)[0]

    assert first.metrics["alignment"]["f1"] == 0.6154
    assert first.result_fingerprint == second.result_fingerprint
    assert first.reproducible is None
    assert second.reproducible is True
    assert first.run_id != second.run_id
    summary = (tmp_path / "results/summary.md").read_text()
    assert "| Precision | Recall | F1 |" in summary
    assert "![Precision, recall, and F1 comparison]" not in summary
    assert (tmp_path / "results/metrics.csv").is_file()


def test_suite_chart_compares_multiple_experiments(tmp_path):
    metrics_code = (
        "import json, os; "
        "json.dump({'alignment': {'precision': 0.8, 'recall': 0.5, "
        "'f1': 0.6154}}, open(os.environ['MOIRA_METRICS_PATH'], 'w'))"
    )
    suite = ExperimentSuite.from_dict(
        {
            "experiments": [
                {
                    "name": name,
                    "kind": "command",
                    "options": {"command": ["{python}", "-c", metrics_code]},
                }
                for name in ("baseline", "refined")
            ]
        }
    )
    repository = JsonResultRepository(tmp_path / "results")

    ExperimentRunner(tmp_path, repository).run_suite(suite)

    summary = (tmp_path / "results/summary.md").read_text()
    assert "![Precision, recall, and F1 comparison]" in summary
    assert (tmp_path / "results/figures/quality.svg").is_file()


def test_standard_experiment_gets_isolated_output_path(tmp_path):
    spec = ExperimentSpec(name="baseline", ontology=["example.owl"])
    result = ExperimentRunner(
        tmp_path,
        JsonResultRepository(tmp_path / "results"),
        dry_run=True,
    ).run_experiment(spec)

    assert "--output" in result.command
    output = Path(result.command[result.command.index("--output") + 1])
    assert result.run_id in str(output)


def test_pipeline_metrics_are_exported_for_experiment_runner(tmp_path, monkeypatch):
    destination = tmp_path / "metrics.json"
    monkeypatch.setenv("MOIRA_METRICS_PATH", str(destination))

    _write_experiment_metrics(
        {
            "metrics": {"precision": 0.75, "recall": 0.5, "f1": 0.6},
            "counts": {"TP": 3, "FP": 1, "FN": 3},
            "positives_only_retrieval": {"Hits@k": 0.8, "MRR": 0.7},
            "strict_time_avg": 0.01,
            "llm_time_avg": 0.02,
        },
        {"similarity_avgs_over_gold": {"hybrid": 0.9}},
    )

    metrics = json.loads(destination.read_text())
    assert metrics["alignment"]["f1"] == 0.6
    assert metrics["retrieval"]["MRR"] == 0.7
    assert metrics["similarity"]["hybrid"] == 0.9


def test_cli_reports_invalid_configuration(tmp_path):
    config = tmp_path / "invalid.json"
    config.write_text('{"experiments": []}')

    assert main([str(config), "--dry-run", "--project-root", str(tmp_path)]) == 2


def test_remote_tracker_uses_ephemeral_artifacts(
    tmp_path, monkeypatch, capsys
):
    config = tmp_path / "suite.json"
    config.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "name": "hosted",
                        "kind": "command",
                        "options": {
                            "command": ["{python}", "-c", "print('complete')"]
                        },
                    }
                ]
            }
        )
    )
    published: dict[str, object] = {}

    class FakeTracker:
        is_remote = True

        def publish_suite(self, suite_name, results, artifact_directory):
            published["suite"] = suite_name
            published["directory"] = artifact_directory
            published["summary_exists"] = (
                artifact_directory / "summary.md"
            ).is_file()
            return "https://wandb.example/run"

    monkeypatch.setattr(
        "experiments.cli.create_tracker", lambda *args, **kwargs: FakeTracker()
    )

    exit_code = main(
        [
            str(config),
            "--tracker",
            "wandb",
            "--project-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert published["suite"] == "suite"
    assert published["summary_exists"] is True
    assert not Path(published["directory"]).exists()
    assert "Dashboard: https://wandb.example/run" in capsys.readouterr().out
    assert not (tmp_path / "results").exists()
