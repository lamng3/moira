"""Compact, comparison-first reports for experiment suites."""

from __future__ import annotations

import csv
from html import escape
from pathlib import Path
from typing import Iterable

from .models import ExperimentResult

QUALITY_METRICS = ("precision", "recall", "f1")
COLORS = {"precision": "#2563eb", "recall": "#059669", "f1": "#7c3aed"}


def flatten_metrics(
    value: object, prefix: str = ""
) -> dict[str, str | int | float | bool | None]:
    rows: dict[str, str | int | float | bool | None] = {}
    if isinstance(value, dict):
        for key, item in sorted(value.items()):
            nested = f"{prefix}.{key}" if prefix else str(key)
            rows.update(flatten_metrics(item, nested))
    elif value is None or isinstance(value, (str, int, float, bool)):
        rows[prefix] = value
    return rows


def _quality(result: ExperimentResult) -> dict[str, float]:
    flat = flatten_metrics(result.metrics)
    quality: dict[str, float] = {}
    aliases = {"f1": ("f1", "f1_score")}
    for name in QUALITY_METRICS:
        endings = aliases.get(name, (name,))
        candidates = [
            value
            for key, value in flat.items()
            if any(key == ending or key.endswith(f".{ending}") for ending in endings)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ]
        if candidates:
            quality[name] = float(candidates[0])
    return quality


class SuiteReporter:
    """Render one concise overview while keeping raw run artifacts available."""

    def write(self, root: Path, results: Iterable[ExperimentResult]) -> None:
        materialized = list(results)
        root.mkdir(parents=True, exist_ok=True)
        self._write_csv(root / "metrics.csv", materialized)
        chart_written = self._write_chart(
            root / "figures" / "quality.svg", materialized
        )
        (root / "summary.md").write_text(
            self._markdown(materialized, chart_written), encoding="utf-8"
        )

    @staticmethod
    def _write_csv(path: Path, results: list[ExperimentResult]) -> None:
        rows: list[dict[str, object]] = []
        metric_names: set[str] = set()
        for result in results:
            metrics = flatten_metrics(result.metrics)
            metric_names.update(metrics)
            rows.append(
                {
                    "experiment": result.name,
                    "status": result.status,
                    "reproducible": result.reproducible,
                    "duration_seconds": result.duration_seconds,
                    "run_id": result.run_id,
                    "result_fingerprint": result.result_fingerprint,
                    **metrics,
                }
            )
        fields = [
            "experiment",
            "status",
            "reproducible",
            "duration_seconds",
            *sorted(metric_names),
            "run_id",
            "result_fingerprint",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _markdown(results: list[ExperimentResult], chart_written: bool) -> str:
        lines = [
            "# Experiment results",
            "",
            "> Comparison first; raw logs and provenance remain under `runs/`.",
            "",
            "## Summary",
            "",
            "| Experiment | Status | Precision | Recall | F1 | Time | Repeatable |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for result in results:
            quality = _quality(result)
            repeatable = (
                "yes"
                if result.reproducible is True
                else "no"
                if result.reproducible is False
                else "first run"
            )
            latest = f"{result.name}.latest.md"
            lines.append(
                f"| [{result.name}]({latest}) | {result.status} | "
                f"{_format_metric(quality.get('precision'))} | "
                f"{_format_metric(quality.get('recall'))} | "
                f"{_format_metric(quality.get('f1'))} | "
                f"{result.duration_seconds:.3f}s | {repeatable} |"
            )
        if chart_written:
            lines.extend(
                [
                    "",
                    "## Quality comparison",
                    "",
                    "![Precision, recall, and F1 comparison](figures/quality.svg)",
                ]
            )
        lines.extend(
            [
                "",
                "## Files",
                "",
                "- [`metrics.csv`](metrics.csv): plot-ready scalar metrics.",
                "- [`suite.manifest.json`](suite.manifest.json): suite metadata.",
                "- [`runs/`](runs/): immutable metrics, logs, and provenance.",
                "",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _write_chart(path: Path, results: list[ExperimentResult]) -> bool:
        series = [(result.name, _quality(result)) for result in results]
        series = [(name, values) for name, values in series if values]
        if len(series) < 2:
            path.unlink(missing_ok=True)
            return False

        width = max(640, 180 + 150 * len(series))
        height = 360
        left, top, bottom = 64, 44, 70
        plot_height = height - top - bottom
        plot_width = width - left - 32
        group_width = plot_width / len(series)
        bar_width = min(28, group_width / 5)
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img">',
            "<title>Precision, recall, and F1 by experiment</title>",
            '<rect width="100%" height="100%" fill="white"/>',
            '<g font-family="system-ui, sans-serif" fill="#1f2937">',
            '<text x="24" y="25" font-size="16" font-weight="600">'
            "Quality comparison</text>",
        ]
        for tick in range(5):
            value = tick / 4
            y = top + plot_height * (1 - value)
            svg.extend(
                [
                    f'<line x1="{left}" y1="{y:.1f}" x2="{width - 32}" '
                    f'y2="{y:.1f}" stroke="#e5e7eb"/>',
                    f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
                    f'font-size="11">{value:.2f}</text>',
                ]
            )
        for index, (name, values) in enumerate(series):
            center = left + group_width * (index + 0.5)
            for offset, metric in enumerate(QUALITY_METRICS):
                value = max(0.0, min(1.0, values.get(metric, 0.0)))
                x = center + (offset - 1) * (bar_width + 4) - bar_width / 2
                y = top + plot_height * (1 - value)
                svg.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                    f'height="{plot_height * value:.1f}" '
                    f'fill="{COLORS[metric]}"><title>{escape(name)} '
                    f'{metric}: {value:.4f}</title></rect>'
                )
            svg.append(
                f'<text x="{center:.1f}" y="{height - 44}" text-anchor="middle" '
                f'font-size="12">{escape(name)}</text>'
            )
        legend_x = width - 260
        for index, metric in enumerate(QUALITY_METRICS):
            x = legend_x + index * 82
            svg.extend(
                [
                    f'<rect x="{x}" y="15" width="12" height="12" '
                    f'fill="{COLORS[metric]}"/>',
                    f'<text x="{x + 17}" y="26" font-size="11">'
                    f"{metric.upper()}</text>",
                ]
            )
        svg.extend(["</g>", "</svg>", ""])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(svg), encoding="utf-8")
        return True


def _format_metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"
