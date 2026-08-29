"""Render measured result tables into protected blocks in the final report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import PROJECT_ROOT


REPORT_PATH = PROJECT_ROOT / "REPORT.md"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"


def _replace_block(text: str, name: str, body: str) -> str:
    start = f"<!-- BEGIN AUTO-GENERATED {name} -->"
    end = f"<!-- END AUTO-GENERATED {name} -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise RuntimeError(f"REPORT.md must contain exactly one {name} marker pair.")
    prefix, remainder = text.split(start, maxsplit=1)
    _, suffix = remainder.split(end, maxsplit=1)
    return f"{prefix}{start}\n{body.rstrip()}\n{end}{suffix}"


def _team_table(frame: pd.DataFrame) -> str:
    header = (
        "| Official model | Accuracy | Error Rate | Precision (`yes`) | Recall (`yes`) "
        "| F1 (`yes`) | ROC-AUC | Depth | Leaves | Nodes |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    rows = [
        f"| {row.Model} | {row.accuracy:.6f} | {row.error_rate:.6f} | "
        f"{row.precision:.6f} | {row.recall:.6f} | {row.f1_score:.6f} | "
        f"{row.roc_auc:.6f} | {int(row.depth)} | {int(row.leaves):,} | "
        f"{int(row.nodes):,} |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, *rows])


def _precall_table(frame: pd.DataFrame) -> str:
    header = (
        "| Configuration | Feature set | Accuracy | Error Rate | Precision (`yes`) | "
        "Recall (`yes`) | F1 (`yes`) | ROC-AUC | Depth | Leaves |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    rows = [
        f"| {row.model} | {row.feature_set} | {row.accuracy:.6f} | "
        f"{row.error_rate:.6f} | {row.precision:.6f} | {row.recall:.6f} | "
        f"{row.f1_score:.6f} | {row.roc_auc:.6f} | {int(row.depth)} | "
        f"{int(row.leaves):,} |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, *rows])


def _uncertainty_table(frame: pd.DataFrame) -> str:
    header = (
        "| Model | Metric | Point estimate | 95% CI |\n"
        "| --- | --- | ---: | ---: |"
    )
    rows = [
        f"| {row.model} | {row.metric.replace('_', ' ').title()} | "
        f"{row.point_estimate:.6f} | [{row.ci_lower:.6f}, {row.ci_upper:.6f}] |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, *rows])


def render_generated_report_tables(
    *,
    report_path: str | Path = REPORT_PATH,
    results_dir: str | Path = RESULTS_DIR,
) -> Path:
    """Replace all generated blocks using current CSV artifacts."""

    report = Path(report_path).resolve()
    results = Path(results_dir).resolve()
    text = _render_generated_report_text(
        report.read_text(encoding="utf-8"),
        results,
    )
    report.write_text(text, encoding="utf-8")
    return report


def _render_generated_report_text(text: str, results: Path) -> str:
    """Return synchronized report text without mutating the filesystem."""

    text = _replace_block(
        text,
        "TEAM COMPARISON",
        _team_table(pd.read_csv(results / "team_model_comparison.csv")),
    )
    text = _replace_block(
        text,
        "PRECALL COMPARISON",
        _precall_table(pd.read_csv(results / "precall_duration_sensitivity.csv")),
    )
    text = _replace_block(
        text,
        "UNCERTAINTY",
        _uncertainty_table(pd.read_csv(results / "bootstrap_confidence_intervals.csv")),
    )
    return text


def assert_report_tables_current(
    *,
    report_path: str | Path = REPORT_PATH,
    results_dir: str | Path = RESULTS_DIR,
) -> None:
    """Fail when rendering current artifacts would change the checked report."""

    report = Path(report_path).resolve()
    before = report.read_text(encoding="utf-8")
    expected = _render_generated_report_text(before, Path(results_dir).resolve())
    if before != expected:
        raise AssertionError(
            "REPORT.md contains stale generated metrics; run run_all.py to refresh it."
        )
