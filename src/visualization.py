"""Generate reproducible static figures for the Bank Marketing EDA report."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Keep Matplotlib's runtime cache in a cross-platform writable temporary folder.
_MATPLOTLIB_CONFIG_DIR = Path(tempfile.gettempdir()) / "lab2-matplotlib"
_MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.data import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PROJECT_ROOT,
    TARGET_COLUMN,
    DatasetSplit,
)


DEFAULT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
COLOR_NO = "#2563EB"
COLOR_YES = "#F59E0B"
COLOR_NEUTRAL = "#64748B"
COLOR_MEDIAN = "#DC2626"


def _figure_path(output_dir: str | Path, filename: str) -> Path:
    """Resolve an output filename and ensure its parent directory exists.

    Args:
        output_dir: Directory in which the rendered PNG should be stored.
        filename: Basename for one figure. Callers provide a fixed ``.png`` name
            rather than a nested path.

    Returns:
        An absolute :class:`~pathlib.Path` ready to be passed to ``savefig``.

    Raises:
        OSError: If the output directory cannot be created.
    """

    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def _save_figure(figure: Figure, path: Path) -> Path:
    """Write a Matplotlib figure as a report-ready PNG and release its memory.

    Args:
        figure: Fully configured Matplotlib figure.
        path: Destination PNG path, normally created by :func:`_figure_path`.

    Returns:
        The same absolute destination path after the image has been written.

    Raises:
        OSError: If the destination cannot be written.
        ValueError: If Matplotlib cannot infer or use the requested image format.
    """

    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def plot_target_distribution(
    frame: pd.DataFrame,
    output_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> Path:
    """Plot target counts and percentages to expose class imbalance.

    Args:
        frame: Bank Marketing DataFrame containing the original string target
            column ``y`` with ``no`` and ``yes`` labels.
        output_dir: Directory in which ``target-distribution.png`` is written.

    Returns:
        Absolute path to the generated PNG.

    Raises:
        KeyError: If the target column is absent.
        OSError: If the output directory or PNG cannot be written.
    """

    counts = frame[TARGET_COLUMN].value_counts().reindex(["no", "yes"])
    percentages = counts.div(counts.sum()).mul(100.0)

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    bars = axis.bar(
        counts.index,
        counts.values,
        color=[COLOR_NO, COLOR_YES],
        width=0.58,
    )
    axis.set_title("Target Distribution: Term-deposit Subscription", pad=14)
    axis.set_xlabel("Target class (y)")
    axis.set_ylabel("Number of observations")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.set_ylim(0, float(counts.max()) * 1.17)

    for bar, label in zip(bars, counts.index, strict=True):
        count = int(counts[label])
        percentage = float(percentages[label])
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.025,
            f"{count:,}\n({percentage:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    figure.tight_layout()
    return _save_figure(
        figure,
        _figure_path(output_dir, "target-distribution.png"),
    )


def plot_numeric_distributions(
    frame: pd.DataFrame,
    output_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> Path:
    """Plot small-multiple histograms for all seven numerical predictors.

    Each plotted series is clipped to its 1st and 99th percentiles so the dense
    center remains readable despite extreme tails. Clipping affects only this
    visualization; the EDA calculations and model pipeline retain original
    values. A dashed line marks the true, unclipped median of each feature.

    Args:
        frame: DataFrame containing all names in :data:`NUMERICAL_FEATURES`.
        output_dir: Directory in which ``numeric-distributions.png`` is written.

    Returns:
        Absolute path to the generated PNG.

    Raises:
        KeyError: If an expected numerical feature is absent.
        TypeError: If a numerical feature cannot be plotted as numbers.
        OSError: If the output image cannot be written.
    """

    figure, axes = plt.subplots(2, 4, figsize=(15.2, 8.0))
    flattened_axes = axes.flatten()

    for axis, column in zip(flattened_axes, NUMERICAL_FEATURES, strict=False):
        values = frame[column].dropna()
        lower, upper = values.quantile([0.01, 0.99])
        displayed = values.clip(lower=lower, upper=upper)
        median = float(values.median())

        axis.hist(displayed, bins=35, color=COLOR_NEUTRAL, alpha=0.88)
        axis.axvline(
            median,
            color=COLOR_MEDIAN,
            linestyle="--",
            linewidth=1.4,
            label=f"Median: {median:,.1f}",
        )
        axis.set_title(column)
        axis.set_xlabel("Value (clipped to p01–p99)")
        axis.set_ylabel("Frequency")
        axis.grid(axis="y", alpha=0.20)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, fontsize=8, loc="upper right")

    for axis in flattened_axes[len(NUMERICAL_FEATURES) :]:
        axis.axis("off")

    figure.suptitle("Numerical Feature Distributions", fontsize=15, y=1.01)
    figure.text(
        0.5,
        0.005,
        "Display is clipped to the 1st–99th percentile only; source data and "
        "reported statistics remain unchanged.",
        ha="center",
        fontsize=9,
        color="#334155",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.98))
    return _save_figure(
        figure,
        _figure_path(output_dir, "numeric-distributions.png"),
    )


def plot_categorical_overview(
    frame: pd.DataFrame,
    output_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> Path:
    """Compare categorical cardinality with the prevalence of ``unknown``.

    Args:
        frame: DataFrame containing all names in :data:`CATEGORICAL_FEATURES`.
        output_dir: Directory in which ``categorical-overview.png`` is written.

    Returns:
        Absolute path to the generated two-panel PNG.

    Raises:
        KeyError: If an expected categorical feature is absent.
        OSError: If the output image cannot be written.
    """

    categorical = frame.loc[:, list(CATEGORICAL_FEATURES)]
    cardinality = categorical.nunique(dropna=False)
    unknown_percentage = categorical.eq("unknown").mean().mul(100.0)
    positions = np.arange(len(CATEGORICAL_FEATURES))

    figure, (axis_categories, axis_unknown) = plt.subplots(
        1,
        2,
        figsize=(14.5, 6.2),
        sharey=True,
    )

    category_bars = axis_categories.barh(
        positions,
        cardinality.values,
        color=COLOR_NO,
        alpha=0.88,
    )
    axis_categories.set_yticks(positions, labels=CATEGORICAL_FEATURES)
    axis_categories.invert_yaxis()
    axis_categories.set_xlabel("Number of distinct categories")
    axis_categories.set_title("Categorical Cardinality")
    axis_categories.grid(axis="x", alpha=0.22)
    axis_categories.set_axisbelow(True)
    axis_categories.bar_label(category_bars, padding=3, fmt="%.0f")

    unknown_bars = axis_unknown.barh(
        positions,
        unknown_percentage.values,
        color=COLOR_YES,
        alpha=0.88,
    )
    axis_unknown.set_xlabel("Observations equal to 'unknown' (%)")
    axis_unknown.set_title("Explicit Unknown-category Prevalence")
    axis_unknown.grid(axis="x", alpha=0.22)
    axis_unknown.set_axisbelow(True)
    axis_unknown.set_xlim(0, max(100.0, float(unknown_percentage.max()) * 1.1))
    axis_unknown.bar_label(unknown_bars, padding=3, fmt="%.1f%%")

    figure.suptitle("Categorical Feature Overview", fontsize=15, y=1.01)
    figure.tight_layout()
    return _save_figure(
        figure,
        _figure_path(output_dir, "categorical-overview.png"),
    )


def plot_split_class_balance(
    frame: pd.DataFrame,
    split: DatasetSplit,
    output_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> Path:
    """Visualize class proportions before and after the stratified split.

    Args:
        frame: Original DataFrame containing string target labels.
        split: Encoded train/test partition whose ``0`` and ``1`` proportions are
            compared with the full dataset.
        output_dir: Directory in which ``stratified-split-balance.png`` is written.

    Returns:
        Absolute path to the generated 100-percent stacked-bar PNG.

    Raises:
        KeyError: If the original target column is absent.
        OSError: If the output image cannot be written.
    """

    full_encoded = frame[TARGET_COLUMN].map({"no": 0, "yes": 1})
    partitions = {
        "Full dataset": full_encoded,
        "Train": split.y_train,
        "Test": split.y_test,
    }
    no_percentages = np.array(
        [float(target.eq(0).mean() * 100.0) for target in partitions.values()]
    )
    yes_percentages = 100.0 - no_percentages
    positions = np.arange(len(partitions))

    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    axis.barh(positions, no_percentages, color=COLOR_NO, label="no / 0")
    axis.barh(
        positions,
        yes_percentages,
        left=no_percentages,
        color=COLOR_YES,
        label="yes / 1",
    )
    axis.set_yticks(positions, labels=partitions.keys())
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("Class share (%)")
    axis.set_title("Class Balance Preserved by Stratified Train/Test Split", pad=12)
    axis.grid(axis="x", alpha=0.20)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2)

    for position, (no_share, yes_share) in enumerate(
        zip(no_percentages, yes_percentages, strict=True)
    ):
        axis.text(
            no_share / 2,
            position,
            f"{no_share:.2f}%",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
        axis.text(
            no_share + yes_share / 2,
            position,
            f"{yes_share:.2f}%",
            ha="center",
            va="center",
            color="#111827",
            fontweight="bold",
        )

    figure.tight_layout()
    return _save_figure(
        figure,
        _figure_path(output_dir, "stratified-split-balance.png"),
    )


def generate_eda_figures(
    frame: pd.DataFrame,
    split: DatasetSplit,
    output_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> list[Path]:
    """Generate the complete static EDA figure set in a deterministic order.

    Existing files with the four standard names are overwritten, allowing the
    figures to stay synchronized with the source CSV and split configuration.
    No model-training or evaluation figures are created by this function.

    Args:
        frame: Validated full Bank Marketing DataFrame.
        split: Stratified train/test partition used by the group experiments.
        output_dir: Directory receiving the four PNG files.

    Returns:
        Absolute paths, in report order, for target distribution, numerical
        distributions, categorical overview, and stratified-split balance.

    Raises:
        KeyError: If required columns are missing from ``frame``.
        OSError: If the output directory or any PNG cannot be written.
    """

    return [
        plot_target_distribution(frame, output_dir),
        plot_numeric_distributions(frame, output_dir),
        plot_categorical_overview(frame, output_dir),
        plot_split_class_balance(frame, split, output_dir),
    ]
