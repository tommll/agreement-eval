"""Figures for the report.

Four charts, each answering one question a reader will ask:
    agreement_vs_accuracy - is a higher agreement bucket actually more correct?
    calibration           - can an agreement rate be read as an accuracy?
    aggregators           - does Dawid-Skene beat majority vote?
    per_field_trust       - where does the trust rule hold, and where not?

Palette: two categorical slots (blue, orange), validated for colour-vision
deficiency separation; recessive grid and axes; values direct-labelled so
identity never rests on colour alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis import AnalysisResult  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834")  # slot 1 blue, slot 2 orange


def _style(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _legend_above(ax: plt.Axes, handles=None, ncol: int = 2) -> None:
    """Legend outside the plot area - inside, it lands on the tallest bars."""
    kwargs = dict(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncol=ncol,
                  loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0)
    ax.legend(handles=handles, **kwargs) if handles else ax.legend(**kwargs)


def _figure(width: float = 8.0, height: float = 4.5):
    fig, ax = plt.subplots(figsize=(width, height), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_agreement_buckets(buckets: pd.DataFrame, path: Path) -> Optional[Path]:
    """P(consensus correct) by agreement bucket, populated vs null gold."""
    splits = [s for s in ("gold_populated=True", "gold_populated=False") if s in set(buckets["split"])]
    if not splits:
        return None
    categories = [c for c in buckets[buckets["split"] == splits[0]]["agreement_bucket"].tolist()]
    fig, ax = _figure()
    width = 0.38
    positions = np.arange(len(categories))
    labels = {"gold_populated=True": "gold populated", "gold_populated=False": "gold null"}

    for i, split in enumerate(splits):
        sub = buckets[buckets["split"] == split].set_index("agreement_bucket")
        values = [sub["p_correct"].get(c, np.nan) for c in categories]
        counts = [sub["n"].get(c, 0) for c in categories]
        offset = (i - (len(splits) - 1) / 2) * width
        # Buckets with no items get no bar at all - an empty bucket is not a
        # zero rate, and drawing one invites exactly that misreading.
        drawn = [v if int(n) > 0 else np.nan for v, n in zip(values, counts)]
        bars = ax.bar(positions + offset, drawn, width * 0.94, label=labels.get(split, split),
                      color=SERIES[i], linewidth=0)
        for bar, value, n in zip(bars, drawn, counts):
            if np.isnan(value):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025,
                    f"{value:.0%}\nn={int(n)}", ha="center", va="bottom",
                    fontsize=8, color=INK_SECONDARY, linespacing=1.4)

    ax.set_xticks(positions)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.3)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, 0.25)])
    _style(ax, "Correctness by agreement level", "pairwise agreement between configs",
           "P(consensus correct)")
    _legend_above(ax)
    return _save(fig, path)


def plot_calibration(calibration: pd.DataFrame, path: Path) -> Optional[Path]:
    """Observed accuracy against the agreement rate, with the identity line."""
    if calibration.empty:
        return None
    fig, ax = _figure(6.4, 4.8)
    ax.plot([0, 1], [0, 1], linestyle=(0, (4, 4)), linewidth=1.5, color=BASELINE,
            label="agreement read as accuracy")
    ax.plot(calibration["mean_agreement"], calibration["observed_accuracy"],
            linewidth=2.0, marker="o", markersize=8, color=SERIES[0], label="observed")
    for _, row in calibration.iterrows():
        ax.annotate(f"n={int(row['n'])}", (row["mean_agreement"], row["observed_accuracy"]),
                    textcoords="offset points", xytext=(6, -12), fontsize=7, color=MUTED)
    ax.set_xlim(-0.04, 1.12)
    ax.set_ylim(-0.03, 1.08)
    for axis in ("x", "y"):
        getattr(ax, f"set_{axis}ticks")(np.arange(0, 1.01, 0.25))
        getattr(ax, f"set_{axis}ticklabels")([f"{v:.0%}" for v in np.arange(0, 1.01, 0.25)])
    _style(ax, "Is agreement calibrated as an accuracy estimate?",
           "mean pairwise agreement", "observed accuracy")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower right")
    return _save(fig, path)


def plot_aggregators(comparison: pd.DataFrame, path: Path) -> Optional[Path]:
    """Accuracy of each way of combining the configs."""
    if comparison.empty:
        return None
    # The binary (null/populated only) rows score a different, easier task -
    # putting them on this axis would invite a comparison that is not one.
    # They stay in aggregator_comparison.csv.
    df = comparison[~comparison["method"].str.contains("null/populated only")]
    df = df.dropna(subset=["accuracy"]).sort_values("accuracy")
    fig, ax = _figure(8.0, 0.42 * len(df) + 1.8)
    is_single = df["method"].str.startswith("single:")
    colors = [SERIES[1] if single else SERIES[0] for single in is_single]
    bars = ax.barh(np.arange(len(df)), df["accuracy"], height=0.62, color=colors, linewidth=0)
    for bar, value in zip(bars, df["accuracy"]):
        ax.text(value + 0.005, bar.get_y() + bar.get_height() / 2, f"{value:.3f}",
                va="center", fontsize=8, color=INK_SECONDARY)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["method"], fontsize=9)
    ax.set_xlim(0, min(1.05, float(df["accuracy"].max()) + 0.08))
    _style(ax, "Aggregation methods on identical items", "item accuracy (exact value match)", "")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], color=SERIES[0], linewidth=8, label="aggregated"),
               plt.Line2D([], [], color=SERIES[1], linewidth=8, label="single config")]
    # One empty row of headroom so the legend sits above the longest bar
    # instead of on top of it.
    ax.set_ylim(-0.8, len(df) - 0.2 + 0.9)
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncol=2,
              loc="upper right", borderaxespad=0.2)
    return _save(fig, path)


def plot_per_field(per_field: pd.DataFrame, path: Path) -> Optional[Path]:
    """Where unanimity buys accuracy, and where it buys nothing."""
    if per_field.empty:
        return None
    df = per_field.sort_values("field")
    fig, ax = _figure(max(7.0, 1.3 * len(df) + 2.5), 4.6)
    positions = np.arange(len(df))
    width = 0.38
    bars_a = ax.bar(positions - width / 2, df["p_correct_when_unanimous"], width * 0.94,
                    color=SERIES[0], linewidth=0, label="unanimous items")
    bars_b = ax.bar(positions + width / 2, df["base_rate"], width * 0.94,
                    color=SERIES[1], linewidth=0, label="all items (base rate)")
    for bars in (bars_a, bars_b):
        for bar in bars:
            height = bar.get_height()
            if np.isnan(height):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.015, f"{height:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color=INK_SECONDARY)
    ax.set_xticks(positions)
    ax.set_xticklabels(df["field"], fontsize=9)
    ax.set_ylim(0, 1.14)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, 0.25)])
    _style(ax, "Does unanimity beat the base rate, field by field?", "", "P(consensus correct)")
    _legend_above(ax)
    return _save(fig, path)


def write_plots(result: AnalysisResult, outdir: Path) -> List[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    tables: Dict[str, pd.DataFrame] = result.tables

    jobs = (
        (plot_agreement_buckets, "agreement_buckets", "agreement_vs_accuracy.png"),
        (plot_calibration, "calibration", "calibration.png"),
        (plot_aggregators, "aggregator_comparison", "aggregators.png"),
        (plot_per_field, "per_field_trust", "per_field_trust.png"),
    )
    for fn, table_name, filename in jobs:
        frame = tables.get(table_name)
        if frame is None or frame.empty:
            continue
        path = fn(frame, outdir / filename)
        if path is not None:
            written.append(path)
    return written
