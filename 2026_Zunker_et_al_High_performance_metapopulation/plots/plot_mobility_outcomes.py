#!/usr/bin/env python3
"""Create the multi-panel mobility-representation outcome figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir.parent / "results" / "mobility_outcomes",
        help="Directory containing mobility_outcomes_timeseries.csv and mobility_outcomes_summary.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir.parent / "figures" / "mobility_outcomes",
        help="Output path without extension.",
    )
    return parser.parse_args()


def aggregate_age_groups(timeseries: pd.DataFrame) -> pd.DataFrame:
    columns = ["model", "cycle_days", "time_days", "patch", "row", "column"]
    return (
        timeseries.groupby(columns, as_index=False, sort=True)[["population", "S", "E", "I", "R"]]
        .sum()
        .sort_values(columns)
    )


def global_trajectory(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby(["model", "cycle_days", "time_days"], as_index=False)[["population", "I"]].sum()
    grouped["infected_percent"] = 100.0 * grouped["I"] / grouped["population"]
    return grouped


def main() -> None:
    args = parse_args()
    timeseries_path = args.results_dir / "mobility_outcomes_timeseries.csv"
    summary_path = args.results_dir / "mobility_outcomes_summary.csv"
    if not timeseries_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"Missing experiment output in {args.results_dir}")

    data = aggregate_age_groups(pd.read_csv(timeseries_path))
    summary = pd.read_csv(summary_path)
    global_data = global_trajectory(data)
    explicit = data[data["model"] == "explicit"]
    cycle_lengths = np.sort(explicit["cycle_days"].unique())

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure = plt.figure(figsize=(12.0, 4.8), constrained_layout=True)
    outer_grid = figure.add_gridspec(1, 2)
    left_grid = outer_grid[0].subgridspec(2, 1, height_ratios=(3.0, 1.0), hspace=0.08)
    trajectory_ax = figure.add_subplot(left_grid[0])
    peak_ax = figure.add_subplot(left_grid[1])
    sensitivity_ax = figure.add_subplot(outer_grid[1])

    ax = trajectory_ax
    implicit_global = global_data[global_data["model"] == "implicit"]
    daily_global = global_data[
        (global_data["model"] == "explicit") & np.isclose(global_data["cycle_days"], 1.0)
    ]
    ax.plot(
        implicit_global["time_days"],
        implicit_global["infected_percent"],
        color="#333333",
        linewidth=3.0,
        label="Implicit mobility",
        zorder=1,
    )
    ax.plot(
        daily_global["time_days"],
        daily_global["infected_percent"],
        color="#0072B2",
        linewidth=2.2,
        linestyle="--",
        label=r"Explicit mobility, $\delta=1$",
        zorder=2,
    )
    ax.set_ylabel("Infectious population [%]")
    ax.grid(alpha=0.22)
    ax.set_ylim(top=8.4)
    ax.legend(loc="upper left", frameon=False)
    peak_ax.plot(
        implicit_global["time_days"],
        implicit_global["infected_percent"],
        color="#333333",
        linewidth=1.8,
    )
    peak_ax.plot(
        daily_global["time_days"],
        daily_global["infected_percent"],
        color="#0072B2",
        linewidth=1.5,
        linestyle="--",
    )
    peak_values = list(daily_global.loc[daily_global["time_days"].between(88.0, 94.0), "infected_percent"])
    peak_values.extend(implicit_global.loc[implicit_global["time_days"].between(88.0, 94.0), "infected_percent"])
    peak_ax.set_xlim(88.0, 94.0)
    peak_ax.set_ylim(min(peak_values) - 0.003, max(peak_values) + 0.003)
    peak_ax.set_xlabel("Time [days]")
    peak_ax.set_ylabel("Peak [%]", fontsize=13)
    peak_ax.text(0.01, 0.90, "Peak detail", transform=peak_ax.transAxes, fontsize=11, va="top")
    peak_ax.tick_params(labelsize=11)
    peak_ax.grid(alpha=0.18)

    ax = sensitivity_ax
    explicit_summary = summary[summary["model"] == "explicit"].sort_values("cycle_days")
    patchwise_difference = 100.0 * explicit_summary["max_normalized_patch_l1"]
    ax.plot(
        explicit_summary["cycle_days"],
        patchwise_difference,
        color="#CC79A7",
        marker="o",
        markersize=7,
        linewidth=2.4,
        label="Maximum over output times",
    )
    ax.axvline(1.0, color="0.45", linestyle=":", linewidth=1.2, label="Daily cycle")
    ax.set_xscale("log", base=2)
    ax.set_xticks(cycle_lengths, [f"{cycle:g}" for cycle in cycle_lengths])
    for cycle, value in zip(explicit_summary["cycle_days"], patchwise_difference, strict=True):
        offset = (-10, 10) if np.isclose(cycle, 2.0) else (0, 9)
        alignment = "right" if np.isclose(cycle, 2.0) else "center"
        ax.annotate(
            f"{value:.3f}",
            (cycle, value),
            xytext=offset,
            textcoords="offset points",
            ha=alignment,
            fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.8},
        )
    ax.set_ylim(bottom=0.0, top=1.12 * patchwise_difference.max())
    ax.set_xlabel(r"Mobility-cycle length $\delta$ [days]")
    ax.set_ylabel("Patchwise $L^1$ difference\n[% of implicit peak]")
    ax.grid(alpha=0.22, which="both")
    ax.legend(frameon=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output.with_suffix(".png")
    pdf_path = args.output.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
