#!/usr/bin/env python3
"""Plot the CPU rooflines from the exported model-level LIKWID summary.

Reads model_summary.csv (medians and quartiles of three repetitions) and the
measured CPU reference limits in the analysis.json.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from plot_runtime_comparison import PAPER, set_style


DATA = PAPER / "results" / "roofline_cpu"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(directory):
    with (directory / "model_summary.csv").open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["backend"] == "cpu"]
    for row in rows:
        for key, value in row.items():
            if key.startswith(("median_", "q1_", "q3_")):
                row[key] = float(value)
    reference = json.loads((directory / "analysis.json").read_text())["ceilings"]["cpu"]
    return rows, reference


def make_figure(rows, reference, output):
    set_style()
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
                         "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.26, top=0.89, wspace=0.13)
    colors = {"implicit": "#333333", "explicit": "#0072B2"}
    markers = {"1024": "o", "8192": "^"}
    x = np.geomspace(0.07, 40, 300)
    for ax, steps in zip(axes, ("1", "32")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(x, np.minimum(reference["bandwidth_gbs"] * x, reference["peak_gflops"]),
                color="#888888", lw=1.6)
        ax.text(38, reference["peak_gflops"] * 1.16, "FP64: 4067 GFLOP/s",
                ha="right", fontsize=11, color="#555555")
        ax.text(0.13, 76, "Memory: 337 GB/s", rotation=39, fontsize=11, color="#555555")
        for row in rows:
            if row["half_day_steps"] != steps:
                continue
            ai, rate = row["median_arithmetic_intensity"], row["median_gflops"]
            require(0.07 < ai < 40 and 8 < rate < 7500, "Point outside figure axes")
            color = colors[row["model"]]
            ax.errorbar(ai, rate,
                        xerr=[[ai - row["q1_arithmetic_intensity"]],
                              [row["q3_arithmetic_intensity"] - ai]],
                        yerr=[[rate - row["q1_gflops"]], [row["q3_gflops"] - rate]],
                        fmt=markers[row["patches"]], color=color, ms=9.5, capsize=2.5,
                        mfc=color, mec="white", mew=1.0, zorder=3)
        ax.set(xlim=(0.07, 40), ylim=(8, 7500), xlabel="Arithmetic intensity [FLOP/byte]",
               title=rf"$h=1/{2 * int(steps)}$ day")
        ax.set_xticks((0.1, 1, 10))
        ax.set_yticks((10, 100, 1000))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.grid(which="major", color="#eeeeee", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("FP64 throughput [GFLOP/s]")
    for model, center in (("implicit", 0.29), ("explicit", 0.76)):
        handles = [Line2D([], [], marker=marker, color=colors[model], ls="none",
                          ms=9.5, markeredgecolor="white", label=rf"$N_P={patches}$")
                   for patches, marker in markers.items()]
        legend = fig.legend(handles=handles, title=f"{model.capitalize()} mobility", ncol=2,
                            loc="lower center", bbox_to_anchor=(center, 0.0),
                            frameon=True, fancybox=False, framealpha=1, edgecolor="#cccccc",
                            columnspacing=1, handletextpad=0.4)
        legend.get_title().set_fontweight("bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output.with_suffix("." + suffix), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=PAPER / "figures" / "cpu_roofline")
    args = parser.parse_args()
    rows, reference = read_rows(args.data)
    make_figure(rows, reference, args.output)
    print(f"Plotted {len(rows)} CPU points: {args.output.with_suffix('.pdf')}")
