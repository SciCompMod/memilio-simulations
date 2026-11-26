#!/usr/bin/env python3
"""Plot validated OpenMP strong/weak scaling of the daily commuting scenario.

"""

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, ScalarFormatter

import summarize_runtime_comparison as summary
from plot_runtime_comparison import MODELS, PAPER, set_style


RESULTS = PAPER / "results"
THREADS = (1, 2, 4, 8, 16, 32, 64, 128)


def read_rows(job, minimum_repetitions=10):
    """Summarize the scaling runs of both step sizes."""
    rows = []
    for half_day_steps in (1, 32):
        directory = Path(job) / f"hsteps_{half_day_steps}"
        rows.extend(summary.summarize(summary.load([directory]), minimum_repetitions, "cpu", "scaling"))
    return rows


def normalized_points(rows, experiment, half_day_steps, model):
    """Return median/quartile ratios, setting the self-normalized baseline to 1."""
    points = sorted((dict(row) for row in rows if row["experiment"] == experiment
                     and row["half_day_steps"] == half_day_steps and row["model"] == model),
                    key=lambda row: row["cpu_threads"])
    if tuple(row["cpu_threads"] for row in points) != THREADS:
        raise ValueError(f"Incomplete {experiment}/{model}/hsteps_{half_day_steps} curve.")
    reference = points[0]["median_seconds"]
    for point in points:
        point["reference_seconds"] = reference
        point["ratio_median"] = reference / point["median_seconds"]
        point["ratio_lower"] = reference / point["q3_seconds"]
        point["ratio_upper"] = reference / point["q1_seconds"]
        if point["cpu_threads"] == 1:
            point["ratio_lower"] = point["ratio_upper"] = 1.0
    return points


def make_figure(rows, experiment):
    if experiment not in {"strong", "weak"}:
        raise ValueError("Expected strong or weak scaling.")
    set_style()
    figure, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), sharex=True, sharey=True)
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.155, top=0.79, wspace=0.15)
    handles, labels = [], []
    scale = 1 if experiment == "strong" else 100
    bounds = [1, 128] if experiment == "strong" else [0, 100]
    for column, (ax, half_day_steps) in enumerate(zip(axes, (1, 32))):
        ideal, = ax.plot(THREADS, THREADS if experiment == "strong" else [100] * len(THREADS),
                         color="0.60", linestyle=":", linewidth=1.3, zorder=1)
        for model, label, color, marker, linestyle in MODELS:
            points = normalized_points(rows, experiment, half_day_steps, model)
            medians = [scale * point["ratio_median"] for point in points]
            lower = [scale * (point["ratio_median"] - point["ratio_lower"]) for point in points]
            upper = [scale * (point["ratio_upper"] - point["ratio_median"]) for point in points]
            bounds.extend(scale * point[key] for point in points
                          for key in ("ratio_lower", "ratio_upper"))
            line = ax.errorbar(THREADS, medians, yerr=[lower, upper], color=color,
                               marker=marker, linestyle=linestyle, linewidth=1.9,
                               markersize=5, markerfacecolor=color if model == "implicit" else "white",
                               markeredgewidth=1.0, elinewidth=0.9, capsize=2.5, zorder=3)
            if column == 0:
                handles.append(line[0])
                labels.append(label)
        ax.set_title(r"$h=1/2$ day" if half_day_steps == 1 else r"$h=1/64$ day", pad=10)
        ax.set_xlabel("CPU threads")
        ax.set_xscale("log", base=2)
        ax.set_xticks(THREADS)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(0.87, 148)
        ax.grid(which="major", color="0.70", alpha=0.22, linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Speedup" if experiment == "strong" else "Weak-scaling efficiency [%]")
    if experiment == "strong":
        axes[0].set_yscale("log", base=2)
        axes[0].set_ylim(0.88 * min(bounds), 1.12 * max(bounds))
        axes[0].set_yticks(THREADS)
        axes[0].yaxis.set_major_formatter(ScalarFormatter())
        axes[0].yaxis.set_minor_locator(NullLocator())
    else:
        upper_limit = 10 * math.ceil(1.05 * max(bounds) / 10)
        axes[0].set_ylim(0, upper_limit)
        axes[0].set_yticks(range(0, int(upper_limit) + 1, 20))
    figure.legend([*handles, ideal], [*labels, "Ideal scaling"], loc="upper center",
                  bbox_to_anchor=(0.53, 0.985), ncol=3, frameon=False,
                  columnspacing=2.0, handlelength=2.8)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, default=RESULTS / "scaling_cpu")
    parser.add_argument("--output-dir", type=Path, default=PAPER / "figures")
    parser.add_argument("--summary-output", type=Path,
                        default=RESULTS / "summary" / "parallel_scalability.csv")
    parser.add_argument("--minimum-repetitions", type=int, default=10)
    args = parser.parse_args()
    rows = read_rows(args.job, args.minimum_repetitions)
    normalized = [point for experiment in ("strong", "weak") for steps in (1, 32)
                  for model, *_ in MODELS
                  for point in normalized_points(rows, experiment, steps, model)]
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", newline="", encoding="utf-8") as handle:
        fields = list(dict.fromkeys(key for row in normalized for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(normalized)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for experiment in ("strong", "weak"):
        figure = make_figure(rows, experiment)
        for suffix in ("pdf", "png"):
            output = args.output_dir / f"{experiment}_scaling.{suffix}"
            figure.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.05)
            print(f"Wrote {output}")
        plt.close(figure)
    print(f"Validated {len(rows)} cases with {sum(row['repetitions'] for row in rows)} repetitions; "
          f"wrote {args.summary_output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
