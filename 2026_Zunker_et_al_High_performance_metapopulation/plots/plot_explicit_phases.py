#!/usr/bin/env python3
"""Plot validated daily-commuting phase timings from the explicit OpenMP runs.

The strong group uses the run with 16/128 threads. The weak group combines the
16/32/64-thread run with the 128-thread weak case from the first run. For the stacks, arithmetic means are used,
so that the phase contributions are additive and are not calculated as sums of the medians. Each job is
validated and exported separately and none of them replaces the scaling baselines.
"""

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import summarize_runtime_comparison as summary
from plot_runtime_comparison import PAPER, set_style


RESULTS = PAPER / "results"
WEAK_RESULTS = RESULTS
PHASES = (
    ("home", "Home phase", "#56B4E9"),
    ("away", "Mobility phase", "#0072B2"),
    ("departure", "Departure", "#E69F00"),
    ("return", "Return", "#333333"),
)


def read_rows(job, minimum_repetitions=10, experiment="phases"):
    rows = []
    for steps in (1, 32):
        directory = Path(job) / f"hsteps_{steps}"
        records = summary.load([directory])
        panel = summary.summarize(records, minimum_repetitions, "cpu", experiment)
        for row in panel:
            if row["half_day_steps"] != steps or row["simulation_days"] != 32:
                raise ValueError(f"Unexpected schedule in {directory}.")
            samples = [r for r in records if r["experiment"] == row["experiment"]
                       and r["cpu_threads"] == row["cpu_threads"]]
            if sorted(r["repetition_index"] for r in samples) != list(range(len(samples))):
                raise ValueError("Missing repetition indices.")
            if any(r.get("control_first") != int(r["repetition_index"] % 2 == 0)
                   for r in samples):
                raise ValueError("Profiled/control run order must alternate.")
            for key in summary.PHASE_SECONDS:
                row["mean_" + key] = statistics.fmean(r[key] for r in samples)
            for phase, *_ in PHASES:
                row[f"mean_{phase}_fraction"] = (
                    row[f"mean_{phase}_seconds"] / row["mean_profiled_seconds"])
            phase_sum = sum(row[f"mean_{phase}_seconds"] for phase, *_ in PHASES)
            if not math.isclose(phase_sum + row["mean_unattributed_seconds"],
                                row["mean_profiled_seconds"], rel_tol=1e-8, abs_tol=1e-9):
                raise ValueError("Mean phase times do not account for the profiled total.")
            if row["mean_unattributed_seconds"] / row["mean_profiled_seconds"] > 0.001:
                raise ValueError("Unattributed time is too large to omit from the plot.")
            row["max_absolute_paired_difference_fraction"] = max(
                abs(r["profiled_seconds"] / r["control_seconds"] - 1) for r in samples)
            row["source_job"] = Path(job).name
        rows.extend(panel)
    return rows


def make_figure(rows, weak_rows):
    set_style()
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.2),
                               gridspec_kw={"width_ratios": (1, 2.0)})
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.25, top=0.79, wspace=0.24)
    panels = ((rows, "phase_strong", (16, 128)),
              (weak_rows + [r for r in rows if r["experiment"] == "phase_weak"
                            and r["cpu_threads"] == 128], "phase_weak", (16, 32, 64, 128)))
    for ax, (data, experiment, thread_counts) in zip(axes, panels):
        positions = [3 * group + offset for group in range(len(thread_counts)) for offset in (0, 1)]
        points = [next(r for r in data if r["experiment"] == experiment
                       and r["cpu_threads"] == threads and r["half_day_steps"] == steps)
                  for threads in thread_counts for steps in (1, 32)]
        bottom = [0.0] * len(points)
        for phase, label, color in PHASES:
            heights = [point[f"mean_{phase}_seconds"] for point in points]
            ax.bar(positions, heights, bottom=bottom, width=0.72, color=color,
                   edgecolor="white", linewidth=0.4, label=label, zorder=3)
            bottom = [a + b for a, b in zip(bottom, heights)]
        for x, total in zip(positions, bottom):
            ax.annotate(f"{total:.2f}", (x, total), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.set_xticks(positions, [r"$h=1/2$", r"$h=1/64$"] * len(thread_counts), fontsize=9)
        ax.tick_params(axis="x", length=0, pad=7)
        for group, threads in enumerate(thread_counts):
            label = f"{threads} threads"
            if experiment == "phase_weak":
                label += f"\n$N_P={points[2 * group]['patches']}$"
            ax.text(3 * group + 0.5, -0.20, label, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=10)
        ax.set_ylabel("Runtime [s]")
        ax.set_title(r"Strong scaling: $N_P=8192$" if experiment == "phase_strong"
                     else "Weak scaling", pad=10)
        ax.set_xlim(-0.65, positions[-1] + 0.65)
        ax.set_ylim(0, max(bottom) * 1.14)
        ax.grid(axis="y", color="0.70", alpha=0.22, linewidth=0.7)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.995),
                  ncol=4, frameon=False, columnspacing=1.5, handlelength=1.3)
    return figure


def write_summary(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Validated {len(rows)} cases with {sum(r['repetitions'] for r in rows)} paired repetitions; "
          f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, default=RESULTS / "phases_cpu")
    parser.add_argument("--weak-job", type=Path, default=RESULTS / "phases-weak_cpu")
    parser.add_argument("--output", type=Path, default=PAPER / "figures" / "explicit_phases")
    parser.add_argument("--summary-output", type=Path,
                        default=RESULTS / "summary" / "explicit_phases.csv")
    parser.add_argument("--weak-summary-output", type=Path,
                        default=WEAK_RESULTS / "summary" / "explicit_phases_weak.csv")
    parser.add_argument("--minimum-repetitions", type=int, default=10)
    args = parser.parse_args()
    rows = read_rows(args.job, args.minimum_repetitions)
    weak_rows = read_rows(args.weak_job, args.minimum_repetitions, "phases-weak")
    write_summary(rows, args.summary_output)
    write_summary(weak_rows, args.weak_summary_output)
    figure = make_figure(rows, weak_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        output = args.output.with_suffix("." + suffix)
        figure.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.05)
        print(f"Wrote {output}")
    plt.close(figure)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
