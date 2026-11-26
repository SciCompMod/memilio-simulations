#!/usr/bin/env python3
"""Plot commuting runtimes versus patch and age-group counts and export the summary.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator

import summarize_runtime_comparison as summary


PAPER = Path(__file__).resolve().parents[1]
RESULTS = PAPER / "results"
MODELS = (
    ("implicit", "Implicit mobility", "#333333", "o", "-"),
    ("explicit", "Explicit mobility (stage-aligned)", "#0072B2", "s", "--"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-job", type=Path, default=RESULTS / "runtime_cpu",
                        help="CPU job directory containing hsteps_1 and hsteps_32.")
    parser.add_argument("--gpu-job", type=Path, default=RESULTS / "runtime_gpu",
                        help="GPU job directory containing hsteps_1 and hsteps_32.")
    parser.add_argument("--gpu-label", default="H100", help="GPU name used in panel titles (default: H100).")
    parser.add_argument("--reference-gpu-job", type=lambda value: Path(value) if value else None,
                        default=None,
                        help="Second GPU job drawn faintly in the GPU panels; pass '' to omit it.")
    parser.add_argument("--reference-gpu-label", default="A100",
                        help="GPU name of --reference-gpu-job (default: A100).")
    parser.add_argument("--age-groups", type=int, choices=(1, 3, 6, 8), default=6,
                        help="Number of age groups in the figure (default: 6).")
    parser.add_argument("--output", type=Path, default=PAPER / "figures" / "runtime_comparison",
                        help="Figure path without extension; writes PDF and PNG.")
    parser.add_argument("--age-output", type=Path, default=PAPER / "figures" / "runtime_age_groups",
                        help="Age-group figure path without extension (1024 patches, both step sizes).")
    parser.add_argument("--summary-output", type=Path,
                        default=RESULTS / "summary" / "runtime_comparison.csv",
                        help="CSV destination for all validated cases, including every age-group count.")
    parser.add_argument("--minimum-repetitions", type=int, default=10)
    args = parser.parse_args()
    if args.minimum_repetitions < 1:
        parser.error("--minimum-repetitions must be positive")
    return args


def read_rows(args, gpu_job):
    """Summarize the CPU and GPU runs of both step sizes."""
    rows = []
    for half_day_steps in (1, 32):
        inputs = [job / f"hsteps_{half_day_steps}" for job in (args.cpu_job, gpu_job)]
        records = [record for directory in inputs for record in summary.load([directory])]
        rows.extend(summary.summarize(records, args.minimum_repetitions, "all"))
    return rows


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10, "mathtext.fontset": "dejavusans",
        "axes.labelsize": 11, "axes.titlesize": 11, "legend.fontsize": 10,
        "text.color": "#333333", "axes.labelcolor": "#333333",
        "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
        "xtick.color": "#333333", "ytick.color": "#333333",
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def backend_labels(gpu_label, reference_label=None):
    title = f"GPU: {gpu_label}" + (f" and {reference_label}" if reference_label else "")
    return (("serial", "CPU: serial"), ("openmp", "CPU: 128 cores"), ("cuda", title))


def plot_reference(ax, reference_rows, x_key, markersize):
    """Draw the second GPU without error bars and lighter, behind the primary GPU."""
    for model, _, color, marker, linestyle in MODELS:
        points = sorted((row for row in reference_rows if row["model"] == model),
                        key=lambda row: row[x_key])
        ax.plot([row[x_key] for row in points], [1000 * row["median_seconds"] for row in points],
                color=color, marker=marker, linestyle=linestyle, linewidth=1.2, alpha=0.35,
                markersize=markersize, markerfacecolor=color if model == "implicit" else "white",
                markeredgewidth=0.8, zorder=1)


def gpu_handle(label, alpha):
    return Line2D([], [], color="#333333", linewidth=1.2, alpha=alpha, label=label)


def make_figure(rows, age_groups, gpu_label="H100", reference_rows=(), reference_label=None):
    set_style()
    figure, axes = plt.subplots(2, 3, figsize=(9.8, 6.0), sharex=True, sharey="row")
    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.095, top=0.89,
                           hspace=0.24, wspace=0.14)
    selected = [row for row in rows if row["age_groups"] == age_groups]
    reference = [row for row in reference_rows if row["age_groups"] == age_groups]
    patches = sorted({row["patches"] for row in selected})
    tick_patches = patches[::2] if len(patches) <= 7 else [patches[i] for i in (0, 3, 6, 8, 10)]
    handles, labels = [], []
    for row_index, half_day_steps in enumerate((1, 32)):
        step_rows = [row for row in selected if row["half_day_steps"] == half_day_steps]
        step_reference = [row for row in reference if row["half_day_steps"] == half_day_steps]
        for column, (backend, backend_label) in enumerate(backend_labels(gpu_label, reference_label)):
            ax = axes[row_index, column]
            if backend == "cuda":
                plot_reference(ax, step_reference, "patches", 3.5)
            for model, label, color, marker, linestyle in MODELS:
                points = sorted((row for row in step_rows
                                 if row["backend"] == backend and row["model"] == model),
                                key=lambda row: row["patches"])
                medians = [1000 * row["median_seconds"] for row in points]
                lower = [1000 * (row["median_seconds"] - row["q1_seconds"]) for row in points]
                upper = [1000 * (row["q3_seconds"] - row["median_seconds"]) for row in points]
                line = ax.errorbar(
                    [row["patches"] for row in points], medians, yerr=[lower, upper],
                    color=color, marker=marker, linestyle=linestyle, linewidth=1.9,
                    markersize=4.3, markerfacecolor=color if model == "implicit" else "white",
                    markeredgewidth=1.0, elinewidth=0.9, capsize=2.3,
                )
                if row_index == column == 0:
                    handles.append(line[0])
                    labels.append(label)
            ax.set_xscale("log", base=2)
            ax.set_xlim(patches[0] / 1.4, patches[-1] * 1.4)
            ax.set_yscale("log")
            ax.set_xticks(tick_patches, [str(value) for value in tick_patches])
            ax.tick_params(axis="x", labelsize=9)
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
            ax.grid(which="major", color="0.70", alpha=0.22, linewidth=0.7)
            ax.set_axisbelow(True)
            ax.minorticks_off()
            if row_index == 0:
                ax.set_title(backend_label, pad=10)
            else:
                ax.set_xlabel(r"Patches $N_P$")
        axes[row_index, 0].set_ylim(
            1000 * min(row["q1_seconds"] for row in step_rows + step_reference) / 1.5,
            1000 * max(row["q3_seconds"] for row in step_rows + step_reference) * 1.5)
        fraction = "1/2" if half_day_steps == 1 else "1/64"
        axes[row_index, 0].set_ylabel(rf"$h={fraction}$ day" + "\nRuntime [ms]")
    if reference:
        handles += [gpu_handle(gpu_label, 1.0), gpu_handle(reference_label, 0.35)]
        labels += [gpu_label, reference_label]
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.535, 1.0),
                  ncol=len(handles), frameon=False, columnspacing=1.6, handlelength=2.8)
    return figure


def make_age_group_figure(rows, gpu_label="H100", reference_rows=(), reference_label=None):
    """Compare age groups for both step sizes at their largest shared patch count."""
    set_style()
    groups = (1, 3, 6, 8)
    selected = [row for row in rows if row["patches"] == 1024
                and row["backend"] in {"openmp", "cuda"}]
    reference = [row for row in reference_rows if row["patches"] == 1024]
    figure, axes = plt.subplots(2, 2, figsize=(9.8, 5.8), sharex=True, sharey="row")
    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.10, top=0.835,
                           hspace=0.25, wspace=0.16)
    handles, labels = [], []
    for row_index, half_day_steps in enumerate((1, 32)):
        step_rows = [row for row in selected if row["half_day_steps"] == half_day_steps]
        step_reference = [row for row in reference if row["half_day_steps"] == half_day_steps]
        for column, (backend, backend_label) in enumerate(
                backend_labels(gpu_label, reference_label)[1:]):
            ax = axes[row_index, column]
            if backend == "cuda":
                plot_reference(ax, step_reference, "age_groups", 4)
            for model, label, color, marker, linestyle in MODELS:
                points = sorted((row for row in step_rows
                                 if row["backend"] == backend and row["model"] == model),
                                key=lambda row: row["age_groups"])
                if tuple(row["age_groups"] for row in points) != groups:
                    raise ValueError(f"Incomplete age-group comparison for {backend}/{model}, "
                                     f"half_day_steps={half_day_steps}.")
                medians = [1000 * row["median_seconds"] for row in points]
                lower = [1000 * (row["median_seconds"] - row["q1_seconds"]) for row in points]
                upper = [1000 * (row["q3_seconds"] - row["median_seconds"]) for row in points]
                line = ax.errorbar(groups, medians, yerr=[lower, upper], color=color, marker=marker,
                                   linestyle=linestyle, linewidth=1.9, markersize=5,
                                   markerfacecolor=color if model == "implicit" else "white",
                                   markeredgewidth=1.0, elinewidth=0.9, capsize=3)
                if row_index == column == 0:
                    handles.append(line[0])
                    labels.append(label)
            if row_index == 0:
                ax.set_title(backend_label, pad=10)
            else:
                ax.set_xlabel(r"Age groups $N_G$")
            ax.set_xticks(groups)
            ax.set_xlim(0.6, 8.4)
            ax.grid(which="major", color="0.70", alpha=0.22, linewidth=0.7)
            ax.set_axisbelow(True)
        fraction = "1/2" if half_day_steps == 1 else "1/64"
        axes[row_index, 0].set_ylabel(rf"$h={fraction}$ day" + "\nRuntime [ms]")
        axes[row_index, 0].set_ylim(
            0, 1.10 * max(1000 * row["q3_seconds"] for row in step_rows + step_reference))
    if reference:
        handles += [gpu_handle(gpu_label, 1.0), gpu_handle(reference_label, 0.35)]
        labels += [gpu_label, reference_label]
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.535, 1.005),
                  ncol=len(handles), frameon=False, columnspacing=1.6, handlelength=2.8)
    figure.text(0.535, 0.925, r"$N_P=1024$", ha="center", color="0.35")
    return figure


def main():
    args = parse_args()
    rows = read_rows(args, args.gpu_job)
    # The reference GPU is plotted only; the CPU rows and the summary CSV stay those of --gpu-job.
    reference_rows, reference_label = [], None
    if args.reference_gpu_job:
        reference_rows = [row for row in read_rows(args, args.reference_gpu_job)
                          if row["backend"] == "cuda"]
        reference_label = args.reference_gpu_label
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for figure, output_path in (
            (make_figure(rows, args.age_groups, args.gpu_label, reference_rows, reference_label),
             args.output),
            (make_age_group_figure(rows, args.gpu_label, reference_rows, reference_label),
             args.age_output)):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            output = output_path.with_suffix(suffix)
            figure.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.05)
            print(f"Wrote {output}")
        plt.close(figure)
    print(f"Validated and summarized {len(rows)} cases: {args.summary_output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
