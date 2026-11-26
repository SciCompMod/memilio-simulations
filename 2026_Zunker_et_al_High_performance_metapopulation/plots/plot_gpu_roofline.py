#!/usr/bin/env python3
"""Plot the GPU rooflines from Nsight Compute CSV exports.

The figure shows medians over the repetitions.
"""

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator

from plot_runtime_comparison import set_style


PAPER = Path(__file__).resolve().parents[1]
RESULTS = PAPER / "results"
# Published theoretical limits:
# https://www.nvidia.com/en-eu/data-center/h100/, https://www.nvidia.com/en-us/data-center/a100/
GPU_LIMITS = {
    "H100": dict(bandwidth_tbs=3.35, simt_tflops=34.0, tensor_tflops=67.0),
    "A100": dict(bandwidth_tbs=2.039, simt_tflops=9.7, tensor_tflops=19.5),
}
PROFILE = re.compile(r"^(implicit|explicit)_p(\d+)_h(\d+)_r(\d+)\.ncu\.csv$")
FP64 = {"smsp__sass_thread_inst_executed_op_dadd_pred_on.sum": 1,
        "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum": 1,
        "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum": 2,
        "sm__ops_path_tensor_src_fp64.sum": 1}
DRAM = ("dram__bytes_read.sum", "dram__bytes_write.sum")
TIME = "gpu__time_duration.sum"
TIME_UNITS = {"ns": 1e-9, "nsecond": 1e-9, "us": 1e-6, "usecond": 1e-6,
              "ms": 1e-3, "msecond": 1e-3, "s": 1.0, "second": 1.0}
STYLES = {
    "gemm_nn": (r"$M^T X_I$", "#333333", "o"),
    "gemm_tn": (r"$MU$", "#333333", "^"),
    "explicit_commuter": ("Travelers", "#0072B2", "s"),
    "explicit_departure": ("Departure", "#E69F00", "D"),
    "explicit_return_stayers": ("Return: stayers", "#009E73", "v"),
    "explicit_return_gather": ("Return: travelers", "#CC79A7", "P"),
    "explicit_integrator_home": ("Aggregates: home phase", "#56B4E9", "h"),
    "explicit_integrator_away": ("Aggregates: mobility phase", "#D55E00", "X"),
}


def kind(name):
    """Classify a kernel by its demangled name."""
    if "gemm" in name and "_nn_" in name:
        return "gemm_nn"
    if "gemm" in name and "_tn_" in name:
        return "gemm_tn"
    if "seir_totals_allpatches_ct_kernel" in name:
        away = re.search(r"seir_totals_allpatches_ct_kernel\s*<[^,>]*,\s*(?:\(bool\)\s*)?(true|1)\s*>", name)
        return "explicit_integrator_away" if away else "explicit_integrator_home"
    for fragment, label in (("seir_traveler", "explicit_commuter"),
                            ("daily_departure_kernel", "explicit_departure"),
                            ("daily_stayers_kernel", "explicit_return_stayers"),
                            ("daily_return_kernel", "explicit_return_gather")):
        if fragment in name:
            return label
    return "other"


def read_profile(path):
    """Sum FP64 operations, DRAM bytes and time per kernel name for one profile."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.reader(handle) if row and not row[0].startswith("==")]
    header, units = rows[0], dict(zip(rows[0], rows[1]))
    kernels = defaultdict(lambda: dict(flops=0.0, bytes=0.0, seconds=0.0))
    for row in rows[2:]:
        launch = dict(zip(header, row))
        value = lambda metric: float(launch[metric].replace(",", "") or 0)
        kernel = kernels[launch["Kernel Name"]]
        kernel["flops"] += sum(weight * value(metric) for metric, weight in FP64.items())
        kernel["bytes"] += sum(value(metric) for metric in DRAM)
        kernel["seconds"] += value(TIME) * TIME_UNITS[units[TIME]]
    return kernels


def read_profiles(job):
    """Return per-kernel medians over the repetitions for every model, patch count and step size."""
    repetitions = defaultdict(list)
    for path in sorted(job.glob("*.ncu.csv")):
        match = PROFILE.match(path.name)
        if not match:
            continue
        model, patches, steps, _ = match.groups()
        kernels = read_profile(path)
        total_seconds = sum(kernel["seconds"] for kernel in kernels.values())
        for name, kernel in kernels.items():
            repetitions[model, int(patches), int(steps), name].append(dict(
                kernel, time_fraction=kernel["seconds"] / total_seconds,
                arithmetic_intensity=kernel["flops"] / kernel["bytes"] if kernel["bytes"] else float("nan"),
                gflops=kernel["flops"] / kernel["seconds"] / 1e9))
    if not repetitions:
        raise ValueError(f"No Nsight Compute CSV files found in {job}")
    rows = []
    for (model, patches, steps, name), samples in sorted(repetitions.items()):
        row = dict(model=model, patches=patches, half_day_steps=steps, kernel_kind=kind(name),
                   kernel_name=name, repetitions=len(samples))
        for field in ("seconds", "flops", "bytes", "time_fraction", "arithmetic_intensity", "gflops"):
            row["median_" + field] = statistics.median(sample[field] for sample in samples)
        rows.append(row)
    return rows


def make_figure(kernels, output, limits):
    selected = [r for r in kernels if r["patches"] == 8192 and r["kernel_kind"] in STYLES]
    if len(selected) != 2 * len(STYLES):
        raise ValueError("Expected all eight plotted kernels at each step size")
    for model in ("implicit", "explicit"):
        for steps in (1, 32):
            shown = sum(r["median_time_fraction"] for r in selected
                        if r["model"] == model and r["half_day_steps"] == steps)
            print(f"{model}, h=1/{2 * steps}: plotted kernels take {100 * shown:.1f}% of kernel time")
    set_style()
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.28, top=0.89, wspace=0.13)
    xs = np.geomspace(0.07, 250, 400)
    for ax, steps in zip(axes, (1, 32)):
        ax.set_xscale("log")
        ax.set_yscale("log")
        bandwidth = limits["bandwidth_tbs"]
        simt, tensor = limits["simt_tflops"], limits["tensor_tflops"]
        ax.plot(xs, np.minimum(bandwidth * xs, simt), color="#999999", lw=1.6)
        ax.plot(xs, np.minimum(bandwidth * xs, tensor), color="#777777", lw=1.6, ls="--")
        ax.text(235, tensor * 1.10, f"FP64 Tensor: {tensor:g}", ha="right", fontsize=11, color="#555555")
        ax.text(235, simt * 1.15, f"FP64: {simt:g}", ha="right", fontsize=11, color="#555555")
        ax.text(0.21, bandwidth * 0.47, f"HBM: {bandwidth:g} TB/s", rotation=44, fontsize=11, color="#555555")
        for row in selected:
            if row["half_day_steps"] != steps:
                continue
            _, color, marker = STYLES[row["kernel_kind"]]
            # Keep the triangle's tip visible beside the nearby implicit circle.
            ax.plot(row["median_arithmetic_intensity"], row["median_gflops"] / 1000,
                    marker=marker, ms=10.5, color=color, markeredgecolor="white", mew=1.0,
                    ls="none", zorder=5 if marker == "^" else 4)
        ax.set(xlim=(0.07, 250), ylim=(0.08, 100 if tensor > 32 else 32), xlabel="Arithmetic intensity [FLOP/byte]",
               title=rf"$h=1/{2 * steps}$ day")
        ax.set_xticks([0.1, 1, 10, 100])
        ax.set_yticks([0.1, 1, 10])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
        ax.grid(which="major", color="#eeeeee", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("FP64 throughput [TFLOP/s]")
    handles = [Line2D([], [], marker=marker, color=color, ls="none", ms=10.5,
                      markeredgecolor="white", mew=1.0, label=label)
               for label, color, marker in STYLES.values()]
    # Separate model legends; Matplotlib fills columns vertically.
    for title, order, columns, center in (
        ("Implicit mobility", [0, 1], 1, 0.18),
        ("Explicit mobility", [6, 7, 2, 3, 4, 5], 3, 0.64),
    ):
        legend = fig.legend(handles=[handles[i] for i in order], title=title,
                            loc="lower center", ncol=columns, bbox_to_anchor=(center, 0.0),
                            frameon=True, fancybox=False, framealpha=1.0, edgecolor="#cccccc",
                            columnspacing=1.0, handletextpad=0.4)
        legend.get_title().set_fontweight("bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output.with_suffix("." + suffix), dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, default=RESULTS / "roofline_gpu",
                        help="Folder with the *.ncu.csv files.")
    parser.add_argument("--gpu", choices=sorted(GPU_LIMITS), default="H100")
    parser.add_argument("--output", type=Path, default=PAPER / "figures" / "gpu_roofline")
    parser.add_argument("--summary-output", type=Path, default=RESULTS / "summary" / "gpu_roofline.csv")
    args = parser.parse_args()
    kernels = read_profiles(args.job)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(kernels[0]))
        writer.writeheader()
        writer.writerows(kernels)
    make_figure(kernels, args.output, GPU_LIMITS[args.gpu])
    print(f"Figure: {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
