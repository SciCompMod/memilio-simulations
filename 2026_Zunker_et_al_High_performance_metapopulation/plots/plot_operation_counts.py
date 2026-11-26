#!/usr/bin/env python3
"""Plot the terms of the complexity bounds with prefactors from the source code.

Per RK stage (averaged over the four RK4 stages):

  implicit mobility
    (i)  patch-state aggregation  2 N_G N_P^2 + N_G N_P
    (ii) intensity evaluation     (3 N_G^2 + N_G) N_P
    (iii) intensity aggregation   2 N_G N_P^2
    (iv) right-hand side          5 N_G N_P
    (v)  RK stage combinations    10.5 N_G N_P

  stage-aligned scheme in the mobility phase
    (i)  intensity evaluation     3 N_G^2 N_P
    (ii) aggregate right-hand side 5 N_G N_P
    (iii) traveler right-hand side 5 N_G N_E
    (iv) RK stage combinations    12 N_G N_E + 12.25 N_G N_P

Per traveler and age group, the 2K = 1/h steps of a day contain K = 1/(2h)
mobility-phase steps with 68 operations each, while departure and return need
4 N_C / N_G = 16 operations per traveler and age group and day.

"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_runtime_comparison import PAPER, set_style

MEASURED = PAPER / "results" / "roofline_cpu" / "model_summary.csv"
G = 6


def implicit_terms(P, G=G):
    return {
        "(i), (iii) aggregations": 4 * G * P**2 + G * P,
        "(ii) intensity evaluation": (3 * G**2 + G) * P,
        "(iv) right-hand side": 5 * G * P,
        "(v) RK stage combinations": 10.5 * G * P,
    }


def explicit_terms(P, G=G):
    E = P * (P - 1)
    return {
        "(i) intensity evaluation": 3 * G**2 * P,
        "(ii) aggregate right-hand side": 5 * G * P,
        "(iii) traveler right-hand side": 5 * G * E,
        "(iv) RK stage combinations": 12 * G * E + 12.25 * G * P,
    }


def per_day(model, P, K, G=G):
    """Summed terms per simulated day with K steps per half-day."""
    E, N_C = P * (P - 1), 4 * G
    if model == "implicit":
        return 2 * K * 4 * sum(implicit_terms(P, G).values())
    aggregates = 4 * (3 * G**2 * P + 5 * G * P + 12.25 * G * P)
    travelers = 4 * (5 * G * E + 12 * G * E)
    return 2 * K * aggregates + K * travelers + 4 * E * N_C


N_P_FIXED = 256
P_MARKERS = 2.0 ** np.arange(1, 15)
G_MARKERS = 2.0 ** np.arange(0, 9)

IMPLICIT_STYLES = (
    ("(i) Patch-state aggregation", dict(color="#0072B2", ls="-", lw=2.2)),
    ("(ii) Intensity evaluation", dict(color="#E69F00", ls="--", lw=1.7)),
    # (iii) differs from (i) only by N_G N_P and is drawn with markers on top of it.
    ("(iii) Intensity aggregation", dict(color="#CC79A7", ls="none", marker="o", ms=5.5, mfc="white",
                                         mew=1.5)),
    ("(iv) Right-hand side", dict(color="#009E73", ls="-.", lw=1.7)),
    ("(v) RK stage combinations", dict(color="#D55E00", ls="-", lw=1.7)),
)
EXPLICIT_STYLES = (
    ("(i) Intensity evaluation", dict(color="#E69F00", ls="--", lw=1.7)),
    ("(ii) Aggregate right-hand side", dict(color="#009E73", ls="-.", lw=1.7)),
    ("(iii) Traveler right-hand side", dict(color="#0072B2", ls="-", lw=2.2)),
    ("(iv) RK stage combinations", dict(color="#D55E00", ls="-", lw=1.7)),
)


def implicit_lines(P, G):
    return (2 * G * P**2 + G * P, (3 * G**2 + G) * P, 2 * G * P**2, 5 * G * P, 10.5 * G * P)


def explicit_lines(P, G):
    return tuple(explicit_terms(P, G).values())


def style_axis(ax):
    ax.minorticks_off()
    ax.grid(which="major", color="0.70", alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)


def make_figure(output):
    set_style()
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 6.4), sharex="col", sharey="col",
                                gridspec_kw=dict(left=0.08, right=0.66, bottom=0.09, top=0.94,
                                                 wspace=0.30, hspace=0.30))
    P = np.geomspace(2, 16384, 300)
    G = np.geomspace(1, 256, 300)
    rows = (("Implicit mobility", "Implicit mobility", implicit_lines, IMPLICIT_STYLES),
            ("Explicit mobility", "Explicit mobility, mobility phase", explicit_lines, EXPLICIT_STYLES))
    for row, (name, legend_title, lines, styles) in enumerate(rows):
        ax_p, ax_g = axes[row]
        for index, (label, style) in enumerate(styles):
            x_p, x_g = (P_MARKERS, G_MARKERS) if "marker" in style else (P, G)
            ax_p.plot(x_p, lines(x_p, 6)[index], zorder=3, **style)
            ax_g.plot(x_g, lines(N_P_FIXED, x_g)[index], zorder=3, **style)
        ax_p.set_title(f"({'ac'[row]}) {name}, " + r"$N_G=6$", loc="left", pad=6)
        ax_g.set_title(f"({'bd'[row]}) {name}, " + rf"$N_P={N_P_FIXED}$", loc="left", pad=6)
        ax_p.set_ylabel("FP64 operations per RK stage")
        ax_g.legend(title=legend_title, loc="center left", bbox_to_anchor=(1.06, 0.5), frameon=False,
                    handlelength=2.8, alignment="left",
                    handles=[Line2D([], [], **style) for _, style in styles],
                    labels=[label for label, _ in styles])
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        style_axis(ax)
    axes[1, 0].set_xticks((2, 16, 128, 1024, 16384), ("2", "16", "128", "1024", "16384"))
    axes[1, 1].set_xticks((1, 4, 16, 64, 256), ("1", "4", "16", "64", "256"))
    axes[1, 0].set_xlabel(r"Patches $N_P$")
    axes[1, 1].set_xlabel(r"Age groups $N_G$")
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(output.with_suffix("." + suffix), dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PAPER / "figures" / "operation_counts")
    parser.add_argument("--measured", type=Path, default=MEASURED,
                        help="optional model_summary.csv of the CPU roofline measurement")
    args = parser.parse_args()
    rows = []
    if args.measured.is_file():
        with open(args.measured, newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream) if row["backend"] == "cpu"]
    for row in sorted(rows, key=lambda r: (r["model"], int(r["patches"]), int(r["half_day_steps"]))):
        P, K = int(row["patches"]), int(row["half_day_steps"])
        model, measured = per_day(row["model"], P, K), float(row["median_fp64_flops"])
        print(f"{row['model']:8s} N_P={P:5d} K={K:2d}: source count {model:.4g}, measured {measured:.4g}, "
              f"ratio {model / measured:.4f}")
    make_figure(args.output)
    print(f"Wrote {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
