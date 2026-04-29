import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "plots" / "figure8.png"

FILES = {
    "November 2020": ROOT
    / "year-2020_county_predictions_scaled-False_runs-15_rounds-75_ma-trailing_fine-eps.csv",
    "March 2022": ROOT
    / "year-2022_county_predictions_scaled-False_runs-15_rounds-75_ma-trailing_fine-eps.csv",
}

COLORS = {
    "November 2020": "#2c89ad",
    "March 2022": "#a73573",
}

MARKERS = {
    "November 2020": "o",
    "March 2022": "s",
}

METRICS = [
    ("MSE", "MSE"),
    ("MAE", "MAE"),
    ("MAPE (%)", "MAPE (%)"),
]

EPSILON_ORDER = ["0.2", "0.3", "0.4", "0.5", "0.75",
                 "1.0", "1.5", "2.0", "3.0", "5.0", "Non-DP"]
X_POS = {
    "0.2": 0.2,
    "0.3": 0.3,
    "0.4": 0.4,
    "0.5": 0.5,
    "0.75": 0.75,
    "1.0": 1.0,
    "1.5": 1.5,
    "2.0": 2.0,
    "3.0": 3.0,
    "5.0": 5.0,
    "Non-DP": 6.2,
}


def normalize_epsilon(value):
    if str(value) == "Non-DP":
        return "Non-DP"
    return str(float(value))


def summarize(path):
    df = pd.read_csv(path)
    df["eps_label"] = df["Epsilon"].map(normalize_epsilon)
    return df.groupby("eps_label", sort=False).agg(
        {
            "MSE": ["mean", "std"],
            "MAE": ["mean", "std"],
            "MAPE (%)": ["mean", "std"],
        }
    )


def values_for(summary, metric):
    means = np.array([summary.loc[eps, (metric, "mean")]
                     for eps in EPSILON_ORDER], dtype=float)
    stds = np.array([summary.loc[eps, (metric, "std")]
                    for eps in EPSILON_ORDER], dtype=float)
    lower = np.minimum(stds, means * 0.98)
    upper = stds
    return means, np.vstack([lower, upper])


def main():
    summaries = {label: summarize(path) for label, path in FILES.items()}

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 2.6,
            "axes.labelweight": "bold",
            "axes.labelsize": 38,
            "xtick.labelsize": 30,
            "ytick.labelsize": 30,
            "legend.fontsize": 34,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(33.2, 8.85), dpi=160)
    x = np.array([X_POS[eps] for eps in EPSILON_ORDER], dtype=float)
    x_dp = x[:-1]
    x_non_dp = x[-1]

    for idx, (ax, (metric, ylabel)) in enumerate(zip(axes, METRICS)):
        for label in ["November 2020", "March 2022"]:
            color = COLORS[label]
            marker = MARKERS[label]
            means, yerr = values_for(summaries[label], metric)

            ax.errorbar(
                x_dp,
                means[:-1],
                yerr=yerr[:, :-1],
                color=color,
                marker=marker,
                markersize=11,
                markerfacecolor=color,
                markeredgecolor=color,
                alpha=0.85,
                linewidth=3.2,
                elinewidth=3.0,
                capsize=8,
                capthick=2.4,
                label=label,
            )
            ax.errorbar(
                [x_non_dp],
                [means[-1]],
                yerr=yerr[:, -1:],
                color=color,
                marker="^",
                markersize=16,
                markerfacecolor=color,
                markeredgecolor=color,
                alpha=0.85,
                linewidth=0,
                elinewidth=3.0,
                capsize=8,
                capthick=2.4,
            )
            ax.plot([x_dp[-1], x_non_dp], [means[-2], means[-1]],
                    color=color, alpha=0.85, linewidth=3.2)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.17, 7.0)
        ax.set_xlabel(r"Privacy Budget $\varepsilon$", labelpad=18)
        ax.set_ylabel(ylabel, labelpad=18)
        ax.set_xticks([0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5])
        ax.set_xticklabels(["0.2", "0.3", "0.5", "0.75", "1",
                           "1.5", "2", "3", "5"], rotation=25)
        ax.grid(which="major", color="#bdbdbd",
                linestyle="--", linewidth=1.8, alpha=0.65)
        ax.grid(which="minor", color="#d9d9d9",
                linestyle=":", linewidth=1.4, alpha=0.55)
        ax.tick_params(axis="both", which="major", width=2.4, length=10)
        ax.tick_params(axis="both", which="minor", width=1.8, length=5)
        ax.legend(loc="upper right", frameon=True,
                  framealpha=0.9, edgecolor="#cccccc")
        ax.text(
            -0.16,
            1.02,
            chr(ord("A") + idx),
            transform=ax.transAxes,
            fontsize=42,
            fontweight="bold",
            va="bottom",
            clip_on=False,
        )
        ax.text(4.45, ax.get_ylim()[0] * 1.45,
                "Non-DP", fontsize=25, fontweight="bold")

    fig.subplots_adjust(left=0.06, right=0.995,
                        bottom=0.20, top=0.98, wspace=0.26)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
