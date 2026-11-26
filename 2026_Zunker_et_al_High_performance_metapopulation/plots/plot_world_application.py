#!/usr/bin/env python3
"""Figure of the global application.

(a) Arrival day of the epidemic per patch, i.e., the first day with at least one infectious
resident, for explicit mobility on an Equal Earth map. (b) Time to solution of the complete
simulation for implicit and explicit mobility on one CPU core, 128 CPU cores, and one H100.

"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

from plot_runtime_comparison import PAPER, set_style

WORLD = PAPER / "world"
BORDERS = WORLD / "ne_50m_admin_0_countries.geojson"
GPU_RESULTS = PAPER / "results" / "world_gpu"
CPU_RESULTS = PAPER / "results" / "world_cpu"
THRESHOLD = 1.0
LAT_RANGE = (-58, 80)

ARRIVAL_BOUNDS = [0, 21, 28, 35, 42, 49, 70]
ARRIVAL_COLORS = ["#67001f", "#b2182b", "#e8492f", "#f78a3a", "#fcbf5c", "#fee98f"]
OCEAN = "#cfe3f1"
LAND = "#e6dfcc"
INK = "#333333"

MODELS = (("implicit", "Implicit mobility", "#333333", "o"),
          ("explicit", "Explicit mobility", "#0072B2", "s"))
HARDWARE = (("serial", "CPU, 1 core"), ("openmp", "CPU, 128 cores"), ("cuda", "GPU, H100"))


def equal_earth(lon, lat):
    """Equal Earth projection (Savric, Patterson, Jenny 2018) on the unit sphere."""
    a1, a2, a3, a4 = 1.340264, -0.081106, 0.000893, 0.003796
    m = np.sqrt(3.0) / 2.0
    theta = np.arcsin(m * np.sin(np.radians(lat)))
    t2 = theta * theta
    t6 = t2 * t2 * t2
    x = np.radians(lon) * np.cos(theta) / (m * (a1 + 3 * a2 * t2 + t6 * (7 * a3 + 9 * a4 * t2)))
    y = theta * (a1 + a2 * t2 + t6 * (a3 + a4 * t2))
    return x, y


def load_daily(path):
    with open(path, "rb") as stream:
        records, patches = np.fromfile(stream, dtype="<i4", count=2)
        return np.fromfile(stream, dtype="<f4", count=records * patches).reshape(records, patches)


def arrival_day(daily):
    reached = daily >= THRESHOLD
    day = np.argmax(reached, axis=0).astype(float)
    day[~reached.any(axis=0)] = np.nan
    return day


def country_polygons(path):
    with open(path) as stream:
        features = json.load(stream)["features"]
    polygons = []
    for feature in features:
        geometry = feature["geometry"]
        parts = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        for part in parts:
            ring = np.asarray(part[0])
            polygons.append(np.column_stack(equal_earth(ring[:, 0], ring[:, 1])))
    return polygons


def cell_polygons(cells):
    north = 90.0 - cells["row"]
    west = cells["col"] - 180.0
    corners_lon = np.stack([west, west + 1, west + 1, west], axis=1)
    corners_lat = np.stack([north, north, north - 1, north - 1], axis=1)
    x, y = equal_earth(corners_lon, corners_lat)
    return np.stack([x, y], axis=2)


def draw_map(ax, cells, days, borders):
    lat = np.linspace(-90, 90, 181)
    left = np.column_stack(equal_earth(np.full_like(lat, -180.0), lat))
    right = np.column_stack(equal_earth(np.full_like(lat, 180.0), lat))[::-1]
    ax.add_collection(PolyCollection([np.vstack([left, right])], facecolor=OCEAN, edgecolor="#a9c4d8",
                                     linewidth=0.6, zorder=0))
    countries = country_polygons(borders)
    ax.add_collection(PolyCollection(countries, facecolor=LAND, edgecolor="none", zorder=1))

    classes = np.digitize(days, ARRIVAL_BOUNDS[1:-1])
    colors = np.asarray(ARRIVAL_COLORS)[classes]
    ax.add_collection(PolyCollection(cell_polygons(cells), facecolors=colors, edgecolors=colors, linewidth=0.15,
                                     zorder=2, rasterized=True))
    ax.add_collection(PolyCollection(countries, facecolor="none", edgecolor="white", linewidth=0.3, zorder=3))

    seed = int(cells["seed"])
    x, y = equal_earth(cells["lon"][seed], cells["lat"][seed])
    ax.plot(x, y, marker="*", markersize=17, color="white", markeredgecolor=INK, markeredgewidth=1.0, zorder=5)

    x0, _ = equal_earth(-180.0, 0.0)
    _, y0 = equal_earth(0.0, LAT_RANGE[0])
    _, y1 = equal_earth(0.0, LAT_RANGE[1])
    ax.set_xlim(x0 - 0.02, -x0 + 0.02)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")

    labels = [f"< {ARRIVAL_BOUNDS[1]}"] + [f"{lo}–{hi - 1}" for lo, hi in zip(ARRIVAL_BOUNDS[1:-2], ARRIVAL_BOUNDS[2:-1])]
    labels.append(f"≥ {ARRIVAL_BOUNDS[-2]}")
    handles = [matplotlib.patches.Patch(facecolor=color, edgecolor="none", label=label)
               for color, label in zip(ARRIVAL_COLORS, labels)]
    legend = ax.legend(handles=handles, title="Arrival day", loc="lower left", bbox_to_anchor=(0.655, 0.02),
                       frameon=False, fontsize=9.5, title_fontsize=10, handlelength=1.4, handleheight=1.1,
                       labelspacing=0.35, handletextpad=0.5, alignment="left")
    legend.get_title().set_color(INK)


def read_logs(directory):
    runtimes = {}
    pattern = re.compile(r"World (implicit|explicit)(?:_(serial|openmp\d+))?: .*?days=(\d+) .*?seconds=([0-9.eE+-]+)")
    for log in sorted(Path(directory).glob("*.log")):
        for line in log.read_text().splitlines():
            match = pattern.search(line)
            if match:
                model, backend, days, seconds = match.groups()
                backend = "cuda" if backend is None else ("serial" if backend == "serial" else "openmp")
                runtimes[(model, backend)] = {"seconds": float(seconds), "days": int(days)}
    return runtimes


def format_time(seconds):
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 600:
        return f"{seconds / 60:.1f} min"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def draw_runtimes(ax, runtimes):
    """Horizontal bars on a linear time axis, such that the bar lengths show the real ratios."""
    rows = {backend: index for index, (backend, _) in enumerate(reversed(HARDWARE))}
    offsets = {"implicit": 0.19, "explicit": -0.19}
    for model, label, color, _ in MODELS:
        for backend, row in rows.items():
            value = runtimes[(model, backend)]
            hours = value["seconds"] / 3600.0
            ax.barh(row + offsets[model], hours, height=0.34, color=color, edgecolor="none", zorder=2,
                    label=label if backend == "serial" else None)
            ax.annotate(format_time(value["seconds"]), (hours, row + offsets[model]), xytext=(5, 0),
                        textcoords="offset points", ha="left", va="center", color=INK,
                        fontsize=11 if backend == "cuda" else 10, fontweight="bold" if backend == "cuda" else "normal")
    longest = max(value["seconds"] for value in runtimes.values()) / 3600.0
    ax.set_xlim(0, 1.16 * longest)
    ax.set_ylim(-0.6, len(HARDWARE) - 0.4)
    ticks = np.arange(0, np.floor(1.16 * longest) + 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0"] + [f"{tick:.0f} h" for tick in ticks[1:]])
    ax.set_yticks(list(rows.values()))
    ax.set_yticklabels([dict(HARDWARE)[backend] for backend in rows])
    for tick, backend in zip(ax.get_yticklabels(), rows):
        if backend == "cuda":
            tick.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, labelsize=10.5, pad=6)
    ax.tick_params(axis="x", length=0, labelsize=10)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.legend(loc="center right", frameon=False, fontsize=10, handlelength=1.2, handleheight=0.9,
              handletextpad=0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=Path, default=GPU_RESULTS)
    parser.add_argument("--cpu", type=Path, default=CPU_RESULTS)
    parser.add_argument("--cells", type=Path, default=WORLD / "data" / "world_cells.npz",
                        help="patch file written by world/preprocess_world.py")
    parser.add_argument("--borders", type=Path, default=BORDERS,
                        help="Natural Earth 1:50m admin 0 countries (GeoJSON)")
    parser.add_argument("--output", type=Path, default=PAPER / "figures" / "world_application")
    args = parser.parse_args()

    set_style()
    cells = np.load(args.cells)
    days = arrival_day(load_daily(args.gpu / "explicit_daily_infectious.bin"))
    runtimes = read_logs(args.gpu)
    runtimes.update(read_logs(args.cpu))
    simulated_days = runtimes[("explicit", "cuda")]["days"]

    # The map spans the full width, the bar chart with its row labels is centered below it.
    figure = plt.figure(figsize=(9.8, 6.7))
    ax_map = figure.add_axes([0.0, 0.35, 1.0, 0.65])
    ax_time = figure.add_axes([0.235, 0.04, 0.64, 0.245])
    draw_map(ax_map, cells, days, args.borders)
    draw_runtimes(ax_time, runtimes)
    ax_time.set_title(f"Time to solution for {simulated_days} days", loc="left", fontsize=11, pad=8)

    for name, value in sorted(runtimes.items()):
        print(f"{name[0]:8s} {name[1]:6s} {value['seconds']:10.1f} s")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(args.output.with_suffix("." + suffix), dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
