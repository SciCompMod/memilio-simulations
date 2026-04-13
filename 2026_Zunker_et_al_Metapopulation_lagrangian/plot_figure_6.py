import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors, LogNorm
import matplotlib.cm as mcm
import matplotlib.lines as mlines

import re
from matplotlib.colors import LinearSegmentedColormap

_BLYIGN_COLORS = [
    "#1D6996",
    "#5BA3C9",
    "#ABD9E9",
    "#E8F4F8",
    "#FFFFBF",
    "#D9EF8B",
    "#91CF60",
    "#31A354",
    "#006837",
]


# -- Colormap helper: yellow pinned at `center` ---------------------------
def make_rdylgn_centered(vmin: float, vmax: float, center: float = 1.0):
    """Return a blue→yellow→green colormap where `center` shows as yellow.

    Use together with a standard LogNorm(vmin, vmax).
    """
    pivot = np.log(center / vmin) / np.log(vmax / vmin)
    pivot = float(np.clip(pivot, 0.01, 0.99))
    n = 512
    n_lo = max(1, int(round(n * pivot)))
    n_hi = n - n_lo
    mid = len(_BLYIGN_COLORS) // 2  # index of yellow
    cmap_lo = LinearSegmentedColormap.from_list(
        "lo", _BLYIGN_COLORS[:mid + 1], N=n_lo)
    cmap_hi = LinearSegmentedColormap.from_list(
        "hi", _BLYIGN_COLORS[mid:],    N=n_hi)
    colors = np.vstack([
        cmap_lo(np.linspace(0.0, 1.0, n_lo)),
        cmap_hi(np.linspace(0.0, 1.0, n_hi)),
    ])
    return LinearSegmentedColormap.from_list("BlYlGn_c", colors, N=n)


# --------------------------- Config & Style -----------------------------------

DIR_IN = "saves"
CSV_BENCH = os.path.join("bench_neu2.csv")
OUTDIR = os.path.join(DIR_IN, "Figures", "Figure6")
os.makedirs(OUTDIR, exist_ok=True)

AGE_PANELS = [1, 3, 6]
T_HORIZON_DAYS = 50.0 
# TIme per day or total time
PER_DAY = False


def set_style():
    COLORS = {
        "hybrid flow-based":              "#5B2A86",
        "stage-aligned flow-based (RK-4)": "#0072B2",
        "stage-aligned flow-based (Euler)": "#009E73",
        "auxiliary Euler":                "#D55E00",
        "Standard Lagrangian (Euler)":    "#E69F00",
    }
    STYLES = {
        "hybrid flow-based":              dict(lw=5.6, linestyle="-"),
        "stage-aligned flow-based (RK-4)": dict(lw=5.0, linestyle="-."),
        "stage-aligned flow-based (Euler)": dict(lw=5.0, linestyle="--"),
        "auxiliary Euler":                dict(lw=4.4, linestyle="--"),
        "Standard Lagrangian (Euler)":    dict(lw=6.2, linestyle=":"),
        "Standard Lagrangian (RK-4)":      dict(lw=6.2, linestyle=":"),
    }

    METHODS_MAIN = ["Standard Lagrangian (Euler)", "Standard Lagrangian (RK-4)",
                    "stage-aligned flow-based (RK-4)",
                    "stage-aligned flow-based (Euler)",
                    "auxiliary Euler", "hybrid flow-based"]
    base = {
        "font.size": 16,
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 16,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.55,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
    mpl.rcParams.update(base)


COLORS = {
    "hybrid flow-based":              "#5B2A86",
    "stage-aligned flow-based (RK-4)": "#0072B2",
    "stage-aligned flow-based (Euler)": "#009E73",
    "auxiliary Euler":                "#D55E00",
    "Standard Lagrangian (Euler)":    "#E69F00",
    "Standard Lagrangian (RK-4)":      "#CC7700",
}
STYLES = {
    "hybrid flow-based":              dict(lw=5.6, linestyle="-"),
    "stage-aligned flow-based (RK-4)": dict(lw=5.0, linestyle="-."),
    "stage-aligned flow-based (Euler)": dict(lw=5.0, linestyle="--"),
    "auxiliary Euler":                dict(lw=4.4, linestyle="--"),
    "Standard Lagrangian (Euler)":    dict(lw=6.2, linestyle=":"),
    "Standard Lagrangian (RK-4)":      dict(lw=6.2, linestyle=":"),
}

METHODS_MAIN = [
    "Standard Lagrangian (Euler)",
    "Standard Lagrangian (RK-4)",
    "stage-aligned flow-based (RK-4)",
    "stage-aligned flow-based (Euler)",
    "auxiliary Euler",
    "hybrid flow-based",
]

USE_SIMPLE_LOG_CBAR = True
LOW_SIDE_COLOR_FRACTION = 0.30

# --------------------------- Data helpers -------------------------------------


def _clean_method_name(s: str) -> str:
    s = str(s).split("/")[0].strip()
    mapping = {
        "Flow-based": "hybrid flow-based",
        "Flow based": "hybrid flow-based",
        "Flow-based(exact-cache)": "stage-aligned flow-based (RK-4)",
        "Flow-based(exact-closed)": "stage-aligned flow-based (RK-4)",
        "Flow-based(exact)": "stage-aligned flow-based (RK-4)",
        "Flow-based(Euler)": "stage-aligned flow-based (Euler)",
        "Explicit": "Standard Lagrangian (Euler)",
        "Euler": "auxiliary Euler",
        "stage-aligned(RK4)": "stage-aligned flow-based (RK-4)",
        "stage-aligned(RK-4)": "stage-aligned flow-based (RK-4)",
        "stage-aligned(Euler)": "stage-aligned flow-based (Euler)",
        "stage-aligned(hybrid)": "hybrid flow-based",
        "stage-aligned(hybrid_flow)": "hybrid flow-based",
        "auxiliary_Euler": "auxiliary Euler",
        "lagrange_euler": "Standard Lagrangian (Euler)",
        "lagrange_rk4": "Standard Lagrangian (RK-4)",
        "lagrange": "Standard Lagrangian (Euler)",
    }
    return mapping.get(s, s)


def _parse_gbm_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "name" not in df.columns:
        raise ValueError("Expected Google Benchmark CSV with a 'name' column.")

    number_patches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    idx_to_patches = {i: v for i, v in enumerate(number_patches)}

    parts = df["name"].astype(str).str.split("/")
    df = df[parts.map(len) == 3].copy()
    parts = df["name"].astype(str).str.split("/")

    df["Method"] = parts.map(lambda x: _clean_method_name(x[0]))
    df["PatchIndex"] = parts.map(lambda x: pd.to_numeric(
        x[1], errors="coerce")).astype("Int64")
    df["NumAgeGroups"] = parts.map(lambda x: pd.to_numeric(
        x[2], errors="coerce")).astype("Int64")
    df = df.dropna(subset=["PatchIndex", "NumAgeGroups"]).copy()
    df["PatchIndex"] = df["PatchIndex"].astype(int)
    df["NumAgeGroups"] = df["NumAgeGroups"].astype(int)
    df["NumPatches"] = df["PatchIndex"].map(idx_to_patches)

    # real_time (µs) bevorzugen
    df["Time"] = pd.to_numeric(df.get("real_time", np.nan), errors="coerce")
    df = df.dropna(subset=["Time"]).copy()

    # optionale GBM-Counter (falls vorhanden)
    for cname in ("advance_frac", "mobility_frac", "advance_us", "mobility_us"):
        if cname in df.columns:
            df[cname] = pd.to_numeric(df[cname], errors="coerce")

    keep_cols = ["Method", "NumAgeGroups", "NumPatches", "Time"]
    keep_cols += [c for c in ("advance_frac", "mobility_frac",
                              "advance_us", "mobility_us") if c in df.columns]
    return df[keep_cols].copy()


def _parse_gbm_table(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # Skip header/separators
            if line.startswith("-----") or line.lower().startswith("benchmark"):
                continue
            # Capture name and time in microseconds
            m = re.match(r"^(?P<name>\S+)\s+(?P<time>[0-9.,]+)\s*us", line)
            if not m:
                continue
            name = m.group("name")
            time_str = m.group("time").replace(",", "")
            try:
                time_val = float(time_str)
            except Exception:
                continue
            rows.append({"name": name, "real_time": time_val})

    if not rows:
        # return empty frame with expected cols to let caller raise informative error
        return pd.DataFrame(columns=["Method", "NumAgeGroups", "NumPatches", "Time"])

    df = pd.DataFrame(rows)

    # Reuse the same parsing logic as in _parse_gbm_csv from here on
    parts = df["name"].astype(str).str.split("/")
    df = df[parts.map(len) == 3].copy()
    parts = df["name"].astype(str).str.split("/")

    df["Method"] = parts.map(lambda x: _clean_method_name(x[0]))
    df["PatchIndex"] = parts.map(lambda x: pd.to_numeric(
        x[1], errors="coerce")).astype("Int64")
    df["NumAgeGroups"] = parts.map(lambda x: pd.to_numeric(
        x[2], errors="coerce")).astype("Int64")
    df = df.dropna(subset=["PatchIndex", "NumAgeGroups"]).copy()
    df["PatchIndex"] = df["PatchIndex"].astype(int)
    df["NumAgeGroups"] = df["NumAgeGroups"].astype(int)

    number_patches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    idx_to_patches = {i: v for i, v in enumerate(number_patches)}
    df["NumPatches"] = df["PatchIndex"].map(idx_to_patches)

    df["Time"] = pd.to_numeric(df.get("real_time", np.nan), errors="coerce")
    df = df.dropna(subset=["Time"]).copy()

    keep_cols = ["Method", "NumAgeGroups", "NumPatches", "Time"]
    return df[keep_cols].copy()


def load_benchmark(csv_path: str) -> pd.DataFrame:
    try:
        df_raw = pd.read_csv(csv_path)
    except Exception:
        df_raw = pd.DataFrame()

    if "name" in df_raw.columns:
        df = _parse_gbm_csv(csv_path)
    else:
        df = _parse_gbm_table(csv_path)

    if "NumCommuterGroups" in df.columns and "NumPatches" not in df.columns:
        df = df.rename(columns={"NumCommuterGroups": "NumPatches"})
    needed = {"Method", "NumAgeGroups", "NumPatches", "Time"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    df["Method"] = df["Method"].map(_clean_method_name)
    df = df[df["Method"].isin(METHODS_MAIN)].copy()

    if PER_DAY:
        df["time_value"] = df["Time"].astype(float) / (T_HORIZON_DAYS * 1e6)
    else:
        df["time_value"] = df["Time"].astype(float) / 1e6

    return df

# --------------------------- Plot helpers -------------------------------------


def _add_scaling_refs(ax, x_vals, y_anchor, x_anchor, color="0.7"):
    x = np.array(sorted(x_vals))
    y1 = y_anchor * (x / x_anchor) ** 1.0
    y2 = y_anchor * (x / x_anchor) ** 2.0
    ax.loglog(x, y1, linestyle="--", color=color,
              alpha=0.55, linewidth=6., zorder=1)
    ax.loglog(x, y2, linestyle=":",  color=color,
              alpha=0.55, linewidth=6., zorder=1)


def _lines_for_methods(ax, df_panel, label_prefix=""):
    for m in METHODS_MAIN:
        dfm = df_panel[df_panel["Method"] == m].sort_values("NumPatches")
        if dfm.empty:
            continue
        x = dfm["NumPatches"].values
        y = dfm["time_value"].values
        style = STYLES[m].copy()
        display_name = m
        if m == "hybrid flow-based":
            zline = 1
            zfill = 0
        else:
            zline = 3
            zfill = 2

        ax.loglog(x, y, label=f"{label_prefix}{display_name}",
                  color=COLORS[m], zorder=zline, **style)
        if {"time_q25", "time_q75"}.issubset(dfm.columns):
            ylo = dfm["time_q25"].values
            yhi = dfm["time_q75"].values
            ax.fill_between(
                x, ylo, yhi, color=COLORS[m], alpha=0.18, linewidth=0, zorder=zfill)


def compute_speedup_pair(df_agg: pd.DataFrame, num: str, den: str) -> pd.DataFrame:
    key = ["NumAgeGroups", "NumPatches"]
    piv = df_agg.pivot_table(index=key, columns="Method",
                             values="time_value", aggfunc="median").reset_index()
    rows = []
    for _, r in piv.iterrows():
        nag = int(r["NumAgeGroups"])
        npc = int(r["NumPatches"])
        t_num = r.get(num, np.nan)
        t_den = r.get(den, np.nan)
        if np.isfinite(t_num) and np.isfinite(t_den) and t_den > 0:
            rows.append({"NumAgeGroups": nag, "NumPatches": npc,
                        "speedup": float(t_num)/float(t_den)})
    return pd.DataFrame(rows)


def _make_speedup_grid(df_agg: pd.DataFrame, num: str, den: str):
    sp = compute_speedup_pair(df_agg, num=num, den=den)
    if sp.empty:
        return None, None, None
    rows = sorted(sp["NumAgeGroups"].unique())
    cols = sorted(sp["NumPatches"].unique())
    grid = np.full((len(rows), len(cols)), np.nan, dtype=float)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            val = sp[(sp["NumAgeGroups"] == r) & (
                sp["NumPatches"] == c)]["speedup"].median()
            if pd.notna(val):
                grid[i, j] = float(val)
    return grid, rows, cols

# --------------------------- Combined 2×3 multipanel --------------------------


def figure_multipanel_combined(df: pd.DataFrame):
    set_style()
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.25,
                          top=0.94, bottom=0.12, left=0.08, right=0.95,
                          height_ratios=[1.0, 1.15])

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1])
    axF = fig.add_subplot(gs[1, 2])

    axs = [[axA, axB, axC], [axD, axE, axF]]

    # ---- Top row: Runtime‑Plots ----
    for nag, ax, lbl in zip(AGE_PANELS, axs[0], list("ABC")):
        sub = df[df["NumAgeGroups"] == nag]
        if sub.empty:
            continue
        ref_method = "Flow-based" if (sub["Method"] ==
                                      "Flow-based").any() else sub["Method"].iloc[0]
        sub_ref = sub[sub["Method"] == ref_method]
        x_vals = sorted(sub_ref["NumPatches"].unique())
        x_anchor = x_vals[0]
        y_vals_at_anchor = sub_ref[sub_ref["NumPatches"]
                                   == x_anchor]["time_value"]
        y_anchor = float(y_vals_at_anchor.min()) * \
            0.3
        _add_scaling_refs(ax, x_vals, y_anchor, x_anchor)
        _lines_for_methods(ax, sub)
        ax.set_xlabel("Number of patches")
        ax.set_ylabel(
            "Time per simulated day [s]" if PER_DAY else "Total time [s]")
        ax.set_title(f"{nag} age group{'s' if nag != 1 else ''}")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.text(-0.09, 1.06, lbl, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=22, fontweight="bold", clip_on=False)

    # ---- Bottom row: Speedup-Heatmaps ----
   
    gridF_new, rowsF_new, colsF_new = _make_speedup_grid(
        df, num="Standard Lagrangian (Euler)", den="stage-aligned flow-based (Euler)")
    titleF = "Euler: Lagr. / stage-aligned"

    gridE, rowsE, colsE = _make_speedup_grid(
        df, num="Standard Lagrangian (RK-4)", den="stage-aligned flow-based (RK-4)")
    titleE = "RK-4: Lagr. / stage-aligned"

    gridD_new, rowsD_new, colsD_new = _make_speedup_grid(
        df, num="auxiliary Euler", den="hybrid flow-based")
    titleD = "auxiliary / hybrid"

    grids = []
    if gridF_new is not None:
        grids.append(("F", axD, gridF_new, rowsF_new, colsF_new, titleF))

    if gridE is not None:
        grids.append(("E", axE, gridE, rowsE, colsE, titleE))

    if gridD_new is not None:
        grids.append(("D", axF, gridD_new, rowsD_new, colsD_new, titleD))

    vals = []
    for _, _, g, *_ in grids:
        vals.extend(list(np.ravel(g[np.isfinite(g)])))
    if vals:
        raw_min, raw_max = min(vals), max(vals)
    else:
        raw_min, raw_max = 0.5, 2.0

    vmin = max(1e-6, min(raw_min, 0.5))
    vmax = max(raw_max, 50.0)
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = max(vmin * 2, 50.0)

    norm_speed = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap_speed = make_rdylgn_centered(vmin, vmax, center=1.0)

    for lbl, ax, grid, rows, cols, title in grids:
        im = ax.imshow(grid, aspect="auto", origin="lower",
                       cmap=cmap_speed, norm=norm_speed, interpolation="nearest")
        ax.set_title(title, pad=4)
        ax.set_xlabel("Number of patches")
        ax.set_ylabel("Age groups")
        ax.set_xticks(list(range(len(cols))))
        ax.set_xticklabels([str(c) for c in cols], rotation=45, ha="right")
        ax.set_yticks(list(range(len(rows))))
        ax.set_yticklabels([str(r) for r in rows])
        ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color="w",
                linestyle="-", linewidth=0.6, alpha=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.text(-0.09, 1.06, lbl, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=22, fontweight="bold", clip_on=False)
        # Cell annotations
        for i in range(len(rows)):
            for j in range(len(cols)):
                v = grid[i, j]
                if np.isnan(v):
                    continue
                txt = f"{v:.1f}" if v < 10 else f"{v:.0f}"
                textcol = "white" if (v > 30 or v < 0.7) else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8.5, color=textcol, fontweight="bold")
    handles, labels = axA.get_legend_handles_labels()
    if not handles:
        for ax_top in (axB, axC):
            h, l = ax_top.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
    if handles:
        fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.52),
                   ncol=min(3, len(handles)), frameon=True, fontsize=16)

    sm = mcm.ScalarMappable(norm=norm_speed, cmap=cmap_speed)
    bottom_axes = [ax for _, ax, *_ in grids]
    cbar = fig.colorbar(sm, ax=bottom_axes,
                        orientation="horizontal", fraction=0.05, pad=0.22,
                        aspect=40, shrink=0.75)
    cbar.set_label("Speedup factor", labelpad=8, fontsize=14)
    tick_candidates = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0]
    ticks = [t for t in tick_candidates if vmin <= t <= vmax]
    if 1.0 not in ticks and vmin < 1 < vmax:
        ticks.append(1.0)
    ticks = sorted(set(ticks))
    cbar.set_ticks(ticks)
    cbar.ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=12)
    cbar.ax.axvline(x=norm_speed(1.0), color="k", lw=1.5, ls="--")

    base = os.path.join(OUTDIR, "Figure_6")
    for ext in (".pdf", ".svg", ".png"):
        fig.savefig(base + ext)
    plt.close(fig)
    print(f"[OK] saved: {base}.{{pdf,svg,png}}")


def save_individual_panels(df: pd.DataFrame):
    set_style()

    # ---- Panels A, B, C: Runtime plots ----
    for nag, lbl in zip(AGE_PANELS, list("ABC")):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        sub = df[df["NumAgeGroups"] == nag]
        if sub.empty:
            plt.close(fig)
            continue

        ref_method = "Flow-based" if (sub["Method"] ==
                                      "Flow-based").any() else sub["Method"].iloc[0]
        sub_ref = sub[sub["Method"] == ref_method]
        x_vals = sorted(sub_ref["NumPatches"].unique())
        x_anchor = x_vals[0]
        y_vals_at_anchor = sub_ref[sub_ref["NumPatches"]
                                   == x_anchor]["time_value"]
        y_anchor = float(y_vals_at_anchor.min()) * \
            0.3
        _add_scaling_refs(ax, x_vals, y_anchor, x_anchor)
        _lines_for_methods(ax, sub)

        ax.set_xlabel("Number of patches")
        ax.set_ylabel(
            "Time per simulated day [s]" if PER_DAY else "Total time [s]")
        ax.set_title(f"{nag} age group{'s' if nag != 1 else ''}")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.text(-0.09, 1.06, lbl, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=22, fontweight="bold", clip_on=False)

        base = os.path.join(OUTDIR, f"Figure_6_{lbl}")
        fig.savefig(base + ".pdf", bbox_inches='tight', dpi=300)
        fig.savefig(base + ".png", bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"[OK] saved: Figure_6_{lbl}.{{pdf,png}}")

    # ---- Panels D, E, F: Speedup heatmaps ----
    gridF_new, rowsF_new, colsF_new = _make_speedup_grid(
        df, num="Standard Lagrangian (Euler)", den="stage-aligned flow-based (Euler)")
    titleF = "Euler: Lagr. / stage-aligned"

    gridE, rowsE, colsE = _make_speedup_grid(
        df, num="Standard Lagrangian (RK-4)", den="stage-aligned flow-based (RK-4)")
    titleE = "RK-4: Lagr. / stage-aligned"
    gridD_new, rowsD_new, colsD_new = _make_speedup_grid(
        df, num="auxiliary Euler", den="hybrid flow-based")
    titleD = "auxiliary / hybrid"

    grids = [
        ("F", gridF_new, rowsF_new, colsF_new, titleF),
        ("E", gridE, rowsE, colsE, titleE),
        ("D", gridD_new, rowsD_new, colsD_new, titleD)
    ]

    vals = []
    for _, grid, _, _, _ in grids:
        if grid is not None:
            vals.extend(list(np.ravel(grid[np.isfinite(grid)])))

    if vals:
        raw_min, raw_max = min(vals), max(vals)
    else:
        raw_min, raw_max = 0.5, 2.0

    vmin = max(1e-6, min(raw_min, 0.5))
    vmax = max(raw_max, 50.0)
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = max(vmin * 2, 50.0)
    norm_speed = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap_speed = make_rdylgn_centered(vmin, vmax, center=1.0)

    for lbl, grid, rows, cols, title in grids:
        if grid is None:
            continue

        max_speedup = np.nanmax(grid)
        min_speedup = np.nanmin(grid)
        print(f"\n[Panel {lbl}] {title}")
        print(f"  Max Speedup: {max_speedup:.2f}x")
        print(f"  Min Speedup: {min_speedup:.2f}x")

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        im = ax.imshow(grid, aspect="auto", origin="lower",
                       cmap=cmap_speed, norm=norm_speed, interpolation="nearest")
        ax.set_title(title, pad=4)
        ax.set_xlabel("Number of patches")
        ax.set_ylabel("Age groups")
        ax.set_xticks(list(range(len(cols))))
        ax.set_xticklabels([str(c) for c in cols], rotation=45, ha="right")
        ax.set_yticks(list(range(len(rows))))
        ax.set_yticklabels([str(r) for r in rows])
        ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color="w",
                linestyle="-", linewidth=0.6, alpha=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.text(-0.09, 1.06, lbl, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=22, fontweight="bold", clip_on=False)
        # Cell annotations
        for i in range(len(rows)):
            for j in range(len(cols)):
                v = grid[i, j]
                if np.isnan(v):
                    continue
                txt = f"{v:.1f}" if v < 10 else f"{v:.0f}"
                textcol = "white" if (v > 30 or v < 0.7) else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8.5, color=textcol, fontweight="bold")
        base = os.path.join(OUTDIR, f"Figure_6_{lbl}")
        fig.savefig(base + ".pdf", bbox_inches='tight', dpi=300)
        fig.savefig(base + ".png", bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"[OK] saved: Figure_6_{lbl}.{{pdf,png}}")

    fig_leg, ax_leg = plt.subplots(1, 1, figsize=(10, 1))
    ax_leg.axis('off')

    handles = []
    labels = []
    for m in METHODS_MAIN:
        style = STYLES[m].copy()
        display_name = m
        line = mlines.Line2D([], [], color=COLORS[m],
                             label=display_name, **style)
        handles.append(line)
        labels.append(display_name)

    legend = ax_leg.legend(handles, labels, loc='center',
                           ncol=3, frameon=True, fontsize=16)

    base = os.path.join(OUTDIR, "Figure_6_Legend")
    fig_leg.savefig(base + ".pdf", bbox_inches='tight', dpi=300)
    fig_leg.savefig(base + ".png", bbox_inches='tight', dpi=300)
    plt.close(fig_leg)
    print(f"[OK] saved: Figure_6_Legend.{{pdf,png}}")

    fig_cbar, ax_cbar = plt.subplots(1, 1, figsize=(10, 1))
    ax_cbar.axis('off')

    sm = mcm.ScalarMappable(norm=norm_speed, cmap=cmap_speed)
    cbar = fig_cbar.colorbar(sm, ax=ax_cbar, orientation="horizontal",
                             fraction=0.8, aspect=30, pad=0.05)
    cbar.set_label("Speedup factor", labelpad=10, fontsize=16)

    tick_candidates = [0.5, 1.0, 2.0, 5.0,
                       10.0, 20.0, 50.0, 100.0, 500.0, 1000.0, 6000.0, 10000.0, 25000.0]
    ticks = [t for t in tick_candidates if vmin <= t <= vmax]
    if 1.0 not in ticks and vmin < 1 < vmax:
        ticks.append(1.0)
    if vmax not in ticks:
        ticks.append(vmax)
    ticks = sorted(set(ticks))
    cbar.set_ticks(ticks)
    cbar.ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=14)

    base = os.path.join(OUTDIR, "Figure_6_Colorbar")
    fig_cbar.savefig(base + ".pdf", bbox_inches='tight', dpi=300)
    fig_cbar.savefig(base + ".png", bbox_inches='tight', dpi=300)
    plt.close(fig_cbar)
    print(f"[OK] saved: Figure_6_Colorbar.{{pdf,png}}")


# ------------------------------ Main ------------------------------------------


def main():
    set_style()
    df = load_benchmark(CSV_BENCH)
    # figure_multipanel_combined(df)
    save_individual_panels(df)


if __name__ == "__main__":
    main()
