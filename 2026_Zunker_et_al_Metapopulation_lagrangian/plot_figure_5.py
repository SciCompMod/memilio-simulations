import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
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


def make_rdylgn_centered(vmin: float, vmax: float, center: float = 1.0):
    """Return a blue→yellow→green colormap where `center` shows as yellow.

    Use together with a standard LogNorm(vmin, vmax).
    """
    pivot = np.log(center / vmin) / np.log(vmax / vmin)
    pivot = float(np.clip(pivot, 0.01, 0.99))
    n = 512
    n_lo = max(1, int(round(n * pivot)))
    n_hi = n - n_lo
    mid = len(_BLYIGN_COLORS) // 2 
    cmap_lo = LinearSegmentedColormap.from_list(
        "lo", _BLYIGN_COLORS[:mid + 1], N=n_lo)
    cmap_hi = LinearSegmentedColormap.from_list(
        "hi", _BLYIGN_COLORS[mid:],    N=n_hi)
    colors = np.vstack([
        cmap_lo(np.linspace(0.0, 1.0, n_lo)),
        cmap_hi(np.linspace(0.0, 1.0, n_hi)),
    ])
    return LinearSegmentedColormap.from_list("BlYlGn_c", colors, N=n)


# -- Paths ---------------------------------------------------------------------
CSV_PATH_DT1 = "benchmark_t05_dt1.csv"
CSV_PATH_ADAPTIVE = "benchmark_t05_dt_adaptiv.csv"

OUTDIR = os.path.join("saves", "Figures", "Figure5")
os.makedirs(OUTDIR, exist_ok=True)

# -- Patch index → number of patches -------------------------------------------
PATCH_MAP = {i: v for i, v in enumerate([16, 32, 64, 128, 256, 512, 1024])}

# -- Panel definitions: (num_method, den_method, csv_path, title) -------------
PANELS = [
    (
        "matrix_phi_reconstruction(Euler)",
        "stage-aligned(Euler)",
        CSV_PATH_DT1,
        "Euler",
    ),
    (
        "matrix_phi_reconstruction(RK4)",
        "stage-aligned(RK4)",
        CSV_PATH_DT1,
        "RK-4 (fixed)",
    ),
    (
        "matrix_phi_reconstruction(RK4)",
        "stage-aligned(RK4)",
        CSV_PATH_ADAPTIVE,
        "RK-4 (adaptive)",
    ),
]

# -- RC settings (matching Figure 4 style) ------------------------------------
def set_style():
    mpl.rcParams.update({
        "font.size":          16,
        "axes.titlesize":     22,
        "axes.labelsize":     18,
        "xtick.labelsize":    14,
        "ytick.labelsize":    14,
        "legend.fontsize":    16,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "axes.grid":          False,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })


# -- Data loading --------------------------------------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    parts = df["name"].astype(str).str.strip('"').str.split("/")
    df = df[parts.map(len) == 3].copy()
    parts = df["name"].astype(str).str.strip('"').str.split("/")
    df["method"] = parts.map(lambda x: x[0])
    df["patch_idx"] = parts.map(lambda x: int(x[1]))
    df["num_age"] = parts.map(lambda x: int(x[2]))
    df["num_patches"] = df["patch_idx"].map(PATCH_MAP)
    df["time_us"] = pd.to_numeric(df["real_time"], errors="coerce")
    df = df.dropna(subset=["time_us", "num_patches"])
    return df[["method", "num_age", "num_patches", "time_us"]].copy()


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    agg = (df.groupby(["method", "num_age", "num_patches"])["time_us"]
             .median()
             .reset_index()
             .rename(columns={"time_us": "time"}))
    agg["time"] = agg["time"] / 1000.0   # µs → ms
    return agg


# -- Speedup grid helper -------------------------------------------------------
def _build_grid(df_agg: pd.DataFrame, num: str, den: str):
    """Compute speedup grid: time(num) / time(den).

    Returns (grid, ages, patches) or (None, None, None) if data is missing.
    """
    piv = (df_agg
           .pivot_table(index=["num_age", "num_patches"],
                        columns="method", values="time", aggfunc="median")
           .reset_index())
    if num not in piv.columns or den not in piv.columns:
        return None, None, None, None
    piv = piv.dropna(subset=[num, den]).copy()
    piv["speedup"] = piv[num] / piv[den]

    ages = sorted(piv["num_age"].unique())
    patches = sorted(piv["num_patches"].unique())
    grid = np.full((len(ages), len(patches)), np.nan)
    # absolute runtime of den (ms)
    den_grid = np.full((len(ages), len(patches)), np.nan)
    for i, a in enumerate(ages):
        for j, p in enumerate(patches):
            sel = piv[(piv["num_age"] == a) & (piv["num_patches"] == p)]
            if not sel.empty:
                grid[i, j] = sel["speedup"].values[0]
                den_grid[i, j] = sel[den].values[0]
    return grid, ages, patches, den_grid


# -- Single heatmap panel (no colorbar) ---------------------------------------
def _plot_heatmap(ax, grid, ages, patches, norm, cmap, rt_grid=None):
    """Draw the heatmap image + cell annotations + axis labels.

    rt_grid: optional 2-D array (same shape as grid) with the absolute
             runtime of the denominator method (stage-aligned) in ms.
             If given, white dashed isolines are drawn at 0.1/1/10/100 ms.
    """
    ax.imshow(grid, aspect="auto", origin="lower",
              norm=norm, cmap=cmap,
              extent=[-0.5, len(patches) - 0.5, -0.5, len(ages) - 0.5])

    for i in range(len(ages)):
        for j in range(len(patches)):
            v = grid[i, j]
            if np.isnan(v):
                continue
            txt = f"{v:.1f}" if v < 10 else f"{v:.0f}"
            textcol = "white" if (v > 10 or v < 0.75) else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=11, color=textcol, fontweight="bold")

    # -- Isolines of absolute stage-aligned runtime ----------------------------
    if rt_grid is not None:
        X = np.arange(len(patches))
        Y = np.arange(len(ages))
        all_levels = [0.1, 1.0, 10.0, 100.0]
        data_min = np.nanmin(rt_grid)
        data_max = np.nanmax(rt_grid)
        levels = [lv for lv in all_levels if data_min *
                  0.5 < lv < data_max * 2.0]
        if levels:
            cs = ax.contour(X, Y, rt_grid, levels=levels,
                            colors="white", linewidths=2.2,
                            linestyles="-", alpha=1.0, zorder=4)
            fmt = {lv: f"{lv:g} ms" for lv in levels}
            cl = ax.clabel(cs, fmt=fmt, fontsize=11, inline=True,
                           inline_spacing=4)
            for txt in cl:
                txt.set_color("white")
                txt.set_fontweight("bold")
                txt.set_alpha(0.75)
                txt.set_bbox(dict(boxstyle="round,pad=0.15",
                                  fc="black", ec="none", alpha=0.25))

    ax.set_xticks(range(len(patches)))
    ax.set_xticklabels(patches, rotation=45, ha="right")
    ax.set_yticks(range(len(ages)))
    ax.set_yticklabels(ages)
    ax.set_xlabel("Number of patches")
    ax.set_ylabel("Age groups")


# -- Main figure ---------------------------------------------------------------
def build_figure(outname: str = "Figure_Bench"):
    set_style()

    # -- Load data -------------------------------------------------------------
    data = {
        CSV_PATH_DT1:      _agg(load_data(CSV_PATH_DT1)),
        CSV_PATH_ADAPTIVE: _agg(load_data(CSV_PATH_ADAPTIVE)),
    }

    # -- Pre-compute grids & global color scale --------------------------------
    all_vals = []
    grids_meta = []
    for num, den, csv_path, title in PANELS:
        grid, ages, patches, rt_grid = _build_grid(data[csv_path], num, den)
        grids_meta.append((grid, ages, patches, title, rt_grid))
        if grid is not None:
            all_vals.extend(grid[~np.isnan(grid)].tolist())

    vmin = max(0.5, min(all_vals) * 0.90) if all_vals else 0.5
    vmax = max(all_vals) if all_vals else 30.0
    vmax = max(vmax, 30.0)

    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = make_rdylgn_centered(vmin, vmax, center=1.0)

    # -- Figure geometry: axes sized so every cell is square -------------------
    n_ag = 9    # age-group rows
    n_pat = 7    # patch-count columns
    cell = 0.50  # target cell size in inches → square by construction

    heat_w = n_pat * cell   # = 3.50"
    heat_h = n_ag * cell   # = 4.50"

    # All margins in inches:
    mL = 0.92   # left of each axes  (y-label + ytick labels)
    mR = 0.18   # right of figure
    mT = 0.72   # above axes         (panel letter + title)
    mB = 0.88   # below axes         (rotated xtick labels + xlabel)
    gX = 1.10   # horizontal gap between columns
    cbH = 0.88   # height of the colorbar strip at the very bottom

    fig_w = mL + 3*heat_w + 2*gX + mR
    fig_h = mT + heat_h + mB + cbH

    fig = plt.figure(figsize=(fig_w, fig_h))

    def _add_ax(col):
        """Return an axes at the given column position (single row)."""
        x0 = (mL + col * (heat_w + gX)) / fig_w
        y0 = (cbH + mB) / fig_h
        return fig.add_axes((x0, y0, heat_w / fig_w, heat_h / fig_h))

    ax_list = [_add_ax(0), _add_ax(1), _add_ax(2)]

    for ax, lbl, (grid, ages, patches, title, rt_grid) in zip(ax_list, "ABC", grids_meta):
        if grid is None:
            ax.text(0.5, 0.5, "Data missing", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            _plot_heatmap(ax, grid, ages, patches, norm, cmap, rt_grid=rt_grid)

        ax.set_title(title, pad=4)
        ax.text(-0.13, 1.04, lbl, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=22,
                fontweight="bold", clip_on=False)

    # -- Shared horizontal colorbar (manually placed at the bottom) ------------
    # Colorbar spans 80% of the total data width, centred.
    data_span = 3*heat_w + 2*gX
    cb_w_inch = data_span * 0.78
    cb_l_inch = mL + (data_span - cb_w_inch) / 2
    cb_b_inch = 0.36   # from figure bottom
    cb_h_inch = 0.22   # height of the coloured bar strip

    cbar_ax = fig.add_axes((
        cb_l_inch / fig_w,
        cb_b_inch / fig_h,
        cb_w_inch / fig_w,
        cb_h_inch / fig_h,
    ))
    sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"$\Phi$-matrix / Stage-aligned runtime ratio",
                   labelpad=8, fontsize=17)
    tick_candidates = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
    ticks = [t for t in tick_candidates if vmin <= t <= vmax]
    if 1.0 not in ticks and vmin < 1.0 < vmax:
        ticks.append(1.0)
    ticks = sorted(set(ticks))
    cbar.set_ticks(ticks)
    cbar.ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=14)
    cbar.ax.axvline(x=float(norm(1.0)), color="k", lw=1.5, ls="--")

    # -- Save ------------------------------------------------------------------
    for ext in ("pdf", "png"):
        path = os.path.join(OUTDIR, f"{outname}.{ext}")
        fig.savefig(path)


if __name__ == "__main__":
    build_figure()
