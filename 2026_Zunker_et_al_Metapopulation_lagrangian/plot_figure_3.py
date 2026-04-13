import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


DIR = "saves/"
CSV = os.path.join(DIR, "commuter_metrics.csv")
OUTDIR = os.path.join(DIR, "Figures", "Figure3")
os.makedirs(OUTDIR, exist_ok=True)

SCENARIO_A = "_stress"
SCENARIO_B = "_cp"
COMP_TO_PLOT = "E"
SELECT_PC_FOR_D = 0.95


def set_style():
    mpl.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.55,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _coerce_time_series_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Time" not in df.columns:
        df.rename(columns={df.columns[0]: "Time"}, inplace=True)
    remaining = [c for c in df.columns if c != "Time"]
    if len(remaining) >= 4:
        sub = remaining[:4]
        tmp = df[["Time"] + sub].copy()
        tmp.columns = ["Time", "S", "E", "I", "R"]
        return tmp
    name_map = {}
    for c in remaining:
        key = c.strip().upper()[0]
        if key in ("S", "E", "I", "R") and key not in name_map:
            name_map[key] = c
    out = {"Time": df["Time"]}
    for k in ["S", "E", "I", "R"]:
        out[k] = df[name_map[k]] if k in name_map else pd.Series(np.nan, index=df.index)
    return pd.DataFrame(out)


def load_timeseries_for_AB(base_dir: str, scenario_suffix: str):
    paths = {
        "ref": os.path.join(base_dir, f"mobile_ref_solution{scenario_suffix}.csv"),
        "euler": os.path.join(base_dir, f"mobile_euler_solution{scenario_suffix}.csv"),
        "total": os.path.join(base_dir, f"seir_solution{scenario_suffix}.csv"),
    }
    data = {}
    for key, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Fehlende Datei: {p}")
        df = pd.read_csv(p)
        data[key] = _coerce_time_series_columns(df)
    return data["ref"], data["euler"], data["total"]


def _plot_AB_core(ax, df_ref, df_euler, df_total, comp: str = "E"):
    lw = 3.0
    ax.plot(df_ref["Time"].to_numpy(), df_ref[comp].astype(float).to_numpy(), "-", color="black", linewidth=lw, label="Reference")
    ax.plot(df_euler["Time"].to_numpy(), df_euler[comp].astype(float).to_numpy(), "--", color="blue", linewidth=lw, label="Euler")
    ax.plot(df_total["Time"].to_numpy(), df_total[comp].astype(float).to_numpy(), "-.", color="red", linewidth=lw, alpha=0.8, label="Patch total")

    common_t = np.intersect1d(df_ref["Time"].to_numpy(), df_total["Time"].to_numpy())
    if common_t.size > 0:
        ref_f = df_ref[df_ref["Time"].isin(common_t)].sort_values("Time")
        tot_f = df_total[df_total["Time"].isin(common_t)].sort_values("Time")
        ax.fill_between(ref_f["Time"].to_numpy(), ref_f[comp].to_numpy(), tot_f[comp].to_numpy(), color="#ffeb8a", alpha=0.35, linewidth=0.0, zorder=1)

    ax.legend(loc="best")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(f"{comp}(t)")
    ax.grid(True, linestyle="--", alpha=0.6)

    xmin = min(df_ref["Time"].min(), df_euler["Time"].min(), df_total["Time"].min())
    xmax = max(df_ref["Time"].max(), df_euler["Time"].max(), df_total["Time"].max())
    ax.set_xlim(xmin, xmax)
    ymin = min(df_ref[comp].min(), df_euler[comp].min(), df_total[comp].min())
    ymax = max(df_ref[comp].max(), df_euler[comp].max(), df_total[comp].max())
    ax.set_ylim(ymin, ymax * 1.01)


def panel_A(ax, scenario_suffix: str = SCENARIO_A, comp: str = COMP_TO_PLOT):
    df_ref, df_euler, df_total = load_timeseries_for_AB(DIR, scenario_suffix)
    _plot_AB_core(ax, df_ref, df_euler, df_total, comp)


def panel_B(ax, scenario_suffix: str = SCENARIO_B, comp: str = COMP_TO_PLOT):
    df_ref, df_euler, df_total = load_timeseries_for_AB(DIR, scenario_suffix)
    _plot_AB_core(ax, df_ref, df_euler, df_total, comp)


def load_metrics(path):
    df = pd.read_csv(path)
    needed = {"p_c", "dt", "n_steps", "viol_frac", "Linf_all"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing cols in CSV: {sorted(missing)}")
    return df


def panel_C(ax, df, viol_thresh=0.02, do_interpolate=True):
    df2 = df.sort_values(["p_c", "dt"]).copy()
    pcs = np.sort(df2["p_c"].unique())
    dts = np.sort(df2["dt"].unique())
    xmin, xmax = float(dts.min()), float(dts.max())

    dt_crit = []
    for pc in pcs:
        sub = df2[df2["p_c"].eq(pc)].sort_values("dt")
        d = sub["dt"].to_numpy(float)
        v = sub["viol_frac"].to_numpy(float)
        idx = np.where(v >= viol_thresh)[0]
        if idx.size == 0:
            dt_crit.append(np.nan)
            continue
        i_hi = idx[0]
        if not do_interpolate or i_hi == 0:
            dt_crit.append(d[i_hi])
        else:
            i_lo = i_hi - 1
            d_lo, d_hi = d[i_lo], d[i_hi]
            v_lo, v_hi = v[i_lo], v[i_hi]
            if v_hi == v_lo:
                dt_crit.append(d_hi)
            else:
                x_lo, x_hi = np.log10(d_lo), np.log10(d_hi)
                w = (viol_thresh - v_lo) / (v_hi - v_lo)
                x_star = x_lo + w * (x_hi - x_lo)
                dt_crit.append(10.0**x_star)

    dt_crit = np.array(dt_crit, dtype=float)
    cutoff = np.where(np.isnan(dt_crit), xmax, dt_crit)

    ax.fill_betweenx(pcs, xmin, cutoff, facecolor="#2f2554", alpha=0.85)
    ax.fill_betweenx(pcs, cutoff, xmax, facecolor="#ffd24d", alpha=0.75)
    ax.plot(dt_crit, pcs, color="white", lw=2.5, marker="o", ms=3, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel(r"Step size (days)")
    ax.set_ylabel(r"$p_c$")
    ax.grid(True, linestyle=":", alpha=0.55)


def panel_D(ax, df, pc=0.9):
    sub = df[np.isclose(df["p_c"], pc)]
    if sub.empty:
        raise ValueError(f"No rows found for p_c={pc}.")
    sub = sub.groupby("dt", as_index=False)["Linf_all"].max().sort_values("dt")
    x = sub["dt"].values.astype(float)
    y = sub["Linf_all"].values.astype(float)
    ax.loglog(x, y, "o-", lw=3)
    mid = len(x)//2
    k = y[mid] / x[mid] if x[mid] > 0 else y[-1] / max(x[-1], 1e-12)
    refx = np.array([x.min(), x.max()])
    ax.loglog(refx, k*refx, ":", color="0.45", lw=2)
    ax.set_xlabel(r"Step size (days)")
    ax.set_ylabel("Max absolute error")
    ax.grid(True, linestyle=":", alpha=0.55)


def main():
    set_style()
    fig_all, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axA, axB, axC, axD = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    panel_A(axA, SCENARIO_A, COMP_TO_PLOT)
    panel_B(axB, SCENARIO_B, COMP_TO_PLOT)
    df_metrics = load_metrics(CSV)
    panel_C(axC, df_metrics)
    panel_D(axD, df_metrics, pc=SELECT_PC_FOR_D)
    for ax, label in zip([axA, axB, axC, axD], "ABCD"):
        ax.text(-0.08, 1.04, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=22, fontweight="bold", clip_on=False)
    base_all = os.path.join(OUTDIR, "Figure_3_ABCD")
    for ext in (".pdf", ".svg", ".png"):
        fig_all.savefig(base_all + ext)
    fig_cd, axes_cd = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    panel_C(axes_cd[0], df_metrics)
    panel_D(axes_cd[1], df_metrics, pc=SELECT_PC_FOR_D)
    base_cd = os.path.join(OUTDIR, "Figure_3_CD")
    for ext in (".pdf", ".svg", ".png"):
        fig_cd.savefig(base_cd + ext)
    for letter, func, scen in (("A", panel_A, SCENARIO_A), ("B", panel_B, SCENARIO_B)):
        fig_single, ax_single = plt.subplots(1, 1, figsize=(6, 4.5), constrained_layout=True)
        func(ax_single, scen, COMP_TO_PLOT)
        base_single = os.path.join(OUTDIR, f"Figure_3_{letter}")
        for ext in (".pdf", ".svg", ".png"):
            fig_single.savefig(base_single + ext)
    print("[OK] saved:", base_all + ".{pdf,svg,png}", base_cd + ".{pdf,svg,png}")

if __name__ == "__main__":
    main()
