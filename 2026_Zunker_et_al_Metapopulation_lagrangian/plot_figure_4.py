import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, LogLocator, FormatStrFormatter

# ----------------  Data----------------
DIR_CSV = "saves//"
PLOT_DIR = "saves//Figures//Figure4//"
os.makedirs(PLOT_DIR, exist_ok=True)

FLOW_CSV = DIR_CSV + "flow_commuter_compare_sol.csv"
EXPL_CSV = DIR_CSV + "explicit_commuter_compare_sol.csv"
FLOW_EXACT_CSV = DIR_CSV + "flow_exact_commuter_compare_sol.csv"

FLOW_CSV_FMT = DIR_CSV + "flow_commuter_compare_sol_dt_{tag}.csv"
EXPL_CSV_FMT = DIR_CSV + "explicit_commuter_compare_sol_dt_{tag}.csv"
FLOW_EXACT_CSV_FMT = DIR_CSV + "flow_exact_commuter_compare_sol_dt_{tag}.csv"

# Euler-specific files
FLOW_CSV_EULER_FMT = DIR_CSV + "flow_commuter_compare_sol_euler_dt_{tag}.csv"
EXPL_CSV_EULER_FMT = DIR_CSV + \
    "explicit_commuter_compare_sol_euler_dt_{tag}.csv"

DT_LIST = [2.0, 1.0, 0.5, 0.25, 0.125] 

METHODS = {
    'euler': {'label': 'Euler', 'color': '#9467bd', 'marker': 'o', 'order': 1},
    'rk2':   {'label': 'RK-2', 'color': '#8c564b', 'marker': 's', 'order': 2},
    'rk3':   {'label': 'RK-3', 'color': '#e377c2', 'marker': '^', 'order': 3},
    'rk4':   {'label': 'RK-4', 'color': '#7f7f7f', 'marker': 'D', 'order': 4},
}
DTS_CONVERGENCE = [2.0, 1.0, 0.5, 0.25, 0.125]  # Convergence plot range


def set_settings():
    mpl.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.55,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _fmt_tag(dt: float) -> str:
    return f"{dt:.3f}"

# ---------------- Load data ---------------------------


def _read_flow_commuter(path_flow: str) -> pd.DataFrame:
    df = pd.read_csv(path_flow)
    if "Time" not in df.columns:
        df.rename(columns={df.columns[0]: "Time"}, inplace=True)
    cols = ["Time", "S", "E", "I", "R"]
    if not set(cols).issubset(df.columns):
        df = df.iloc[:, :5]
        df.columns = cols
    else:
        df = df[cols]
    return df


def _read_flow_exact_commuter(path_flow: str) -> pd.DataFrame:
    """Read FLOW_EXACT CSV with the same 5 columns as _read_flow_commuter."""
    df = pd.read_csv(path_flow)
    if "Time" not in df.columns:
        df.rename(columns={df.columns[0]: "Time"}, inplace=True)
    cols = ["Time", "S", "E", "I", "R"]
    if not set(cols).issubset(df.columns):
        df = df.iloc[:, :5]
        df.columns = cols
    else:
        df = df[cols]
    return df


def _read_explicit_commuter(path_expl: str) -> pd.DataFrame:
    df = pd.read_csv(path_expl)
    if "Time" not in df.columns:
        df.rename(columns={df.columns[0]: "Time"}, inplace=True)
    if df.shape[1] < 9:
        raise ValueError(
            "Explicit CSV expects (Time, 4 locals, 4 commuters).")
    out = df[["Time"] + list(df.columns[-4:])].copy()
    out.columns = ["Time", "S", "E", "I", "R"]
    return out


def _read_flow_commuter_euler(path_flow: str) -> pd.DataFrame:
    """Read Euler flow-based commuter data."""
    df = pd.read_csv(path_flow)
    if "Time" not in df.columns:
        df.rename(columns={df.columns[0]: "Time"}, inplace=True)
    cols = ["Time", "S", "E", "I", "R"]
    if not set(cols).issubset(df.columns):
        df = df.iloc[:, :5]
        df.columns = cols
    else:
        df = df[cols]
    return df


def _read_explicit_commuter_euler(path_expl: str) -> pd.DataFrame:
    """Read Euler explicit commuter data."""
    df = pd.read_csv(path_expl)
    if "Time" not in df.columns:
        df.rename(columns={df.columns[0]: "Time"}, inplace=True)
    if df.shape[1] < 9:
        raise ValueError(
            "Explicit CSV expects (Time, 4 locals, 4 commuters).")
    out = df[["Time"] + list(df.columns[-4:])].copy()
    out.columns = ["Time", "S", "E", "I", "R"]
    return out


def _read_explicit_full(path_expl: str):
    df = pd.read_csv(path_expl)
    if "Time" not in df.columns:
        df.rename(columns={df.columns[0]: "Time"}, inplace=True)
    time = df["Time"].to_numpy()
    if df.shape[1] < 9:
        raise ValueError(
            "Explicit CSV expects (Time, 4 locals, 4 commuters).")
    locals_mat = df[df.columns[1:5]].to_numpy()
    comm_mat = df[df.columns[5:9]].to_numpy()
    totals_mat = locals_mat + comm_mat
    return time, locals_mat, comm_mat, totals_mat


def _load_pair_baseline():
    if not os.path.exists(FLOW_CSV) or not os.path.exists(EXPL_CSV):
        tag = _fmt_tag(1.0)
        f_flow = FLOW_CSV_FMT.format(tag=tag)
        f_expl = EXPL_CSV_FMT.format(tag=tag)
        dfF = _read_flow_commuter(f_flow)
        dfX = _read_explicit_commuter(f_expl)
        df_full_time, locals_mat, comm_mat, totals_mat = _read_explicit_full(
            f_expl)
    else:
        dfF = _read_flow_commuter(FLOW_CSV)
        dfX = _read_explicit_commuter(EXPL_CSV)
        df_full_time, locals_mat, comm_mat, totals_mat = _read_explicit_full(
            EXPL_CSV)
    df = pd.merge(dfF, dfX, on="Time", suffixes=("_Flow", "_Explicit"))
    return df, df_full_time, locals_mat, comm_mat, totals_mat


def _load_pair_for_dt(dt: float):
    tag = _fmt_tag(dt)
    df_flow = _read_flow_commuter(FLOW_CSV_FMT.format(tag=tag))
    df_expl = _read_explicit_commuter(EXPL_CSV_FMT.format(tag=tag))
    return pd.merge(df_flow, df_expl, on="Time", suffixes=("_Flow", "_Explicit"))


def _load_pair_for_dt_exact(dt: float):
    """Merge FLOW_EXACT (as FlowExact) with Explicit for a given dt tag."""
    tag = _fmt_tag(dt)
    df_flow_exact = _read_flow_exact_commuter(
        FLOW_EXACT_CSV_FMT.format(tag=tag))
    df_expl = _read_explicit_commuter(EXPL_CSV_FMT.format(tag=tag))
    return pd.merge(df_flow_exact, df_expl, on="Time", suffixes=("_FlowExact", "_Explicit"))


def _load_exact_baseline():
    """Load baseline pair for FLOW_EXACT vs Explicit (prefer non-tagged, fallback to dt=1.0)."""
    if not os.path.exists(FLOW_EXACT_CSV) or not os.path.exists(EXPL_CSV):
        tag = _fmt_tag(1.0)
        f_flow = FLOW_EXACT_CSV_FMT.format(tag=tag)
        f_expl = EXPL_CSV_FMT.format(tag=tag)
        dfF = _read_flow_exact_commuter(f_flow)
        dfX = _read_explicit_commuter(f_expl)
    else:
        dfF = _read_flow_exact_commuter(FLOW_EXACT_CSV)
        dfX = _read_explicit_commuter(EXPL_CSV)
    return pd.merge(dfF, dfX, on="Time", suffixes=("_FlowExact", "_Explicit"))


def load_convergence_data_all_methods(dts, phi=False):
    """Load convergence data for all methods (Euler, RK2, RK3, RK4)."""
    all_data = {}
    fn = 'convergence_phi' if phi else 'convergence'
    for method_name in METHODS.keys():
        data = []
        for dt in dts:
            try:
                filename = os.path.join(
                    DIR_CSV, f"{fn}_{method_name}_dt_{dt:.4f}.csv")
                df = pd.read_csv(filename)
                if len(df) > 0:
                    row = df.iloc[0]
                    data.append({
                        'dt': row['dt'],
                        'max_abs_error': row['max_abs_error'],
                        'max_rel_error': row['max_rel_error'],
                    })
            except FileNotFoundError:
                print(
                    f"[Warning] Convergence file not found: {method_name} dt={dt}")
            except Exception as e:
                print(
                    f"[Warning] Error loading convergence {method_name} dt={dt}: {e}")

        if data:
            all_data[method_name] = pd.DataFrame(data)

    return all_data


def _max_errors_over_time(df: pd.DataFrame):
    comps = ["E", "I", "R"]  # S is intentionally omitted
    abs_max, rel_max = [], []
    eps = 1e-12
    by_comp = {}
    for c in comps:
        diff = (df[f"{c}_Flow"] - df[f"{c}_Explicit"]).to_numpy()
        a = np.abs(diff)
        abs_max.append(np.max(a))
        denom = np.maximum(eps, np.abs(df[f"{c}_Explicit"].to_numpy()))
        rel = np.max(a / denom)
        rel_max.append(rel)
        by_comp[c] = np.max(a)
    return float(np.max(abs_max)), float(np.max(rel_max)), by_comp

# --------------- Convergence ------------------------


def compute_convergence_metrics(dt_list=DT_LIST):
    data = []
    for dt in dt_list:
        try:
            df = _load_pair_for_dt(dt)
            absM, relM, abs_by_comp = _max_errors_over_time(df)
            data.append({"dt": dt, "abs_max": absM,
                        "rel_max": relM, "abs_by_comp": abs_by_comp})
        except FileNotFoundError as e:
            print(f"[Warnung] dt={dt} übersprungen: {e}")
    data.sort(key=lambda d: d["dt"])
    return data


def compute_convergence_metrics_exact(dt_list=DT_LIST):
    """Convergence metrics using FLOW_EXACT vs Explicit (finest reference)."""
    data = []
    # Load finest explicit solution as reference (fixed dt=0.0001)
    tag_ref = "0.000"  # corresponds to dt=0.0001
    df_expl_ref = _read_explicit_commuter(EXPL_CSV_FMT.format(tag=tag_ref))

    for dt in dt_list:
        try:
            tag = _fmt_tag(dt)
            df_flow_exact = _read_flow_exact_commuter(
                FLOW_EXACT_CSV_FMT.format(tag=tag))
            df = pd.merge(df_flow_exact, df_expl_ref, on="Time",
                          suffixes=("_FlowExact", "_Explicit"))
            absM, relM, abs_by_comp = _max_errors_over_time(
                df.rename(columns={
                    "S_FlowExact": "S_Flow",
                    "E_FlowExact": "E_Flow",
                    "I_FlowExact": "I_Flow",
                    "R_FlowExact": "R_Flow",
                })
            )
            data.append({"dt": dt, "abs_max": absM,
                        "rel_max": relM, "abs_by_comp": abs_by_comp})
        except FileNotFoundError as e:
            print(f"[Warning][EXACT] dt={dt} skipped: {e}")
    data.sort(key=lambda d: d["dt"])
    return data

# --------------- Multipanel 2×3 (A–F) --------------


def build_multipanel_figure(dt_list=DT_LIST, outname="Figure_4_multipanel"):
    set_settings()

    # Colors for compartments (A–D, F) and metrics (E)
    color_map = {"S": "#1f77b4", "E": "#ff7f0e",
                 "I": "#2ca02c", "R": "#d62728"}
    abs_col, rel_col, ref_col = "#6A3D9A", "#1B9E77", "0.45"
    lw_main, lw_dash = 3.0, 4.5
    dash_pattern = (0, (10, 5))

    # Load base data
    dfA, time_full, locals_mat, comm_mat, totals_mat = _load_pair_baseline()

    diff = pd.DataFrame({
        "Time": dfA["Time"],
        "S": dfA["S_Flow"]-dfA["S_Explicit"],
        "E": dfA["E_Flow"]-dfA["E_Explicit"],
        "I": dfA["I_Flow"]-dfA["I_Explicit"],
        "R": dfA["R_Flow"]-dfA["R_Explicit"],
    })
    eps = 1e-12
    relS = np.abs(diff["S"].to_numpy())/np.maximum(eps,
                                                   np.abs(dfA["S_Explicit"].to_numpy()))
    relE = np.abs(diff["E"].to_numpy())/np.maximum(eps,
                                                   np.abs(dfA["E_Explicit"].to_numpy()))
    relI = np.abs(diff["I"].to_numpy())/np.maximum(eps,
                                                   np.abs(dfA["I_Explicit"].to_numpy()))
    relR = np.abs(diff["R"].to_numpy())/np.maximum(eps,
                                                   np.abs(dfA["R_Explicit"].to_numpy()))
    xi = comm_mat / np.maximum(1e-15, totals_mat)

    metrics = compute_convergence_metrics(dt_list)
    metrics_exact = compute_convergence_metrics_exact(dt_list)
    try:
        dfC_exact = _load_exact_baseline()
    except FileNotFoundError as e:
        print(f"[Warning] FLOW_EXACT Baseline missing: {e}")
        dfC_exact = None

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    # Physical layout: (A, B, C) / (D, E, F)
    (ax_pos_A, ax_pos_B, ax_pos_C), (ax_pos_D, ax_pos_E, ax_pos_F) = axs

    # Assign content to panels:
    # A: Trajectories
    # B: Convergence - Relative Error (stage-algined)
    # C: Convergence - Relative Error (Phi)
    # D: RK4 Error different h
    # E: Shares
    # F: RK4 Error same h
    axA = ax_pos_A  # Trajectories
    axB_conv_rel = ax_pos_B  # Convergence relative
    axC = ax_pos_C  # Euler-Euler Error
    axD = ax_pos_D  # RK4 error different h
    axE = ax_pos_E  # Shares
    axF = ax_pos_F  # RK4 error same h

    # Load convergence data for all methods
    all_conv_data_stage_algined = load_convergence_data_all_methods(
        DTS_CONVERGENCE)
    all_conv_data_phi = load_convergence_data_all_methods(
        DTS_CONVERGENCE, phi=True)

    # (A) Trajectories
    for comp in ['S', 'E', 'I', 'R']:
        # Flow (solid), Explicit (dashed)
        axA.plot(dfA["Time"], dfA[f"{comp}_Flow"],
                 color=color_map[comp], lw=lw_main, alpha=0.90, zorder=2)
        axA.plot(dfA["Time"], dfA[f"{comp}_Explicit"],
                 color=color_map[comp], lw=lw_dash, ls=dash_pattern, alpha=0.95, zorder=3)
    axA.set_xlabel("Time (days)")
    axA.set_ylabel("Population")
    axA.set_title("Population trajectories", fontsize=16, pad=4)

    # (B) Convergence - Relative Error (all methods)
    if all_conv_data_stage_algined:
        dt_range = np.array([min(DTS_CONVERGENCE), max(DTS_CONVERGENCE)])
        for method_name, method_info in METHODS.items():
            if method_name not in all_conv_data_stage_algined:
                continue
            df = all_conv_data_stage_algined[method_name]
            dts = df['dt'].values
            errors = df['max_rel_error'].values

            axB_conv_rel.loglog(dts, errors,
                                marker=method_info['marker'],
                                color=method_info['color'],
                                linewidth=2.5,
                                markersize=8,
                                label=method_info['label'],
                                zorder=3)

            # Reference line
            if len(df) > 0:
                dt_anchor = df['dt'].iloc[0]
                err_anchor = df['max_rel_error'].iloc[0]
                order = method_info['order']
                C = err_anchor / (dt_anchor ** order)
                ref_line = C * (dt_range ** order)
                axB_conv_rel.loglog(dt_range, ref_line, ':',
                                    color=method_info['color'],
                                    linewidth=2,
                                    alpha=0.5,
                                    zorder=1)

            last_dt = df['dt'].iloc[-1]
            last_err = df['max_rel_error'].iloc[-1]
            axB_conv_rel.text(last_dt * 1.15, last_err * 15.5, method_info['label'],
                              color=method_info['color'], fontsize=13,
                              fontweight='bold', verticalalignment='center')

    axB_conv_rel.set_xlabel(r'Step size $h$ (days)')
    axB_conv_rel.set_ylabel('Max. rel. error')
    axB_conv_rel.set_xscale('log', base=2)
    axB_conv_rel.set_xticks(DTS_CONVERGENCE)
    axB_conv_rel.get_xaxis().set_major_formatter(FormatStrFormatter('%.3g'))
    axB_conv_rel.grid(True, which='both', linestyle=':', alpha=0.55)
    axB_conv_rel.set_title("Convergence: Stage-aligned", fontsize=16, pad=4)

    # (C) Convergence - Relative Error (Phi)
    if all_conv_data_phi:
        dt_range = np.array([min(DTS_CONVERGENCE), max(DTS_CONVERGENCE)])
        for method_name, method_info in METHODS.items():
            if method_name not in all_conv_data_phi:
                continue
            df = all_conv_data_phi[method_name]
            dts = df['dt'].values
            errors = df['max_rel_error'].values

            axC.loglog(dts, errors,
                       marker=method_info['marker'],
                       color=method_info['color'],
                       linewidth=2.5,
                       markersize=8,
                       label=method_info['label'],
                       zorder=3)

            # Reference line
            if len(df) > 0:
                dt_anchor = df['dt'].iloc[0]
                err_anchor = df['max_rel_error'].iloc[0]
                order = method_info['order']
                C = err_anchor / (dt_anchor ** order)
                ref_line = C * (dt_range ** order)
                axC.loglog(dt_range, ref_line, ':',
                           color=method_info['color'],
                           linewidth=2,
                           alpha=0.5,
                           zorder=1)

            # Add text label near the last data point (shifted up)
            last_dt = df['dt'].iloc[-1]
            last_err = df['max_rel_error'].iloc[-1]
            axC.text(last_dt * 1.15, last_err * 15.5, method_info['label'],
                     color=method_info['color'], fontsize=13,
                     fontweight='bold', verticalalignment='center')

    axC.set_xlabel(r'Step size $h$ (days)')
    axC.set_ylabel('Max. rel. error')
    axC.set_xscale('log', base=2)
    axC.set_xticks(DTS_CONVERGENCE)
    axC.get_xaxis().set_major_formatter(FormatStrFormatter('%.3g'))
    axC.grid(True, which='both', linestyle=':', alpha=0.55)
    axC.set_title("Convergence: Fundamental matrix", fontsize=16, pad=4)

    # (D) RK4 error different h (FlowExact vs Explicit)
    if dfC_exact is not None:
        for comp in ['S', 'E', 'I', 'R']:
            y = np.abs(
                dfC_exact[f"{comp}_FlowExact"].to_numpy(
                ) - dfC_exact[f"{comp}_Explicit"].to_numpy()
            )
            y[y <= 0] = 1e-16
            axD.plot(dfC_exact["Time"], y, color=color_map[comp], lw=lw_main)
        axD.set_yscale("log")
        axD.set_xlabel("Time (days)")
        axD.set_ylabel("Abs. error")
    else:
        axD.text(0.5, 0.5, "No FLOW_EXACT data", ha="center",
                 va="center", transform=axD.transAxes)
    axD.set_title("Exact equivalence (RK-4)", fontsize=16, pad=4)

    # (E) Shares
    for i, lab in enumerate(['S', 'E', 'I', 'R']):
        axE.plot(time_full, xi[:, i], color=color_map[lab], lw=lw_main)
    axE.set_xlabel("Time (days)")
    axE.set_ylabel(r"Commuter share $\xi_j(t)$")
    axE.set_title("Evolution of shares", fontsize=16, pad=4)

    # (F) RK4 error over time
    for comp in ['S', 'E', 'I', 'R']:
        y = np.abs(diff[comp].to_numpy())
        y[y <= 0] = 1e-16
        axF.plot(diff["Time"], y, color=color_map[comp], lw=lw_main)
    axF.set_yscale("log")
    axF.set_xlabel("Time (days)")
    axF.set_ylabel("Abs. error")
    axF.set_title("Error of hybrid update", fontsize=16, pad=4)

    # Panel labels
    for ax, label in zip([ax_pos_A, ax_pos_B, ax_pos_C, ax_pos_D, ax_pos_E, ax_pos_F],
                         list("ABCDEF")):
        ax.text(-0.06, 1.03, f"{label}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=23, fontweight="bold", clip_on=False)

    # Global legend for compartments (S, E, I, R)
    comp_handles = [Line2D([0], [0], color=color_map[k],
                           lw=lw_main, label=k) for k in ["S", "E", "I", "R"]]

    plt.subplots_adjust(bottom=0.18, top=0.92, wspace=0.25, hspace=0.42)

    fig.legend(handles=comp_handles,
               loc="lower center", bbox_to_anchor=(0.5, 0.05),
               ncol=4, frameon=True, columnspacing=1.2, handlelength=2.2)

    # Save figure
    out_pdf = os.path.join(PLOT_DIR, f"{outname}.pdf")
    out_png = os.path.join(PLOT_DIR, f"{outname}.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    print("[OK] Multipanel gespeichert:", out_pdf, "und", out_png)


if __name__ == "__main__":
    set_settings()
    build_multipanel_figure(DT_LIST, outname="Figure_4")
