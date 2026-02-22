#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Pfade
DIR = "saves/"
CSV = os.path.join(DIR, "commuter_metrics.csv")
OUTDIR = os.path.join(DIR, "Figures", "Figure3")
os.makedirs(OUTDIR, exist_ok=True)

# Szenarien für Panels A und B
SCENARIO_A = "_stress"  # hohe Transmission
SCENARIO_B = "_cp"      # plötzliche Kontaktänderung
COMP_TO_PLOT = "E"      # Panel A/B zeigen E(t) (Exposed)

# Auswahl für (D)
SELECT_PC_FOR_D = 0.95  # wähle einen p_c für die Konvergenzkurve (D)


def set_style():
    base = {
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
    }
    mpl.rcParams.update(base)


# ------------------------- Helper: Daten A/B ----------------------------------

def _coerce_time_series_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sorgt dafür, dass die Spalten genau ['Time','S','E','I','R'] heißen."""
    df = df.copy()
    if "Time" not in df.columns:
        df.rename(columns={df.columns[0]: "Time"}, inplace=True)
    # Falls weitere Spaltennamen nicht exakt S,E,I,R sind, mappe sie robust:
    cols = ["Time"]
    remaining = [c for c in df.columns if c != "Time"]
    if len(remaining) >= 4:
        # Nimm die ersten vier als S,E,I,R (wie in deinen CSVs)
        sub = remaining[:4]
        tmp = df[["Time"] + sub].copy()
        tmp.columns = ["Time", "S", "E", "I", "R"]
        return tmp
    # Fallback (selten nötig): versuche per Anfangsbuchstaben zu mappen
    name_map = {}
    for c in remaining:
        key = c.strip().upper()[0]
        if key in ("S", "E", "I", "R") and key not in name_map:
            name_map[key] = c
    out = {"Time": df["Time"]}
    for k in ["S", "E", "I", "R"]:
        out[k] = df[name_map[k]] if k in name_map else pd.Series(
            np.nan, index=df.index)
    return pd.DataFrame(out)


def load_timeseries_for_AB(base_dir: str, scenario_suffix: str):
    """
    Lädt Zeitreihen-Dateien eines Szenarios für Panels A/B:
    - Reference (explizit)
    - Euler (auxiliary step)
    - Total (Patch-Gesamtsystem)
    Gibt drei DataFrames mit Spalten ['Time','S','E','I','R'] zurück.
    """
    paths = {
        "ref":   os.path.join(base_dir, f"mobile_ref_solution{scenario_suffix}.csv"),
        "euler": os.path.join(base_dir, f"mobile_euler_solution{scenario_suffix}.csv"),
        "total": os.path.join(base_dir, f"seir_solution{scenario_suffix}.csv"),
    }
    data = {}
    for key, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Fehlende Datei für {key}: {p}")
        df = pd.read_csv(p)
        data[key] = _coerce_time_series_columns(df)
    return data["ref"], data["euler"], data["total"]


def _plot_AB_core(ax, df_ref: pd.DataFrame, df_euler: pd.DataFrame,
                  df_total: pd.DataFrame, comp: str = "E"):
    """
    Zeichnet (Reference, Euler, Total) für ein Kompartiment comp in eine gegebene Achse.
    Füllt außerdem die Fläche zwischen Reference und Total in Gelb.
    """
    # Stile wie in deinen bisherigen Abbildungen
    line_width = 3.0
    styles = {
        "ref":   dict(linestyle="-",  color="black", linewidth=line_width),
        "euler": dict(linestyle="--", color="blue",  linewidth=line_width),
        "total": dict(linestyle="-.", color="red",   linewidth=line_width, alpha=0.8),
    }

    ax.plot(
        df_ref["Time"].to_numpy(),
        df_ref[comp].astype(float).to_numpy(),
        linestyle="-", color="black", linewidth=line_width,
        label="Reference"
    )
    ax.plot(
        df_euler["Time"].to_numpy(),
        df_euler[comp].astype(float).to_numpy(),
        linestyle="--", color="blue", linewidth=line_width,
        label="Euler auxiliary"
    )
    ax.plot(
        df_total["Time"].to_numpy(),
        df_total[comp].astype(float).to_numpy(),
        linestyle="-.", color="red", linewidth=line_width, alpha=0.8,
        label="Patch total"
    )

    # Gelbe Differenzfläche zwischen Reference und Total
    # (nur über gemeinsame Zeitpunkte, sortiert)
    common_t = np.intersect1d(
        df_ref["Time"].to_numpy(), df_total["Time"].to_numpy())
    if common_t.size > 0:
        ref_f = df_ref[df_ref["Time"].isin(common_t)].sort_values("Time")
        tot_f = df_total[df_total["Time"].isin(common_t)].sort_values("Time")
        ax.fill_between(ref_f["Time"].to_numpy(),
                        ref_f[comp].to_numpy(),
                        tot_f[comp].to_numpy(),
                        color="#ffeb8a", alpha=0.35, linewidth=0.0, zorder=1,
                        label="Difference")

    # Legende
    ax.legend(loc="best", frameon=True, fancybox=False, shadow=False)

    # Achsentitel/Labels wie in Fig. 2
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(f"{comp}(t)")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Angemessene Limits
    xmin = min(df_ref["Time"].min(), df_euler["Time"].min(),
               df_total["Time"].min())
    xmax = max(df_ref["Time"].max(), df_euler["Time"].max(),
               df_total["Time"].max())
    ax.set_xlim(xmin, xmax)
    ymin = min(df_ref[comp].min(), df_euler[comp].min(), df_total[comp].min())
    ymax = max(df_ref[comp].max(), df_euler[comp].max(), df_total[comp].max())
    ax.set_ylim(ymin, ymax * 1.01)


# ------------------------- Panels A und B -------------------------------------

def panel_A(ax, scenario_suffix: str = SCENARIO_A, comp: str = COMP_TO_PLOT):
    df_ref, df_euler, df_total = load_timeseries_for_AB(DIR, scenario_suffix)
    _plot_AB_core(ax, df_ref, df_euler, df_total, comp)
    # ax.set_title("(A) Stress scenario")


def panel_B(ax, scenario_suffix: str = SCENARIO_B, comp: str = COMP_TO_PLOT):
    df_ref, df_euler, df_total = load_timeseries_for_AB(DIR, scenario_suffix)
    _plot_AB_core(ax, df_ref, df_euler, df_total, comp)
    # ax.set_title("(B) Contact-pattern change")


# ------------------------- Panels C und D -------------------------------------

def load_metrics(path):
    df = pd.read_csv(path)
    needed = {"p_c", "dt", "n_steps", "viol_frac", "Linf_all"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten in metrics CSV: {sorted(missing)}")
    return df


def panel_C(ax, df, viol_thresh=0.02, do_interpolate=True):
    """
    (C) Feasibility-Frontier: kritische Schrittweite Δt_crit(p_c),
    ab der Verletzungen signifikant werden (viol_frac >= viol_thresh).
    Gelb: Verletzungsbereich (unsafe), Dunkel: feasible.
    """
    df2 = df.sort_values(["p_c", "dt"]).copy()
    pcs = np.sort(df2["p_c"].unique())
    dts = np.sort(df2["dt"].unique())
    xmin, xmax = float(dts.min()), float(dts.max())

    dt_crit = []
    for pc in pcs:
        sub = df2[df2["p_c"].eq(pc)].sort_values("dt")
        d = sub["dt"].to_numpy(float)
        v = sub["viol_frac"].to_numpy(float)

        # Finde ersten Index mit v >= Schwelle
        idx = np.where(v >= viol_thresh)[0]
        if idx.size == 0:
            dt_crit.append(np.nan)  # keine Verletzung im Raster
            continue

        i_hi = idx[0]
        if not do_interpolate or i_hi == 0:
            # kein Intervall oder erstem Punkt: nimm d[i_hi]
            dt_crit.append(d[i_hi])
        else:
            # log-lineare Interpolation zwischen (d[i_lo], v_lo) und (d[i_hi], v_hi)
            i_lo = i_hi - 1
            d_lo, d_hi = d[i_lo], d[i_hi]
            v_lo, v_hi = v[i_lo], v[i_hi]
            # Schutz gegen konstante Werte
            if v_hi == v_lo:
                dt_crit.append(d_hi)
            else:
                # Interpolation in log10(Δt)
                x_lo, x_hi = np.log10(d_lo), np.log10(d_hi)
                w = (viol_thresh - v_lo) / (v_hi - v_lo)
                x_star = x_lo + w * (x_hi - x_lo)
                dt_crit.append(10.0**x_star)

    dt_crit = np.array(dt_crit, dtype=float)
    cutoff = np.where(np.isnan(dt_crit), xmax, dt_crit)

    # Flächen
    ax.fill_betweenx(pcs, xmin, cutoff, facecolor="#2f2554",
                     alpha=0.85, label="feasible")
    ax.fill_betweenx(pcs, cutoff, xmax, facecolor="#ffd24d",
                     alpha=0.75, label="violation")

    # Grenzlinie
    ax.plot(dt_crit, pcs, color="white", lw=2.5, marker="o", ms=3, zorder=3,
            label=r"Critical")

    ax.legend(loc="best", frameon=True, fancybox=False, shadow=False)

    ax.set_xscale("log")
    ax.set_xlabel(r"Step size (days)")
    ax.set_ylabel(r"$p_c$")
    ax.grid(True, linestyle=":", alpha=0.55)


def panel_D(ax, df, pc=0.9):
    sub = df[np.isclose(df["p_c"], pc)]
    if sub.empty:
        raise ValueError(f"Keine Zeilen für p_c={pc} gefunden.")
    sub = sub.groupby("dt", as_index=False)["Linf_all"].max().sort_values("dt")
    x = sub["dt"].values.astype(float)
    y = sub["Linf_all"].values.astype(float)

    ax.loglog(x, y, "o-", lw=3, label=f"$p_c={pc}$")

    # slope-1 Referenz durch den mittleren Punkt
    mid = len(x)//2
    k = y[mid] / x[mid] if x[mid] > 0 else y[-1] / max(x[-1], 1e-12)
    refx = np.array([x.min(), x.max()])
    ax.loglog(refx, k*refx, ":", color="0.45", lw=2, label="Order 1")

    ax.legend(loc="best", frameon=True, fancybox=False, shadow=False)

    ax.set_xlabel(r"Step size (days)")
    ax.set_ylabel("Max absolute error")
    ax.grid(True, linestyle=":", alpha=0.55)


# ------------------------------ Main -----------------------------------------

def main():
    set_style()

    # ---- 2×2 Panel: A, B, C, D ----
    fig_all, axs = plt.subplots(
        2, 2, figsize=(14, 10), constrained_layout=True)
    # Hinweis: der zusätzliche Abstand wird über rcParams (set_style) geregelt
    axA, axB, axC, axD = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    # A/B
    panel_A(axA, SCENARIO_A, COMP_TO_PLOT)
    panel_B(axB, SCENARIO_B, COMP_TO_PLOT)

    # C/D
    df_metrics = load_metrics(CSV)
    panel_C(axC, df_metrics)
    panel_D(axD, df_metrics, pc=SELECT_PC_FOR_D)

    # Panel-Buchstaben (links oben in jeder Achse)
    for ax, label in zip([axA, axB, axC, axD], "ABCD"):
        ax.text(-0.08, 1.04, label, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=22, fontweight="bold", clip_on=False)

    # Globaler Titel (dezent)
    # fig_all.suptitle("Euler auxiliary step: feasibility and convergence",
    #                  y=0.995, fontsize=12)

    base_all = os.path.join(OUTDIR, "Figure_3_ABCD")
    for ext in (".pdf", ".svg", ".png"):
        fig_all.savefig(base_all + ext)

    # ---- Nur C & D (wie bisher) ----
    fig_cd, axes_cd = plt.subplots(
        1, 2, figsize=(12, 4.5), constrained_layout=True)
    # Hinweis: der zusätzliche Abstand wird über rcParams (set_style) geregelt
    panel_C(axes_cd[0], df_metrics)
    panel_D(axes_cd[1], df_metrics, pc=SELECT_PC_FOR_D)
    # fig_cd.suptitle("Euler auxiliary step: feasibility and convergence",
    #                 y=1.02, fontsize=12)
    base_cd = os.path.join(OUTDIR, "Figure_3_CD")
    for ext in (".pdf", ".svg", ".png"):
        fig_cd.savefig(base_cd + ext)

    # ---- Einzelpanels A & B (für Figma) ----
    for letter, func, scen in (("A", panel_A, SCENARIO_A), ("B", panel_B, SCENARIO_B)):
        fig_single, ax_single = plt.subplots(
            1, 1, figsize=(6, 4.5), constrained_layout=True)
        func(ax_single, scen, COMP_TO_PLOT)
        # keine Buchstaben-Textbox hier, damit du Warn-Symbol in Figma frei platzierst
        base_single = os.path.join(OUTDIR, f"Figure_3_{letter}")
        for ext in (".pdf", ".svg", ".png"):
            fig_single.savefig(base_single + ext)

    print("[OK] saved:",
          base_all + ".{pdf,svg,png}",
          base_cd + ".{pdf,svg,png}",
          os.path.join(OUTDIR, "Figure_3_A.{pdf,svg,png}"),
          os.path.join(OUTDIR, "Figure_3_B.{pdf,svg,png}")
          )


if __name__ == "__main__":
    main()
