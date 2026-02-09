#!/usr/bin/env python3
"""
Create a the panels for Figure 6a of Bicker et al. Memilio paper:
    - runtime_seir_metapop.png / .pdf
    - speedup_total_seir_metapop.png / .pdf
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import importlib.util as _importlib_util


def _load_plotting_settings():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # plotting_settings.py is in the parent folder
    settings_path = os.path.join(script_dir, "..", "plotting_settings.py")
    try:
        spec = _importlib_util.spec_from_file_location(
            "plotting_settings", settings_path)
        if spec and spec.loader:
            mod = _importlib_util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            _set = getattr(mod, "set_fontsize", lambda *a, **k: None)
            _dpi = getattr(mod, "dpi", 300)
            _colors = getattr(mod, "colors", {})
            return _set, _dpi, _colors
    except Exception:
        pass
    # Fallbacks
    return (lambda *a, **k: None), 150, {}


_set_fontsize, _dpi, _colors = _load_plotting_settings()

PY_BINDINGS_EULER_LABEL = "Python Euler (C++)"
PY_BINDINGS_RK4_LABEL = "Python RK4 (C++)"


def _import_or_exit():
    try:
        import pandas as pd  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        plt.style.use('default')
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['figure.facecolor'] = 'white'
        return pd, plt
    except Exception as e:
        msg = (
            "Fehlende Python-Abhängigkeiten. Bitte installieren: pandas, matplotlib\n"
            "Beispiel:\n"
            "  python3 -m pip install --user pandas matplotlib\n\n"
            f"Originalfehler: {e}\n")
        print(msg, file=sys.stderr)
        sys.exit(1)


def load_benchmarks(base_dir: str, pd):
    """Load four benchmark files (C++/R x Euler/RK4). Falls back to legacy names if necessary.

    Returns:
      runtimes_long: DataFrame with columns [Regions, Impl, Runtime]
      speedups_total: DataFrame with columns [Regions, Integrator, SpeedupTotal]
    """
    def read_csv_safe(path):
        return pd.read_csv(path) if os.path.exists(path) else None

    paths = {
        "cpp_euler": os.path.join(base_dir, "cpp_benchmark_euler.csv"),
        "r_euler": os.path.join(base_dir, "r_benchmark_euler.csv"),
        "cpp_rk4": os.path.join(base_dir, "cpp_benchmark_rk4.csv"),
        "r_rk4": os.path.join(base_dir, "r_benchmark_rk4.csv"),
        "py_euler": os.path.join(base_dir, "python_benchmark_euler.csv"),
        "py_rk4": os.path.join(base_dir, "python_benchmark_rk4.csv"),
    }
    dfs = {k: read_csv_safe(p) for k, p in paths.items()}

    # Legacy fallback
    if dfs["cpp_euler"] is None or dfs["r_euler"] is None:
        legacy_cpp = os.path.join(base_dir, "cpp_benchmark.csv")
        legacy_r = os.path.join(base_dir, "r_benchmark.csv")
        if os.path.exists(legacy_cpp) and os.path.exists(legacy_r):
            legacy_cpp_df = pd.read_csv(legacy_cpp)
            legacy_r_df = pd.read_csv(legacy_r)
            dfs["cpp_euler"] = legacy_cpp_df.copy()
            dfs["r_euler"] = legacy_r_df.copy()
        else:
            print(
                "Benchmark-CSV-Dateien nicht gefunden. Bitte zuerst Benchmarks ausführen.\n"
                f"Erwartet: {paths['cpp_euler']}, {paths['r_euler']} (und optional RK4) oder Fallback {legacy_cpp}, {legacy_r}",
                file=sys.stderr,)
            sys.exit(1)

    # Build runtime long DataFrame for all available combos
    frames = []
    label_map = {
        "cpp_euler": "C++ Euler",
        "r_euler": "R Euler (pure R)",
        "cpp_rk4": "C++ RK4",
        "r_rk4": "R RK4 (deSolve, C)",
        "py_euler": PY_BINDINGS_EULER_LABEL,
        "py_rk4": PY_BINDINGS_RK4_LABEL,
    }
    for key, label in label_map.items():
        df = dfs.get(key)
        if df is not None:
            frames.append(pd.DataFrame({
                "Regions": df["Regions"],
                "Impl": label,
                "Runtime": df["RuntimeSeconds"],
            }))
    runtimes_long = pd.concat(frames, ignore_index=True)

    # Compute total speedup per integrator where both C++ and R exist
    speedups = []
    if (dfs["cpp_euler"] is not None) and (dfs["r_euler"] is not None):
        m = pd.merge(
            dfs["cpp_euler"],
            dfs["r_euler"],
            on="Regions", suffixes=("_cpp", "_r"))
        if "TotalNoIOSeconds_cpp" in m.columns and "TotalNoIOSeconds_r" in m.columns:
            sp = m["TotalNoIOSeconds_r"] / m["TotalNoIOSeconds_cpp"]
        else:
            sp = m["RuntimeSeconds_r"] / m["RuntimeSeconds_cpp"]
        speedups.append(
            pd.DataFrame(
                {"Regions": m["Regions"],
                 "Integrator": "Euler", "SpeedupTotal": sp}))
    if (dfs["cpp_rk4"] is not None) and (dfs["r_rk4"] is not None):
        m = pd.merge(
            dfs["cpp_rk4"],
            dfs["r_rk4"],
            on="Regions", suffixes=("_cpp", "_r"))
        if "TotalNoIOSeconds_cpp" in m.columns and "TotalNoIOSeconds_r" in m.columns:
            sp = m["TotalNoIOSeconds_r"] / m["TotalNoIOSeconds_cpp"]
        else:
            sp = m["RuntimeSeconds_r"] / m["RuntimeSeconds_cpp"]
        speedups.append(
            pd.DataFrame(
                {"Regions": m["Regions"],
                 "Integrator": "RK4", "SpeedupTotal": sp}))

    speedups_total = pd.concat(
        speedups, ignore_index=True) if speedups else pd.DataFrame(
        {"Regions": [],
         "Integrator": [],
         "SpeedupTotal": []})

    if runtimes_long.empty:
        print("Keine Benchmarkdaten gefunden.", file=sys.stderr)
        sys.exit(1)
    return runtimes_long, speedups_total


def load_results_for_accuracy(base_dir: str, pd):
    """Kept for backward compatibility; accuracy panels are not used in this layout."""
    return None, None, None


def _annotate_cpp_r_speedups(ax, runtimes_long, fontsize=10):
    """Draw connector lines between C++ Euler and R Euler runtimes and label the speedup."""
    df_cpp = runtimes_long[runtimes_long["Impl"] == "C++ Euler"][
        ["Regions", "Runtime"]
    ].dropna()
    df_r = runtimes_long[runtimes_long["Impl"] == "R Euler (pure R)"][
        ["Regions", "Runtime"]
    ].dropna()

    if df_cpp.empty or df_r.empty:
        return

    df_cpp = (
        df_cpp.sort_values(["Regions", "Runtime"])
        .groupby("Regions", as_index=False)
        .first()
    )
    df_r = (
        df_r.sort_values(["Regions", "Runtime"])
        .groupby("Regions", as_index=False)
        .first()
    )

    merged = df_cpp.merge(df_r, on="Regions", suffixes=("_cpp", "_r"))
    if merged.empty:
        return

    connector_color = _colors.get("DarkGrey", "#4B4B4B")
    fontsize = max(int(fontsize * 0.85), 6)
    try:
        x_limits = ax.get_xlim()
        is_log_x = ax.get_xscale() == "log"
    except Exception:
        x_limits = None
        is_log_x = False
    try:
        import matplotlib.patheffects as mpatheffects  # type: ignore
    except Exception:
        mpatheffects = None

    def _shift_label_x(x_value):
        if x_limits is None:
            return x_value
        xmin, xmax = x_limits
        if not (np.isfinite(xmin) and np.isfinite(xmax)):
            return x_value
        if is_log_x and x_value > 0:
            candidate = x_value * 1.25
            max_allowed = xmax / 1.01 if xmax > 0 else xmax
            if x_value >= max_allowed:
                return x_value
            return candidate if candidate < max_allowed else max_allowed
        span = xmax - xmin
        if span <= 0:
            return x_value
        candidate = x_value + 0.07 * span
        max_allowed = xmax - 0.01 * span
        if x_value >= max_allowed:
            return x_value
        return candidate if candidate < max_allowed else max_allowed

    for _, row in merged.iterrows():
        y_cpp = row["Runtime_cpp"]
        y_r = row["Runtime_r"]
        x = row["Regions"]

        if not (np.isfinite(y_cpp) and np.isfinite(y_r) and y_cpp > 0 and y_r > 0):
            continue

        lower, upper = (y_cpp, y_r) if y_cpp < y_r else (y_r, y_cpp)
        # Erstelle mehrere Punkte entlang der vertikalen Linie für Dreiecks-Marker
        n_points = 12  # Anzahl der Dreiecke entlang der Linie
        y_points = np.logspace(np.log10(lower), np.log10(upper), n_points)
        x_points = np.full(n_points, x)

        ax.plot(
            x_points,
            y_points,
            color=connector_color,
            linewidth=0,  # keine durchgehende Linie
            linestyle='None',
            marker='^',
            markersize=4,
            markerfacecolor=connector_color,
            markeredgecolor=connector_color,
            alpha=0.45,
            zorder=1.5,
        )

        speedup = y_r / y_cpp if y_cpp else np.nan
        if not np.isfinite(speedup):
            continue

        y_mid = np.sqrt(y_cpp * y_r)
        shift_factor = 2.05
        offset_upper = upper * 0.995
        offset_lower = lower * 1.005
        if y_cpp < y_r:
            y_pos = min(y_mid * shift_factor, offset_upper)
        else:
            y_pos = max(y_mid / shift_factor, offset_lower)
        label = f"{speedup:.0f}×"
        text_obj = ax.text(
            _shift_label_x(x),
            y_pos,
            label,
            ha="center",
            va="center",
            color=connector_color,
            fontsize=fontsize,
            zorder=4,
            clip_on=False,
        )
        if mpatheffects is not None:
            text_obj.set_path_effects(
                [mpatheffects.withStroke(
                    linewidth=1.1, foreground="white")])





def plot_runtime(*args, **kwargs):
    base_dir, runtimes_long, plt = args[0], args[1], args[2]
    _set_fontsize()
    base_tick = int(0.8 * 17)
    legend_size = int(0.8 * 17)

    # Verwende explizit figure + axes mit add_axes und gewünschter Panel-Geometrie
    figsize = (8, 5)
    panel = (0.2, 0.2, 0.78, 0.75)
    fig = plt.figure(figsize=figsize, dpi=_dpi)
    ax = fig.add_axes(panel)

    try:
        ax.tick_params(axis='both', which='both', labelsize=base_tick)
    except Exception:
        pass

    # Farben und Linienstile konsistent mit Mehrpanel-Plot
    cpp_color = _colors.get("Blue", "#155489")
    r_color = _colors.get("Orange", "#E89A63")
    py_color = _colors.get("Green", "#1B8A5A")
    color_cycle = {
        "C++ Euler": cpp_color,
        "C++ RK4": cpp_color,
        "R Euler (pure R)": r_color,
        "R RK4 (deSolve, C)": r_color,
        PY_BINDINGS_EULER_LABEL: py_color,
        PY_BINDINGS_RK4_LABEL: py_color,
    }
    linestyle_map = {
        "C++ Euler": "-",
        "C++ RK4": "--",
        "R Euler (pure R)": "-",
        "R RK4 (deSolve, C)": "--",
        PY_BINDINGS_EULER_LABEL: "-",
        PY_BINDINGS_RK4_LABEL: "--",
    }

    lines_by_impl = {}
    for impl, df_sub in runtimes_long.groupby("Impl"):
        df_sub = df_sub.sort_values("Regions")
        line_obj, = ax.plot(
            df_sub["Regions"], df_sub["Runtime"],
            marker="o", label=impl, color=color_cycle.get(impl),
            linestyle=linestyle_map.get(impl, "-"))
        lines_by_impl[impl] = line_obj

    ax.set_yscale("log")
    ax.set_xlabel("Regions [#]")
    ax.set_xscale("log")
    ax.set_ylabel("Runtime [s]")
    preferred_order = [
        "C++ Euler", "C++ RK4", "R Euler (pure R)", "R RK4 (deSolve, C)",
        PY_BINDINGS_EULER_LABEL, PY_BINDINGS_RK4_LABEL]
    handles = [lines_by_impl[k] for k in preferred_order if k in lines_by_impl]
    labels = [k for k in preferred_order if k in lines_by_impl]
    ax.legend(handles=handles, labels=labels,
              loc="lower right", fontsize=legend_size-1,
              framealpha=0.95, edgecolor='gray', fancybox=False)
    ax.grid(True, alpha=0.3)
    _annotate_cpp_r_speedups(ax, runtimes_long, fontsize=base_tick - 1)

    # tight_layout nicht verwenden, da add_axes verwendet wird und Layout explizit ist
    out_png = os.path.join(base_dir, "runtime_seir_metapop.png")
    out_pdf = os.path.join(base_dir, "runtime_seir_metapop.pdf")
    fig.savefig(out_png, dpi=_dpi)
    fig.savefig(out_pdf, dpi=_dpi)
    return out_png, out_pdf


def plot_speedup(*args, **kwargs):
    raise NotImplementedError


def plot_total_runtime(*args, **kwargs):
    raise NotImplementedError


def plot_speedup_total(*args, **kwargs):
    base_dir, speedups_total, plt = args[0], args[1], args[2]
    _set_fontsize()
    base_tick = int(0.8 * 17)
    legend_size = int(0.8 * 17)

    # Verwende explizit figure + axes mit add_axes und gewünschter Panel-Geometrie
    figsize = (8, 5)
    panel = (0.2, 0.2, 0.78, 0.75)
    fig = plt.figure(figsize=figsize, dpi=_dpi)
    ax = fig.add_axes(panel)

    try:
        ax.tick_params(axis='both', which='both', labelsize=base_tick)
    except Exception:
        pass

    if not hasattr(speedups_total, 'empty') or speedups_total.empty:
        ax.axis('off')
        ax.text(0.5, 0.5, "No total-no-IO timing available",
                ha='center', va='center')
    else:
        integrator_colors = {
            "Euler": _colors.get("Teal", "#20A398"),
            "RK4": _colors.get("Purple", "#741194"),
        }
        marker_map = {"Euler": "o", "RK4": "s"}
        for integrator, df_sub in speedups_total.groupby("Integrator"):
            df_sub = df_sub.sort_values("Regions")
            ax.plot(
                df_sub["Regions"], df_sub["SpeedupTotal"],
                marker=marker_map.get(integrator, "o"),
                label=f"{integrator}",
                color=integrator_colors.get(integrator))
        ax.set_yscale("log")
        # Mehr Ticks auf der Log-Skala: Minor-Ticks bei 2..9 jeder Dekade
        try:
            from matplotlib.ticker import LogLocator, NullFormatter
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
            ax.yaxis.set_minor_locator(
                LogLocator(
                    base=10,
                    subs=tuple(
                        np.arange(2, 10) * 0.1)))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis='y', which='minor', length=3.5, width=0.8)
        except Exception:
            pass
        ax.set_xlabel("Regions [#]")
        ax.set_ylabel("Speedup [R/C++]")
        ax.legend(loc="upper right", fontsize=legend_size)
        ax.grid(True, alpha=0.3)

    # tight_layout nicht verwenden, da add_axes verwendet wird und Layout explizit ist
    out_png = os.path.join(base_dir, "speedup_total_seir_metapop.png")
    out_pdf = os.path.join(base_dir, "speedup_total_seir_metapop.pdf")
    fig.savefig(out_png, dpi=_dpi)
    fig.savefig(out_pdf, dpi=_dpi)
    return out_png, out_pdf


def main():
    parser = argparse.ArgumentParser(
        description="Plot speedup for SEIR metapop benchmarks (C++ vs R)")
    parser.add_argument(
        "--base-dir",
        default=os.getcwd(),
        help="Directory containing cpp_benchmark.csv and r_benchmark.csv",
    )
    args = parser.parse_args()

    pd, plt = _import_or_exit()
    runtimes_long, speedups_total = load_benchmarks(args.base_dir, pd)
    _, _, _ = load_results_for_accuracy(args.base_dir, pd)
    rt_png, rt_pdf = plot_runtime(args.base_dir, runtimes_long, plt)
    sp_png, sp_pdf = plot_speedup_total(args.base_dir, speedups_total, plt)
    print("Plots written to:\n  {}\n  {}\n  {}\n  {}".format(
        rt_png, rt_pdf, sp_png, sp_pdf))


if __name__ == "__main__":
    main()
