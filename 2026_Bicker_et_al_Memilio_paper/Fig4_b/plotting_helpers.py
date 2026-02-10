#############################################################################
# Copyright (C) 2020-2026 MEmilio
#
# Authors: Vincent Wieland
#
# Contact: Martin J. Kuehn <Martin.Kuehn@DLR.de>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#############################################################################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def set_fontsize(base_fontsize=17):
    fontsize = base_fontsize
    plt.rcParams.update({
        'font.size': fontsize,
        'axes.titlesize': fontsize * 1,
        'axes.labelsize': fontsize,
        'xtick.labelsize': fontsize * 0.8,
        'ytick.labelsize': fontsize * 0.8,
        'legend.fontsize': fontsize * 0.8,
        'font.family': "Arial"
    })


plt.style.use('default')

dpi = 300

colors = {"Blue": "#155489",
          "Medium blue": "#64A7DD",
          "Light blue": "#B4DCF6",
          "Lilac blue": "#AECCFF",
          "Turquoise": "#76DCEC",
          "Light green": "#B6E6B1",
          "Medium green": "#54B48C",
          "Green": "#5D8A2B",
          "Teal": "#20A398",
          "Yellow": "#FBD263",
          "Orange": "#E89A63",
          "Rose": "#CF7768",
          "Red": "#A34427",
          "Purple": "#741194",
          "Grey": "#C0BFBF",
          "Dark grey": "#616060",
          "Light grey": "#F1F1F1"}

if __name__ == '__main__':
    set_fontsize()


def posterior_predictive_plot(
    simfun,
    chain_array,
    obs_t,
    obs_y,
    *,
    n_ens=500,
    burn_in=0,
    quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
    state=0,
    sum_states=None,
    rng=None,
    # plotting customization:
    interval_colors=None,
    interval_alphas=None,
    median_color="black",
    median_lw=2.0,
    data_color="black",
    data_ms=16,
    ax=None,
    **plot_kwargs,
):
    """
    Parameters
    ----------
    simfun : callable
        simfun(theta_vec, obs_t) -> array shaped (states, T) or (T, states)
    chain_array : np.ndarray
        Posterior draws array with shape (n_iter, n_param_plus_lp, n_chain).
        Column 0 is assumed to be lp (log posterior) and is dropped like in Julia.
    obs_t, obs_y : array-like
        Observed times and values, same length.
    state : int or None
        Which state index to plot (0-based). Ignored if sum_states is provided.
    sum_states : list[int] or None
        If provided, sum these state indices (0-based).
    """

    obs_t = np.asarray(obs_t)
    obs_y = np.asarray(obs_y)
    if obs_t.shape[0] != obs_y.shape[0]:
        raise ValueError("obs_t and obs_y must have the same length")
    if (state is None) and (sum_states is None):
        raise ValueError("Provide either `state` or `sum_states`")

    if rng is None:
        rng = np.random.default_rng()

    # Drop lp column (assumed first column), like arr = chain.value.data[:,2:end,:] in Julia
    arr = chain_array[:, 1:, :]
    n_iter, n_param, n_chain = arr.shape

    if not (0 <= burn_in < n_iter):
        raise ValueError("burn_in must be between 0 and n_iter-1")
    first_it = burn_in
    n_it_kept = n_iter - burn_in
    n_kept = n_it_kept * n_chain

    n_draw = min(n_ens, n_kept)

    # Sample k in [0, n_kept) uniformly without replacement
    inds = rng.choice(n_kept, size=n_draw, replace=False)

    def idx_to_it_ch(k):
        it = (k % n_it_kept) + first_it
        ch = (k // n_it_kept)
        return it, ch

    T = obs_t.shape[0]
    ensemble = np.empty((n_draw, T), dtype=float)

    sum_states = None if sum_states is None else list(sum_states)

    for e, k in enumerate(inds):
        it, ch = idx_to_it_ch(k)
        theta_vec = arr[it, :, ch]  # shape (n_param,)

        sim = np.asarray(simfun(theta_vec, obs_t))

        # Interpret sim as (states, T). Accept (states, T) or (T, states).
        if sim.ndim != 2:
            raise ValueError(
                "Simulator output must be 2D (states×T) or (T×states)")
        if sim.shape[1] == T:
            sim_st_t = sim
        elif sim.shape[0] == T:
            sim_st_t = sim.T
        else:
            raise ValueError(
                "Simulator output must be (states×T) or (T×states) with T=len(obs_t)")

        if sum_states is not None:
            vals = sim_st_t[sum_states, :].sum(axis=0)
        else:
            vals = sim_st_t[state, :]

        ensemble[e, :] = vals

    # ---- Quantiles over ensemble at each time point ----
    q = np.sort(np.asarray(quantiles, dtype=float))
    print(q, q.shape)
    qtraj = np.quantile(ensemble, q, axis=0)  # shape (nq, T)

    # ---- Plotting ----
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.), dpi=300)
    else:
        fig = ax.figure

    ax.set_ylabel("Infected [# per 100k]")
    ax.grid(False)

    nq = q.shape[0]
    # Find median position (if present)
    median_pos = np.where(np.isclose(q, 0.5, atol=1e-8))[0]
    median_pos = int(median_pos[0]) if median_pos.size else None

    # Shaded bands for symmetric pairs: (q[0], q[-1]), (q[1], q[-2]), ...
    n_pairs = nq // 2

    # Defaults: colors and alphas per band (outer -> inner)
    if interval_colors is None:
        # one color reused; you can pass a list for per-band colors
        interval_colors = ["C0"] * n_pairs
    if interval_alphas is None:
        # lighter outside, darker inside (feel free to tweak)
        # create n_pairs values between 0.15 and 0.35
        interval_alphas = np.linspace(0.15, 0.35, n_pairs).tolist()

    # If user provided a single color/alpha, broadcast it
    if isinstance(interval_colors, str):
        interval_colors = [interval_colors] * n_pairs
    if isinstance(interval_alphas, (float, int)):
        interval_alphas = [float(interval_alphas)] * n_pairs

    for i in range(n_pairs):
        low = qtraj[i, :]
        high = qtraj[nq - 1 - i, :]

        label = f"q={q[i]:g}, {q[nq-1-i]:g}"
        ax.fill_between(
            obs_t,
            low,
            high,
            color=interval_colors[i] if i < len(interval_colors) else "C0",
            alpha=interval_alphas[i] if i < len(interval_alphas) else 0.2,
            # label=label,
            linewidth=0,
        )

    # Median line
    if median_pos is not None:
        ax.plot(obs_t, qtraj[median_pos, :],
                color=median_color, lw=median_lw, label="Simulation")

    # Observed data (x-markers)
    ax.plot(
        obs_t,
        obs_y,
        linestyle="none",
        marker="x",
        color=data_color,
        # markersize=6,
        markeredgewidth=1.5,
        label="Reported data",
    )
    ax.set_ylim((0, 63000))

    # Define start date
    start_date = pd.to_datetime("2016-08-07")
    ticks = ax.get_xticks()[1:-1]
    # add one at tobs[-1]
    ticks = np.append(ticks, obs_t[-1])
    ticklabels = start_date + pd.to_timedelta(ticks, unit="D")
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels.strftime(
        "%Y-%m-%d"), rotation=35, ha="right")


    return {
        "fig": fig,
        "ax": ax,
        "times": obs_t,
        "quantiles": q,
        "qtraj": qtraj,
        "ensemble": ensemble,
    }
