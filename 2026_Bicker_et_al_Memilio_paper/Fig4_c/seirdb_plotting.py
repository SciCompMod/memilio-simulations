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

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import norm

import plotting_helpers
import os

# import importlib
# importlib.reload("plotting")

plt.style.use('default')

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def set_fontsize(base_fontsize=17):
    fontsize = base_fontsize
    plt.rcParams.update({
        'font.size': fontsize,
        'axes.titlesize': fontsize,
        'axes.labelsize': fontsize,
        'xtick.labelsize': fontsize * 0.8,
        'ytick.labelsize': fontsize * 0.8,
        'legend.fontsize': fontsize * 0.8,
        'font.family': "Arial",
    })

set_fontsize(17)


nsamples = 50000
# ebola_samples = np.load(f"output/converged_samples_{nsamples}_out.npy")
ebola_samples = np.load(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), f"seirdb_ebola/output/cutted_early_flow_converged_samples_{nsamples}_out.npy"))

from simulation import flow_seirdb as simulation_function

data = pd.read_csv(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "seirdb_ebola/data/cut_early_guinea_conf_cases.csv"))
obs = data["new_cases"].values
tobs = data["Date"].values


testp = [0.0071, 5, 4.0, 0.53, 0.025, 2.0]
N0 = 10000000.0
E0 = 0.0
I0 = 10.0
R0 = 0.0
D0 = 0.0
B0 = 0.0
S0 = N0-(E0+I0+R0+D0+B0)

x_initial = [S0, E0, I0, R0, D0, B0]

t_initial = 0.0
t_end = 487 # 3 years

dispersion = 5.0

def flow_simfun(par, tobs):
    full_sim = simulation_function(par, x_initial, tobs[0], tobs[-1])[1]

    # Find indices in full_sim time grid closest to observation times
    tobs_ind = [
        np.where(np.abs(full_sim[0, :] - t) < 1e-3)[0][0]
        for t in tobs
    ]

    # Extract simulated states at observation times
    flows_obs = full_sim[1:, tobs_ind]


    diffs = []
    for i in range(flows_obs.shape[0]):
        row = flows_obs[i, :]
        d = np.diff(np.concatenate(([0.0], row)))
        diffs.append(d)

    mu = np.column_stack(diffs).T

    # nb_p =  dispersion ./(dispersion .+ mu)
    nb_p = dispersion / (dispersion + mu)
    noisy_sim = stats.nbinom.rvs(dispersion, 
                                 nb_p)
    # remove negative values
    noisy_sim[noisy_sim < 0] = 0
    return noisy_sim

def cum_simfun(par, tobs):
    return np.cumsum(flow_simfun(par, tobs))

plot_samples = ebola_samples[:, :, :]
plot_dict = plotting_helpers.posterior_predictive_plot(flow_simfun,
                          plot_samples,
                          tobs,
                          obs,
                          state=0,
                          median_color = "#E89A63",
                          median_lw=3,
                          interval_colors = "#E89A63",
                          interval_alphas=[0.2, 0.4, 0.6],
                          quantiles=(0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975),
                          labels=["50% CI", "90% CI", "95% CI"])


plot_dict["fig"].savefig(f"ebola_guinea_fit.png", bbox_inches='tight')


