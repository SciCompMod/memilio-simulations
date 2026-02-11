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


nsamples = 500
influenza_samples = np.load(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), f"sirs_influenza/output/joined_samples_{nsamples}_out.npy"))


from simulation import simple_ssirs as simulation_function


data = pd.read_csv(os.path.join(os.path.dirname(
        os.path.abspath(__file__)),"sirs_influenza/data/ILI_germany_2016_2019.csv"))


testp = [0.1, 7, 16, 0.28235, 0.35, 0.23]
N0 = 100000
I0 = data["Inzidenz_sum"][0] # 3280
R0 = 20000
t_initial = 0.0
# u0 in the ordering [S, I, R]
u_initial = [N0-I0-R0, I0, R0]

obs = data["Inzidenz_sum"]
tobs = 7*data["time"].values
t_end = 1095 # 3 years


def simfun(par, tobs):
    full_sim = simulation_function(par, u_initial, tobs[0], tobs[-1])
    # Find indices in full_sim time grid closest to observation times
    tobs_ind = [
        np.where(np.abs(full_sim[0, :] - t) < 1e-3)[0][0]
        for t in tobs
    ]

    # Extract simulated states at observation times
    obs_sim = full_sim[1:, tobs_ind]

    # Add multiplicative Gaussian noise: Normal(mean, 0.4 * mean)
    noisy_sim = norm.rvs(loc=obs_sim, scale=0.5 * obs_sim)
    # remove negative values
    noisy_sim[noisy_sim < 0] = 0
    return noisy_sim


n_sim = 1000
test_samples = np.array([np.array([[p] for p in testp]) for _ in range(n_sim)])
test_samples = np.concatenate((np.random.normal(size=(1000,1,1)), test_samples), axis=1) # add dummy llh column


plot_samples = influenza_samples[:, [6, 0, 1, 2, 3, 4, 5], :]
plot_dict = plotting_helpers.posterior_predictive_plot(simfun,
                          plot_samples,
                          tobs,
                          obs,
                          state=1,
                          median_color = "#E89A63",
                          median_lw = 3,
                          interval_colors = "#E89A63",
                          interval_alphas=[0.2, 0.4, 0.6],
                          quantiles=(0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975))
# plot_dict["fig"].savefig("influenza_joined_best_fit.pdf", bbox_inches="tight")
plot_dict["fig"].savefig("influenza_joined_best_fit.png", bbox_inches="tight")


