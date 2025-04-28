#############################################################################
# Copyright (C) 2020-2025 MEmilio
#
# Authors: Lena Ploetzke
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
"""@plot_additional.py
Additional plot functions.
"""

import h5py
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize

import memilio.epidata.getDataIntoPandasDataFrame as gd

# Define compartments.
secir_dict = {0: 'Susceptible', 1: 'Exposed', 2: 'Carrier', 3: 'Infected', 4: 'Hospitalized',
              5: 'ICU', 6: 'Recovered', 7: 'Dead'}
age_groups = ['0-4 Years', '5-14 Years', '15-34 Years',
              '35-59 Years', '60-79 Years', '80+ Years']

# Define color to be used while plotting for different models to make plots consistent.
color_dict = {'ODE': 'C0',
              'LCT3': 'C1',
              'LCT10': 'C2',
              'LCT50':  'C3',
              'LCTvar':  'C4',
              }
linestyle_dict = {'ODE': 'solid',
                  'LCT3': 'solid',
                  'LCT10': 'solid',
                  'LCT50':  'solid',
                  'LCTvar':  'dashed',
                  }
fontsize_labels = 14
fontsize_legends = 14
plotfolder = 'Plots/Plots_numerical_experiments_add'


def plot_compartments_age_resolved(file, file_name=""):
    """ Creates a plot of the simulation results for the compartments resolved by age groups. 

    @param[in] file: Path to file (without .h5 extension) containing simulation results to be plotted.
         Results should contain exactly 8 compartments with age resolution (so use accumulated numbers for 
         subcompartments of LCT models). 
    @param[in] file_name: The name of the file where the plot will be saved.
            If an empty string is provided, the plot will not be saved.
    """
    fig, axs = plt.subplots(
        2, 4, sharex='all', num=file_name, tight_layout=True)

    # Load data.
    h5file = h5py.File(str(file) + '.h5', 'r')

    if (len(list(h5file.keys())) > 1):
        raise gd.DataError("File should contain one dataset.")
    if (len(list(h5file[list(h5file.keys())[0]].keys())) > 3):
        raise gd.DataError("Expected only one group.")

    data = h5file[list(h5file.keys())[0]]
    dates = data['Time'][:]
    # As there should be only one Group, total is the simulation result.
    total = data['Total'][:, :]
    if (total.shape[1] != len(age_groups)*len(secir_dict)):
        raise gd.DataError("Expected a different number of compartments.")

    # Plot result.
    left_right = 0
    up_down = 0
    for i in range(len(secir_dict)):
        for age in range(len(age_groups)):
            axs[up_down, left_right].plot(dates,
                                          total[:, age*len(secir_dict) +
                                                i], label=age_groups[age], linewidth=1.2)
        axs[up_down, left_right].set_title(
            secir_dict[i], fontsize=8, pad=3)
        if (left_right < math.ceil(len(range(8))/2)-1):
            left_right += 1
        else:
            left_right = 0
            up_down += 1
    h5file.close()

    # Define some characteristics of the plot.
    for i in range(math.ceil(len(range(8))/2)*2):
        axs[i % 2, int(i/2)].set_xlim(left=0, right=dates[-1])
        axs[i % 2, int(i/2)].set_ylim(bottom=0)
        axs[i % 2, int(i/2)].grid(True, linestyle='--')
        axs[i % 2, int(i/2)].tick_params(axis='y', labelsize=7)
        axs[i % 2, int(i/2)].tick_params(axis='x', labelsize=7)

    fig.supxlabel('Simulation time [days]', fontsize=9)

    lines, labels = axs[0, 0].get_legend_handles_labels()
    lgd = fig.legend(lines, labels, ncol=len(age_groups),  loc='outside lower center',
                     fontsize=8, bbox_to_anchor=(0.5, - 0.065), bbox_transform=fig.transFigure)
    # Size is random such that plot is beautiful.
    fig.set_size_inches(7.5/3*math.ceil(len(range(8))/2), 10.5/2.5)
    fig.tight_layout(pad=0, w_pad=0.3, h_pad=0.4)
    fig.subplots_adjust(bottom=0.09)

    # Save result.
    if file_name:
        fig.savefig(plotfolder+'/'+file_name+'.png',
                    bbox_extra_artists=(lgd,),  bbox_inches='tight', dpi=500)
    plt.close()


def plot_compartments_rel_deviation(unresolved_filename, files, legend_labels, file_name=""):
    """ TODO Creates a plot of the simulation results for the compartments. 
    It has one subplot per compartment in two rows. The plotted compartments can be set with 
    compartment_indices and the number of columns depends on this setting. 
    This function can be used to compare the size of the compartments of results,
     e.g., obtained with different models or different parameter specifications.

    @param[in] files: List of paths to files (without .h5 extension) containing simulation results to be plotted.
         Results should contain exactly 8 compartments (so use accumulated numbers for LCT models). 
    @param[in] legend_labels: List of names for the results to be used for the plot legend.
    @param[in] file_name: The name of the file where the plot will be saved.
            If an empty string is provided, the plot will not be saved.
    @param[in] compartment_indices: The indices of the compartments that should be included in the plot.
    """
    fig, axs = plt.subplots(
        2, 4, sharex='all', num=file_name, tight_layout=True)

    # Load unresolved base data.
    h5file = h5py.File(str(unresolved_filename) + '.h5', 'r')
    data_base = h5file[list(h5file.keys())[0]]
    dates_base = data_base['Time'][:]
    # As there should be only one Group, total is the simulation result.
    total_base = data_base['Total'][:, :]

    # Add simulation results to plot.
    for file in range(len(files)):
        # Load data.
        h5file = h5py.File(str(files[file]) + '.h5', 'r')

        if (len(list(h5file.keys())) > 1):
            raise gd.DataError("File should contain one dataset.")
        if (len(list(h5file[list(h5file.keys())[0]].keys())) > 3):
            raise gd.DataError("Expected only one group.")

        data = h5file[list(h5file.keys())[0]]
        dates = data['Time'][:]
        # As there should be only one Group, total is the simulation result.
        total = data['Total'][:, :]
        if (total.shape[1] != 8):
            raise gd.DataError("Expected a different number of compartments.")
        # Plot result.
        left_right = 0
        up_down = 0
        for i in range(len(secir_dict)):
            axs[up_down, left_right].plot(dates,
                                          (total[:, i]-total_base[:, i])/(total_base[:, i]), label=legend_labels[file], linewidth=1.2)
            axs[up_down, left_right].set_title(
                secir_dict[i], fontsize=8, pad=3)
            if (left_right < 3):
                left_right += 1
            else:
                left_right = 0
                up_down += 1
        h5file.close()

    # Define some characteristics of the plot.
    for i in range(8):
        axs[i % 2, int(i/2)].set_xlim(left=0, right=dates[-1])
        axs[i % 2, int(i/2)].grid(True, linestyle='--')
        axs[i % 2, int(i/2)].tick_params(axis='y', labelsize=7)
        axs[i % 2, int(i/2)].tick_params(axis='x', labelsize=7)

    fig.supxlabel('Simulation time [days]', fontsize=9)

    lines, labels = axs[0, 0].get_legend_handles_labels()
    lgd = fig.legend(lines, labels, ncol=len(legend_labels),  loc='outside lower center',
                     fontsize=8, bbox_to_anchor=(0.5, - 0.065), bbox_transform=fig.transFigure)
    # Size is random such that plot is beautiful.
    fig.set_size_inches(7.5/3*4, 10.5/2.5)
    fig.tight_layout(pad=0, w_pad=0.3, h_pad=0.4)
    fig.subplots_adjust(bottom=0.09)

    # Save result.
    if file_name:
        fig.savefig(plotfolder+'/'+file_name+'.png',
                    bbox_extra_artists=(lgd,),  bbox_inches='tight', dpi=500)
    plt.close()


def plot_rel_deviation(ode_filename, files, legend_labels,  compartment_idx=1, file_name=""):
    """ TODO Creates a plot of the simulation results for one specified compartment. 
    The result should consist of accumulated numbers for subcompartments.
    This function can be used to compare the size of one specific compartment for different simulation results,
    e.g., obtained with different models or different parameter specifications.


    @param[in] files: List of paths to files (without .h5 extension) containing simulation results to be plotted.
         Results should contain exactly 8 compartments (so use accumulated numbers for LCT models). 
    @param[in] legend_labels: List of names for the results to be used for the plot legend.
    @param[in] compartment_idx: The index of the compartment to be plotted.
    @param[in] file_name: The name of the file where the plot will be saved.
            If an empty string is provided, the plot will not be saved.
    """
    plt.figure(file_name)
    # Load ode data.
    h5file = h5py.File(str(ode_filename) + '.h5', 'r')
    data_ode = h5file[list(h5file.keys())[0]]
    dates_ode = data_ode['Time'][:]
    # As there should be only one Group, total is the simulation result.
    total_ode = data_ode['Total'][:, :]
    if compartment_idx == -1:
        result_ode = (
            total_ode[:-1, 0]-total_ode[1:, 0])/(dates_ode[1:]-dates_ode[:-1])
    else:
        result_ode = total_ode[:, compartment_idx]

    # Add simulation results to plot.
    for file in range(len(files)):
        # Load data.
        h5file = h5py.File(str(files[file]) + '.h5', 'r')

        if (len(list(h5file.keys())) > 1):
            raise gd.DataError("File should contain one dataset.")
        if (len(list(h5file[list(h5file.keys())[0]].keys())) > 3):
            raise gd.DataError("Expected only one group.")

        data = h5file[list(h5file.keys())[0]]
        dates = data['Time'][:]
        # As there should be only one Group, total is the simulation result.
        total = data['Total'][:, :]
        if (total.shape[1] != 8):
            raise gd.DataError(
                "Expected a different number of compartments.")

        # Plot result.
        if compartment_idx == -1:
            result = (
                total[:-1, 0]-total[1:, 0])/(dates[1:]-dates[:-1])
        else:
            result = total[:, compartment_idx]

        if legend_labels[file] in color_dict:
            if compartment_idx == -1:
                plt.plot(dates[1:], (result-result_ode)/result_ode, linewidth=1.2,
                         linestyle=linestyle_dict[legend_labels[file]], color=color_dict[legend_labels[file]])
            else:
                plt.plot(dates, (result-result_ode)/result_ode, linewidth=1.2,
                         linestyle=linestyle_dict[legend_labels[file]], color=color_dict[legend_labels[file]])
        else:
            if compartment_idx == -1:
                plt.plot(dates[1:], (result-result_ode) /
                         result_ode, linewidth=1.2)
            else:
                plt.plot(dates, (result-result_ode)/result_ode, linewidth=1.2)

        h5file.close()

    plt.xlabel('Simulation time [days]', fontsize=fontsize_labels)
    plt.yticks(fontsize=9)
    plt.ylabel('Relative deviation from ODE model result',
               fontsize=fontsize_labels)
    plt.xlim(left=0, right=dates[-1])
    plt.legend(legend_labels, fontsize=fontsize_legends, framealpha=0.5)
    plt.grid(True, linestyle='--')
    plt.tight_layout()

    # Save result.
    if file_name:
        plt.savefig(plotfolder+'/'+file_name +
                    '.png', bbox_inches='tight', dpi=500)
    plt.close()


def get_file_name(result_dir, Reff, tReff, num_subcompartments, boolsubcomp=False):
    """ Gives a paths to a file with the simulation results for an LCT model with num_subcompartments subcompartments, 
    where the effective reproduction number is set to Reff at simulation time 2.
    This uses standard defined naming convention of the LCT simulations.

    @param[in] result_dir: Directory pointing to the folder where the simulation result file lies in. 
    @param[in] Reff: Effective reproduction number at simulation time 2 of the simulation result.
    @param[in] num_subcompartments: Number of subcompartments for the LCT model used to obtain the simulation results.
    @param[in] boolsubcomp: Specifies whether the result should contain subcompartments (or accumulated results). 
    """
    filename = "lct_Reff" + f"{Reff:.{1}f}" + "_t" + f"{tReff:.{1}f}" +\
        "_subcomp" + f"{num_subcompartments}"
    if boolsubcomp:
        filename += "_subcompartments"
    return os.path.join(result_dir, filename)


def main():
    if not os.path.isdir(plotfolder):
        os.makedirs(plotfolder)

    # Simulation results should be stored in this folder.
    result_dir = os.path.join(os.path.dirname(
        __file__), "simulation_results", "simulation_lct_numerical_experiments")

    # Define which figures of the paper should be created. Figure 12 is created with another python script.
    figures = [20, 21]

    if 20 in figures:
        folder = os.path.join(result_dir, "dropReff40")
        plot_rel_deviation(get_file_name(folder, 0.5, 2, 1), [get_file_name(folder, 0.5, 2, 3),
                                                              get_file_name(folder, 0.5, 2, 10), get_file_name(folder, 0.5, 2, 50), get_file_name(folder, 0.5, 2, 0)],
                           legend_labels=list(
            ["LCT3", "LCT10", "LCT50", "LCTvar"]), compartment_idx=-1,
            file_name="new_infections_drophalf_reldeviation")
        plot_rel_deviation(get_file_name(folder, 0.5, 2, 1), [get_file_name(folder, 0.5, 2, 3),
                                                              get_file_name(folder, 0.5, 2, 10), get_file_name(folder, 0.5, 2, 50), get_file_name(folder, 0.5, 2, 0)],
                           legend_labels=list(
            ["LCT3", "LCT10", "LCT50", "LCTvar"]), compartment_idx=2,
            file_name="carrier_compartment_drophalf_reldeviation")

        plot_rel_deviation(get_file_name(folder, 0.5, 2, 1), [get_file_name(folder, 0.5, 2, 3),
                                                              get_file_name(folder, 0.5, 2, 10), get_file_name(folder, 0.5, 2, 50), get_file_name(folder, 0.5, 2, 0)],
                           legend_labels=list(
            ["LCT3", "LCT10", "LCT50", "LCTvar"]), compartment_idx=3,
            file_name="carrier_compartment_drophalf_reldeviation")

        folder = os.path.join(result_dir, "riseReffTo2_40")
        plot_rel_deviation(get_file_name(folder, 2, 2, 1), [get_file_name(folder, 2, 2, 3),
                                                            get_file_name(folder, 2, 2, 10), get_file_name(folder, 2,  2, 50), get_file_name(folder, 2,  2, 0)],
                           legend_labels=list(
            ["LCT3", "LCT10", "LCT50", "LCTvar"]), compartment_idx=-1,
            file_name="new_infections_rise2_reldeviation")
        plot_rel_deviation(get_file_name(folder, 2, 2, 1), [get_file_name(folder, 2, 2, 3),
                                                            get_file_name(folder, 2, 2, 10), get_file_name(folder, 2, 2, 50), get_file_name(folder, 2, 2, 0)],
                           legend_labels=list(
            ["LCT3", "LCT10", "LCT50", "LCTvar"]), compartment_idx=2,
            file_name="carrier_compartment_rise2_reldeviation")
        plot_rel_deviation(get_file_name(folder, 2, 2, 1), [get_file_name(folder, 2, 2, 3),
                                                            get_file_name(folder, 2, 2, 10), get_file_name(folder, 2, 2, 50), get_file_name(folder, 2, 2, 0)],
                           legend_labels=list(
            ["LCT3", "LCT10", "LCT50", "LCTvar"]), compartment_idx=3,
            file_name="infected_compartment_rise2_reldeviation")
    if 21 in figures:
        folder = os.path.join(result_dir, "age_resolution_short")
        plot_compartments_age_resolved(
            os.path.join(
                folder, "lct_ageresolved_subcomp10_agegroupinit2_ageresolved"),
            file_name="compartments_ageresolved_A15–34_scenario")
        plot_compartments_age_resolved(os.path.join(
            folder, "lct_ageresolved_subcomp10_agegroupinit5_ageresolved"),
            file_name="compartments_ageresolved_A80+_scenario")
    if 21 in figures:
        folder = os.path.join(result_dir, "age_resolution_short")
        plot_compartments_rel_deviation(os.path.join(folder, "lct_nonageresolved_subcomp10"), [
            os.path.join(folder, "lct_ageresolved_subcomp10_agegroupinit2"), os.path.join(
                folder, "lct_ageresolved_subcomp10_agegroupinit5")], legend_labels=list(
            ["A15–34 scenario", "A80+ scenario"]),
            file_name="compartments_agevsnoage_relativedeviation")
        folder = os.path.join(result_dir, "age_resolution_long")
        plot_compartments_rel_deviation(os.path.join(folder, "lct_nonageresolved_subcomp10"), [
            os.path.join(folder, "lct_ageresolved_subcomp10_agegroupinit2"), os.path.join(
                folder, "lct_ageresolved_subcomp10_agegroupinit5")], legend_labels=list(
            ["A15–34 scenario", "A80+ scenario"]),
            file_name="compartments_agevsnoage_relativedeviation_long")


if __name__ == "__main__":
    main()
