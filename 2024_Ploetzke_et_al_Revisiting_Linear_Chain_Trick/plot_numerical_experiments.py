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
"""@plot_numerical_experiments.py
Functions to create the plots for sections 4.2 and 4.3 of the paper.

The simulation results to be plotted should be stored in a '../../data/simulation_lct_numerical_experiments' folder 
as .h5 files. Have a look at the README for an explanation of how to create the simulation result data.

Please note that the memilio.epidata package needs to be installed beforehand. 
Have a look at the pycode and the pycode/memilio-epidata READMEs.
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
plotfolder = 'Plots/Plots_numerical_experiments'


def plot_compartments(files, legend_labels, file_name="", compartment_indices=range(8)):
    """ Creates a plot of the simulation results for the compartments. 
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
    fig, axs = plt.subplots(2,
                            math.ceil(len(compartment_indices)/2), sharex='all', num=file_name, tight_layout=True)

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
        for i in compartment_indices:
            if legend_labels[file] in color_dict:
                axs[up_down, left_right].plot(dates,
                                              total[:, i], label=legend_labels[file], linewidth=1.2,
                                              linestyle=linestyle_dict[legend_labels[file]],
                                              color=color_dict[legend_labels[file]])
            else:
                axs[up_down, left_right].plot(dates,
                                              total[:, i], label=legend_labels[file], linewidth=1.2)
            axs[up_down, left_right].set_title(
                secir_dict[i], fontsize=8, pad=3)
            if (left_right < math.ceil(len(compartment_indices)/2)-1):
                left_right += 1
            else:
                left_right = 0
                up_down += 1
        h5file.close()

    # Define some characteristics of the plot.
    for i in range(math.ceil(len(compartment_indices)/2)*2):
        axs[i % 2, int(i/2)].set_xlim(left=0, right=dates[-1])
        if legend_labels[0] == "ODE":
            axs[i % 2, int(i/2)].set_ylim(bottom=0)
        axs[i % 2, int(i/2)].grid(True, linestyle='--')
        axs[i % 2, int(i/2)].tick_params(axis='y', labelsize=7)
        axs[i % 2, int(i/2)].tick_params(axis='x', labelsize=7)

    fig.supxlabel('Simulation time [days]', fontsize=9)

    lines, labels = axs[0, 0].get_legend_handles_labels()
    lgd = fig.legend(lines, labels, ncol=len(legend_labels),  loc='outside lower center',
                     fontsize=8, bbox_to_anchor=(0.5, - 0.065), bbox_transform=fig.transFigure)
    # Size is random such that plot is beautiful.
    fig.set_size_inches(7.5/3*math.ceil(len(compartment_indices)/2), 10.5/2.5)
    fig.tight_layout(pad=0, w_pad=0.3, h_pad=0.4)
    fig.subplots_adjust(bottom=0.09)

    # Save result.
    if file_name:
        fig.savefig(plotfolder+'/'+file_name+'.png',
                    bbox_extra_artists=(lgd,),  bbox_inches='tight', dpi=500)
    plt.close()


def plot_rel_deviation(ode_filename, files, legend_labels,  compartment_idx=1, file_name=""):
    """ Creates a plot of the simulation results for one specified compartment. 
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


def plot_single_compartment(files, legend_labels,  compartment_idx=1, file_name=""):
    """ Creates a plot of the simulation results for one specified compartment. 
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
        if legend_labels[file] in color_dict:
            plt.plot(dates, total[:, compartment_idx], linewidth=1.2,
                     linestyle=linestyle_dict[legend_labels[file]], color=color_dict[legend_labels[file]])
        else:
            plt.plot(dates, total[:, compartment_idx], linewidth=1.2)

        h5file.close()

    plt.xlabel('Simulation time [days]', fontsize=fontsize_labels)
    plt.yticks(fontsize=9)
    plt.ylabel('Number of ' +
               secir_dict[compartment_idx] + " individuals", fontsize=fontsize_labels)
    plt.xlim(left=0, right=dates[-1])
    plt.title(secir_dict[compartment_idx], fontsize=fontsize_legends)
    plt.legend(legend_labels, fontsize=fontsize_legends, framealpha=0.5)
    plt.grid(True, linestyle='--')
    plt.tight_layout()

    # Save result.
    if file_name:
        plt.savefig(plotfolder+'/'+file_name +
                    '.png', bbox_inches='tight', dpi=500)
    plt.close()


def plot_subcompartments3D(file, num_subcompartments, compartment_idx, first_time_idx, file_name=""):
    """ Creates a 3d plot visualizing the distribution of people in the subcompartments over time 
    (starting from first_time_idx) of one simulation result and one specific compartment. 

    @param[in] file: Paths to file (without .h5 extension) containing the simulation result to be plotted.
        The result should contain a splitting in subcompartments. The function assumes that each of the compartments
         uses the same number of subcompartments num_subcompartments.
        To make the plot clearer, the time steps provided should not be too small (e.g. days).
    @param[in] num_subcompartments: Number of subcompartments used for the LCT model in the simulation.
    @param[in] compartment_idx: The index of the compartment to be plotted.
    @param[in] first_time_idx: The first time index to be plotted. 
            Can be used to skip the initial period in which no change is visible.
    @param[in] file_name: The name of the file where the plot will be saved.
            If an empty string is provided, the plot will not be saved.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    h5file = h5py.File(str(file) + '.h5', 'r')

    if not ((compartment_idx > 0) and (compartment_idx < 7)):
        raise Exception("Use valid compartment index.")
    if (len(list(h5file.keys())) > 1):
        raise gd.DataError("File should contain one dataset.")
    if (len(list(h5file[list(h5file.keys())[0]].keys())) > 3):
        raise gd.DataError("Expected only one group.")

    data = h5file[list(h5file.keys())[0]]
    dates = data['Time'][:]
    # As there should be only one Group, total is the simulation result.
    total = data['Total'][:, :]
    # Assume, that each of the compartments uses the same number of subcompartments num_subcompartments.
    first_idx_comp = 1+(compartment_idx-1)*num_subcompartments

    # Create data.
    x = [i-0.5 for i in range(1, num_subcompartments+1)]
    x, y = np.meshgrid(x, dates[first_time_idx:]-0.5)
    flat_x, flat_y = x.ravel(), y.ravel()
    comp_total = 0
    z = np.zeros(x.shape)
    for i in range(x.shape[0]):
        comp_total = np.sum(
            total[first_time_idx+i, first_idx_comp:first_idx_comp+num_subcompartments])
        for j in range(x.shape[1]):
            z[i, j] = total[first_time_idx+i, first_idx_comp+j]
    flat_z = z.ravel()
    bottom = np.zeros_like(flat_x)

    # Define color of the plot.
    cmap = plt.get_cmap('plasma')
    norm = Normalize(vmin=min(flat_z), vmax=max(flat_z))
    colors = cmap(norm(flat_z))

    ax.bar3d(flat_x, flat_y, 0, 1, 1, flat_z, shade=True, color=colors)

    ax.set_xlabel('Index of subcompartment', labelpad=-5, fontsize=9)
    a = list(range(0, num_subcompartments+1, 5))
    a[0] += 1
    ax.set_xticks(a)
    ax.set_xlim(left=0.5, right=num_subcompartments+0.5)
    ax.tick_params(axis='x', pad=-5, labelsize=9)
    ax.set_xticks(range(1, num_subcompartments), minor=True)

    ax.set_ylabel('Simulation time [days]', labelpad=-5, fontsize=9)
    ax.set_ylim(bottom=first_time_idx-0.5, top=dates[-1]+0.5)
    ax.set_yticks(range(int(dates[first_time_idx]),
                  int(dates[-1])+1, 2))
    ax.set_yticklabels(range(int(dates[first_time_idx]),
                             int(dates[-1])+1, 2), ha='left', va='center')
    ax.set_yticks(range(int(dates[first_time_idx]),
                        int(dates[-1])+1), minor=True)
    ax.tick_params(axis='y', pad=-4, labelsize=9)

    ax.set_zlim(bottom=0)
    ax.tick_params(axis='z', pad=1, labelsize=9)
    ax.set_title(secir_dict[compartment_idx],
                 fontsize=fontsize_labels, pad=-50)

    # Beautiful colorbar.
    sc = cm.ScalarMappable(cmap=cmap, norm=norm)
    sc.set_array([])
    cbar = plt.colorbar(sc, ax=ax, shrink=0.75)

    h5file.close()

    if file_name:
        plt.savefig(plotfolder+'/'+file_name +
                    '.png', bbox_inches='tight', dpi=500)
    plt.close()


def plot_epidemic_peak_size(func_get_file_name, Reffs, subcompartments, file_name=""):
    """ Creates a plot to visualize the maximal number of daily new transmissions (size of the epidemic peak) 
    for different effective reproduction numbers and different assumptions regarding the number of subcompartments.

    @param[in] func_get_file_name: Function that returns the paths to a file with suitable simulation results
        using the parameters "Reff" and "subcompartment".
    @param[in] Reffs: List with the effective reproduction number for which the peak sizes should be compared.
    @param[in] subcompartments: List with the subcompartment numbers for which the peak sizes should be compared.
    @param[in] file_name: The name of the file where the plot will be saved.
            If an empty string is provided, the plot will not be saved.
    """
    fig = plt.figure()

    # For each Reff, get the peak size for each number of subcompartments and plot the result depending on
    # the subcompartment numbers.
    for Reff in Reffs:
        y = np.zeros(len(subcompartments))
        for i in range(len(subcompartments)):
            y[i] = get_epidemic_peak_size(
                func_get_file_name(Reff=Reff, subcompartment=subcompartments[i]))
        plt.plot(subcompartments, y, 'o-', linewidth=1.2,
                 label=str(Reff))

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10,
               title="$\\mathcal{R}_{\\text{eff}}(0)\\approx$", title_fontsize=11)
    plt.ylim(bottom=0)
    plt.xlabel('Number of subcompartments', fontsize=fontsize_labels)
    plt.ylabel('Maximum daily new transmissions', fontsize=fontsize_labels)
    plt.grid(True, linestyle='--')
    plt.tight_layout()

    if file_name:
        plt.savefig(plotfolder+'/'+file_name +
                    '.png', bbox_inches='tight', dpi=500)
    plt.close()


def get_epidemic_peak_size(file):
    """ Returns the maximum of the daily new transmissions (peak size) of a simulation result.
    @param[in] file: Paths to file (without .h5 extension) containing the simulation result.
    """
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

    # Calculate daily new transmissions and return the maximum value.
    daily_new_transmissions = (
        total[:-1, 0]-total[1:, 0])/(dates[1:]-dates[:-1])
    max = np.max(daily_new_transmissions)
    h5file.close()
    return max


def plot_epidemic_peak_timing(func_get_file_name, Reffs, subcompartments, file_name=""):
    """ Creates a plot to visualize the day at which the maximal number of daily new transmissions is reached 
    (timing of the epidemic peak) for different effective reproduction numbers and different assumptions 
    regarding the number of subcompartments.

    @param[in] func_get_file_name: Function that returns the path to the files with suitable simulation results
        using the parameters "Reff" and "subcompartment".
    @param[in] Reffs: List with the effective reproduction number for which the peak timing should be compared.
    @param[in] subcompartments: List with the subcompartment numbers for which the peak timing should be compared.
    @param[in] file_name: The name of the file where the plot will be saved.
            If an empty string is provided, the plot will not be saved.
    """
    fig = plt.figure()
    # For each Reff, get the peak timing for each number of subcompartments and plot the result in dependence of
    # the subcompartment numbers.
    for Reff in Reffs:
        y = np.zeros(len(subcompartments))
        for i in range(len(subcompartments)):
            y[i] = get_epidemic_peak_timing(
                func_get_file_name(Reff=Reff, subcompartment=subcompartments[i]))
        plt.plot(subcompartments, y, 'o-', linewidth=1.2,
                 label=str(Reff))

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10,
               title="$\\mathcal{R}_{\\text{eff}}(0)\\approx$", title_fontsize=11)
    plt.ylim(bottom=0)
    plt.xlabel('Number of subcompartments', fontsize=fontsize_labels)
    plt.ylabel('Simulation day of peak [days]', fontsize=fontsize_labels)
    plt.grid(True, linestyle='--')
    plt.tight_layout()

    if file_name:
        plt.savefig(plotfolder+'/'+file_name +
                    '.png', bbox_inches='tight', dpi=500)
    plt.close()


def get_epidemic_peak_timing(file):
    """ Returns the day at which the maximum number of the daily new transmissions is reached (peak timing)
     of a simulation result.
    @param[in] file: Paths to file (without .h5 extension) containing the simulation result.
    """
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

    # Calculate daily new transmissions and return date referring to the maximum value.
    daily_new_transmissions = (
        total[:-1, 0]-total[1:, 0])/(dates[1:]-dates[:-1])
    argmax = np.argmax(daily_new_transmissions)
    h5file.close()
    return dates[argmax]


def plot_daily_new_transmissions(files, legend_labels, file_name="", tmax=0):
    """ Creates a plot to compare the simulation results for the daily new transmissions (number of people leaving the
    susceptible compartment per day).  
    This function can be used to compare daily new transmissions for different simulation results,
    e.g., obtained with different models or different parameter specifications.

    @param[in] files: List of paths to files (without .h5 extension) containing simulation results to be plotted.
         Results should contain exactly 8 compartments (so use accumulated numbers for LCT models). 
    @param[in] legend_labels: List of names for the results to be used for the plot legend.
    @param[in] file_name: The name of the file where the plot will be saved.
            If an empty string is provided, the plot will not be saved.
    @param[in] tmax: The time frame to be plotted is [0, tmax]. 
    """
    plt.figure(file_name)

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
        daily_new_transmissions = (
            total[:-1, 0]-total[1:, 0])/(dates[1:]-dates[:-1])
        # Plot result.
        if legend_labels[file] in color_dict:
            plt.plot(dates[1:], daily_new_transmissions, linewidth=1.2,
                     linestyle=linestyle_dict[legend_labels[file]], color=color_dict[legend_labels[file]])
        else:
            plt.plot(dates[1:], daily_new_transmissions, linewidth=1.2)

        h5file.close()

    plt.xlabel('Simulation time [days]', fontsize=fontsize_labels)
    plt.yticks(fontsize=9)
    plt.ylabel('Daily new transmissions', fontsize=fontsize_labels)
    plt.ylim(bottom=0, top=np.max(daily_new_transmissions)*1.05)
    if tmax > 0:
        plt.xlim(left=0, right=tmax)
    else:
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
    figures = list(range(3, 12)) + [13] + [20]

    if 3 in figures:
        folder = os.path.join(result_dir, "dropReff")
        plot_daily_new_transmissions([get_file_name(folder, 0.5, 2, 1), get_file_name(folder, 0.5, 2, 3),
                                      get_file_name(folder, 0.5, 2, 10), get_file_name(folder, 0.5, 2, 50), get_file_name(folder, 0.5, 2, 0)],
                                     legend_labels=list(
            ["ODE", "LCT3", "LCT10", "LCT50", "LCTvar"]),
            file_name="new_infections_drophalf")
        folder = os.path.join(result_dir, "riseReffTo2short")
        plot_daily_new_transmissions([get_file_name(folder, 2, 2, 1), get_file_name(folder, 2, 2, 3),
                                      get_file_name(folder, 2, 2, 10), get_file_name(folder, 2,  2, 50), get_file_name(folder, 2,  2, 0)],
                                     legend_labels=list(
            ["ODE", "LCT3", "LCT10", "LCT50", "LCTvar"]),
            file_name="new_infections_rise2")
    if 4 in figures:
        folder = os.path.join(result_dir, "dropReff")
        plot_single_compartment([get_file_name(folder, 0.5, 2, 1), get_file_name(folder, 0.5, 2, 3),
                                 get_file_name(folder, 0.5, 2, 10), get_file_name(folder, 0.5, 2, 50), get_file_name(folder, 0.5, 2, 0)],
                                legend_labels=list(
            ["ODE", "LCT3", "LCT10", "LCT50", "LCTvar"]),
            compartment_idx=2, file_name="carrier_compartment_drophalf")
        plot_single_compartment([get_file_name(folder, 0.5, 2, 1), get_file_name(folder, 0.5, 2, 3),
                                 get_file_name(folder, 0.5, 2, 10), get_file_name(folder, 0.5, 2, 50), get_file_name(folder, 0.5, 2, 0)],
                                legend_labels=list(
            ["ODE", "LCT3", "LCT10", "LCT50", "LCTvar"]),
            compartment_idx=3, file_name="infected_compartment_drophalf")
        folder = os.path.join(result_dir, "riseReffTo2short")
        plot_single_compartment([get_file_name(folder, 2, 2, 1), get_file_name(folder, 2, 2, 3),
                                 get_file_name(folder, 2, 2, 10), get_file_name(folder, 2, 2, 50), get_file_name(folder, 2, 2, 0)],
                                legend_labels=list(
            ["ODE", "LCT3", "LCT10", "LCT50", "LCTvar"]),
            compartment_idx=2, file_name="carrier_compartment_rise2")
        plot_single_compartment([get_file_name(folder, 2, 2, 1), get_file_name(folder, 2, 2, 3),
                                 get_file_name(folder, 2, 2, 10), get_file_name(folder, 2, 2, 50), get_file_name(folder, 2, 2, 0)],
                                legend_labels=list(
            ["ODE", "LCT3", "LCT10", "LCT50", "LCTvar"]),
            compartment_idx=3, file_name="infected_compartment_rise2")
    if 5 in figures:
        folder = os.path.join(result_dir, "riseReffTo2short")
        subcompartments = [10, 50]
        for i in subcompartments:
            plot_subcompartments3D(get_file_name(folder, 2, 2, i,  True), i, 1,
                                   1, file_name="subcompartments"+f"{i}"+"_exposed")
            plot_subcompartments3D(get_file_name(folder, 2, 2, i,  True), i, 2,
                                   1, file_name="subcompartments"+f"{i}"+"_carrier")
            plot_subcompartments3D(get_file_name(folder, 2, 2, i,  True), i, 3,
                                   1, file_name="subcompartments"+f"{i}"+"_infected")
    if 6 in figures:
        folder = os.path.join(result_dir, "riseReffTo2shortTEhalved")
        plot_daily_new_transmissions([get_file_name(folder, 2, 2, 1), get_file_name(folder, 2, 2, 3),
                                      get_file_name(folder, 2, 2, 10), get_file_name(folder, 2, 2, 50)],
                                     legend_labels=list(
            ["ODE", "LCT3", "LCT10", "LCT50"]),
            file_name="new_infections_rise2_TEhalved")
        plot_subcompartments3D(get_file_name(
            folder, 2, 2, 50,  True), 50, 2, 1, file_name="subcompartments50_carrier_TEhalved")
    if 7 in figures:
        folder = os.path.join(result_dir, "riseRefflong")
        plot_daily_new_transmissions([get_file_name(folder, 2, 0, 1), get_file_name(folder, 2, 0, 3),
                                      get_file_name(folder, 2, 0, 10), get_file_name(folder, 2, 0, 50), get_file_name(folder, 2, 0, 0)],
                                     legend_labels=list(
            ["ODE", "LCT3", "LCT10", "LCT50", "LCTvar"]),
            file_name="new_infections_rise2_long", tmax=150)
        plot_daily_new_transmissions([get_file_name(folder, 4, 0, 1), get_file_name(folder, 4, 0, 3),
                                      get_file_name(folder, 4, 0, 10), get_file_name(folder, 4, 0, 50), get_file_name(folder, 4, 0, 0)],
                                     legend_labels=list(
            ["ODE", "LCT3", "LCT10", "LCT50", "LCTvar"]),
            file_name="new_infections_rise4_long", tmax=70)
    if 8 in figures:
        folder = os.path.join(result_dir, "riseRefflong")
        Reffs = list(range(2, 11))
        subcompartments = list(range(1, 11))
        plot_epidemic_peak_size(lambda Reff, subcompartment: get_file_name(folder, Reff, 0, subcompartment),
                                Reffs, subcompartments, file_name="compare_peak_size")
        plot_epidemic_peak_timing(lambda Reff, subcompartment: get_file_name(folder, Reff, 0, subcompartment),
                                  Reffs, subcompartments, file_name="compare_peak_days")
    if 9 in figures:
        folder = os.path.join(result_dir, "riseRefflongTEhalved")
        Reffs = list(range(2, 11))
        subcompartments = list(range(1, 11))
        plot_epidemic_peak_size(lambda Reff, subcompartment: get_file_name(folder, Reff, 0, subcompartment),
                                Reffs, subcompartments, file_name="compare_peak_size_TEhalved")
        plot_epidemic_peak_timing(lambda Reff, subcompartment: get_file_name(folder, Reff, 0, subcompartment),
                                  Reffs, subcompartments, file_name="compare_peak_days_TEhalved")
    if 10 in figures:
        folder = os.path.join(result_dir, "riseRefflongTEdoubled")
        Reffs = list(range(2, 11))
        subcompartments = list(range(1, 11))
        plot_epidemic_peak_size(lambda Reff, subcompartment: get_file_name(folder, Reff, 0, subcompartment),
                                Reffs, subcompartments, file_name="compare_peak_size_TEdoubled")
        plot_epidemic_peak_timing(lambda Reff, subcompartment: get_file_name(folder, Reff, 0, subcompartment),
                                  Reffs, subcompartments, file_name="compare_peak_days_TEdoubled")
    if 11 in figures:
        folder = os.path.join(result_dir, "riseRefflong")
        plot_compartments([get_file_name(folder, 2, 0, 1), get_file_name(folder, 2, 0, 3),
                           get_file_name(folder, 2, 0, 10), get_file_name(folder, 2, 0, 50), get_file_name(folder, 2, 0, 0)],
                          legend_labels=list(
            ["ODE", "LCT3", "LCT10", "LCT50", "LCTvar"]),
            file_name="compartments_rise2_long")
    if 13 in figures:
        folder = os.path.join(result_dir, "age_resolution_short")
        plot_compartments([
            os.path.join(folder, "lct_ageresolved_subcomp10_agegroupinit2"), os.path.join(
                folder, "lct_ageresolved_subcomp10_agegroupinit5"),
            os.path.join(folder, "lct_nonageresolved_subcomp10")], legend_labels=list(
            ["A15–34 scenario", "A80+ scenario", "Non age-resolved scenario"]),
            file_name="compartments_agevsnoage_short")
        folder = os.path.join(result_dir, "age_resolution_long")
        plot_compartments([os.path.join(folder, "lct_ageresolved_subcomp10_agegroupinit2"),
                           os.path.join(
                               folder, "lct_ageresolved_subcomp10_agegroupinit5"),
                           os.path.join(folder, "lct_nonageresolved_subcomp10")], legend_labels=list(
            ["A15–34 scenario", "A80+ scenario", "Non age-resolved scenario"]),
            file_name="compartments_agevsnoage_long")
        a = get_epidemic_peak_size(os.path.join(
            folder, "lct_ageresolved_subcomp10_agegroupinit2"))
        print("Peak number of daily new transmissions of A15-34 scenario: " +
              f"{a}")
        b = get_epidemic_peak_size(os.path.join(
            folder, "lct_ageresolved_subcomp10_agegroupinit5"))
        print("Peak number of daily new transmissions of A80+ scenario: " +
              f"{b}")

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


if __name__ == "__main__":
    main()
