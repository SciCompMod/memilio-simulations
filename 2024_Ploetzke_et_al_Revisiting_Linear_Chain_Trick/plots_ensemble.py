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
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import json
import h5py

colors = ["tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown"]
fontsize_labels = 16
fontsize_legends = 12
plotfolder = 'Plots/Plots_ensemble'

color_dict = {'p50': 'C0',
              'p25': 'C0',
              'p75': 'C0',
              'p05':  'C0',
              'p95':  'C0',
              }
linestyle_dict = {'p50': 'solid',
                  'p25': 'dashed',
                  'p75': 'dashed',
                  'p05':  'dotted',
                  'p95':  'dotted',
                  }


def extract_json_segments(input_file, output_file):
    """ Cleans up the output of the run time measurement using the get_runtimes_lct.sh
    script and writes the relevant results in a '.json' file.

    @param[in] input_file: Path to the '.txt' file where the information with the run time measurements is stored.
        The relevant data should be contained in brackets {} within the file and should have the same
         format as one json entry. Information in the file which is not contained in brackets will be ignored.
    @param[in] output_file: Path to the '.json' file with the pure run time measurement data.
            This file will be created in this function.
    """
    # Open file and read data.
    with open(input_file, 'r') as file:
        text = file.read()

    # Extract data in {}.
    segments = re.findall(r'\{(.*?)\}', text, re.DOTALL)

    # Convert segments into JSON objects.
    json_data = []
    for segment in segments:
        json_object = json.loads('{' + segment + '}')
        json_data.append(json_object)

    # Save JSON-objects in JSON file.
    with open(output_file, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)


def plot_scaling(json_file, file_name=''):
    """

    @param[in] json_file: Path to the json file containing run time measurements
    @param[in] file_name: The name of the file where the plot will be saved.
            If an empty string is provided, the plot will not be saved.
    """
    fig = plt.figure()
    df = pd.read_json(json_file)

    plt.loglog(df["Processes"], df["Time"],  marker='o', linewidth=1.5)

    start_optimal_line = df["Time"].max()
    optimal_line = 1.0/df['Processes'].values * start_optimal_line
    plt.loglog(df['Processes'], optimal_line, linewidth=2,
               linestyle='dashed', color='gray', label='Optimal')

    plt.xlabel('Number of cores', fontsize=fontsize_labels)
    plt.ylabel('Run time [seconds]', fontsize=fontsize_labels)
    plt.yticks(fontsize=fontsize_legends)
    plt.xticks(fontsize=fontsize_legends)
    plt.grid(True, linestyle='--')
    plt.xticks(df["Processes"], labels=[str(int(x))
               for x in df["Processes"]], fontsize=fontsize_legends)
    plt.tight_layout()

    if file_name:
        fig.savefig(plotfolder+'/'+file_name+'.png',
                    bbox_inches='tight', dpi=500)
    plt.close()


def plot_daily_new_transmissions(files, start_date, tmax, legend_labels, file_name=""):
    """ 
    """

    if len(files) != 5:
        raise gd.DataError("Expect 5 poercentiles.")

    plt.figure(file_name)
    num_days = tmax + 1

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
        incidence = (total[:-1, 0] - total[1:, 0])/(dates[1:]-dates[:-1])

        # Plot result.

        plt.plot(dates[:-1], incidence, linewidth=1.2,
                 linestyle=linestyle_dict[legend_labels[file]], color=color_dict[legend_labels[file]])
        h5file.close()

    # Percentile25 75
    h5filep25 = h5py.File(str(files[1]) + '.h5', 'r')
    datap25 = h5filep25[list(h5filep25.keys())[0]]
    dates = datap25['Time'][:]
    h5filep75 = h5py.File(str(files[3]) + '.h5', 'r')
    datap75 = h5filep75[list(h5filep75.keys())[0]]
    totalp25 = datap25['Total'][:, :]
    incidencep25 = (totalp25[:-1, 0] - totalp25[1:, 0])/(dates[1:]-dates[:-1])
    totalp75 = datap75['Total'][:, :]
    incidencep75 = (totalp75[:-1, 0] - totalp75[1:, 0])/(dates[1:]-dates[:-1])

    plt.fill_between(dates[:-1], incidencep25,
                     incidencep75,
                     color=color_dict['p25'], alpha=0.4)
    h5filep25.close()
    h5filep75.close()

    # Percentile 0595
    h5filep05 = h5py.File(str(files[0]) + '.h5', 'r')
    datap05 = h5filep05[list(h5filep05.keys())[0]]
    dates = datap05['Time'][:]
    h5filep95 = h5py.File(str(files[4]) + '.h5', 'r')
    datap95 = h5filep95[list(h5filep95.keys())[0]]

    totalp05 = datap05['Total'][:, :]
    incidencep05 = (totalp05[:-1, 0] - totalp05[1:, 0])/(dates[1:]-dates[:-1])
    totalp95 = datap95['Total'][:, :]
    incidencep95 = (totalp95[:-1, 0] - totalp95[1:, 0])/(dates[1:]-dates[:-1])

    plt.fill_between(dates[:-1], incidencep05,
                     incidencep95,
                     color=color_dict['p95'], alpha=0.2)
    h5filep05.close()
    h5filep95.close()

    plt.ylabel('Daily new transmissions', fontsize=fontsize_labels)
    plt.ylim(bottom=0)
    plt.xlabel('Simulation time [days]', fontsize=fontsize_labels)
    plt.xlim(left=0, right=tmax-1)
    # Define x-ticks as dates.
    # datelist = np.array(pd.date_range(start_date.date(),
    #                                   periods=tmax, freq='D').strftime('%m-%d').tolist())
    # tick_range = (np.arange(int((tmax - 1) / 5) + 1) * 5)
    # plt.xticks(tick_range, datelist[tick_range],
    #            rotation=45, fontsize=12)
    # plt.xticks(np.arange(tmax), minor=True)

    plt.legend(legend_labels, fontsize=fontsize_legends, framealpha=0.5)
    plt.grid(True, linestyle='--')
    plt.tight_layout()

    # Save result.
    if file_name:
        plt.savefig(plotfolder+'/'+file_name +
                    '.png', bbox_inches='tight', dpi=500)
    plt.close()
    print(" ")


def main():
    if not os.path.isdir(plotfolder):
        os.makedirs(plotfolder)

    # Simulation results should be stored in this folder.
    result_dir = os.path.join(os.path.dirname(
        __file__), "simulation_results", "simulation_lct_covid19_ensemble")

    start_date = '2020-10-1'
    start_date_timestamp = pd.Timestamp(start_date)
    tmax = 30

    # Runtime plot
    file_name = 'runtimes'
    paths_to_file = os.path.join(result_dir, file_name)
    extract_json_segments(paths_to_file+'.txt', paths_to_file+'.json')
    plot_scaling(paths_to_file+'.json', file_name)

    # Percentile plot
    plot_daily_new_transmissions([os.path.join(result_dir, "lct_2020-10-1_subcomp0_np128_percentiles_p5"), os.path.join(result_dir, "lct_2020-10-1_subcomp0_np128_percentiles_p25"),
                                  os.path.join(result_dir, "lct_2020-10-1_subcomp0_np128_percentiles_p50"), os.path.join(result_dir, "lct_2020-10-1_subcomp0_np128_percentiles_p75"), os.path.join(result_dir, "lct_2020-10-1_subcomp0_np128_percentiles_p95"),],
                                 start_date_timestamp, tmax,
                                 legend_labels=list(
        ["p05", "p25", "p50", "p75", "p95"]),
        file_name="ensemble_runs")


if __name__ == "__main__":
    main()
