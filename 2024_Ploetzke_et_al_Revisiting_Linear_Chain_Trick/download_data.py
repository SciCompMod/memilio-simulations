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
############################################################################
from datetime import date
import pandas as pd
import os
from memilio.epidata import defaultDict as dd
from memilio.epidata import getCaseData
from memilio.epidata import getDIVIData


def main():

    data_folder= os.path.join(os.path.dirname(
        __file__), "data")

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    # Download data in format required.
    getCaseData.get_case_data(read_data=False,
                              file_format=dd.defaultDict['file_format'],
                              out_folder=data_folder,
                              no_raw=False,
                              start_date=date(2020, 1, 1),
                              end_date=date(2020, 12, 31),
                              impute_dates=True,
                              moving_average=0,
                              make_plot=dd.defaultDict['make_plot'],
                              split_berlin=dd.defaultDict['split_berlin'],
                              rep_date=dd.defaultDict['rep_date'],
                              files='All'
                              )
    getCaseData.get_case_data(read_data=False,
                              file_format=dd.defaultDict['file_format'],
                              out_folder=data_folder,
                              no_raw=False,
                              start_date=date(2020, 1, 1),
                              end_date=date(2020, 12, 31),
                              impute_dates=True,
                              moving_average=7,
                              make_plot=dd.defaultDict['make_plot'],
                              split_berlin=dd.defaultDict['split_berlin'],
                              rep_date=dd.defaultDict['rep_date'],
                              files='All'
                              )
    getDIVIData.get_divi_data(read_data=False,
                              file_format=dd.defaultDict['file_format'],
                              out_folder=data_folder,
                              start_date=date(2020, 1, 1),
                              end_date=date(2020, 12, 31),
                              impute_dates=True,
                              moving_average=0
                              )


if __name__ == "__main__":
    main()
