#############################################################################
# Copyright (C) 2020-2025 MEmilio
#
# Authors: Anna Wendler
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
from scipy.stats import lognorm


def get_lognormal_parameters(mean, std):
    """
    Compute shape and scale parameters to use in lognormal distribution for given mean and standard deviation. 
    The lognormal distribution we consider in state_age_function.h is based on the implementation in scipy and the parameters 
    shape and scale are defined accordingly.

    @param[in] mean Mean of the distribution.
    @param[in] std Standard deviation of the distribution.
    @returns Shape and scale parameter of lognormal distribution. 
    """
    variance = std**2

    mean_tmp = np.log(mean**2/np.sqrt(mean**2+variance))
    variance_tmp = np.log(variance/mean**2 + 1)

    shape = np.sqrt(variance_tmp)
    scale = np.exp(mean_tmp)

    # Test if mean and std are as expected for computed shape and scale parameters.
    mean_lognorm, variance_lognorm = lognorm.stats(
        shape, loc=0, scale=scale, moments='mv')

    if np.abs(mean_lognorm-mean) > 1e-8:
        print('Distribution does not have expected mean value.')

    if np.abs(np.sqrt(variance_lognorm)-std) > 1e-8:
        print('Distribution does not have expected standard deviation.')

    return round(shape, 8), round(scale, 8)


def get_weighted_mean(prob_1, stay_time_1, stay_time_2):

    weighted_mean = prob_1*stay_time_1 + (1-prob_1)*stay_time_2

    return weighted_mean


def main():
    shape, scale = get_lognormal_parameters(2.183, 1.052)
    print(
        f"For the given mean and standard deviation, the shape parameter is {shape:.12f} and the scale parameter is {scale:.12f}.")

    weighted_mean = get_weighted_mean(0.793099, 1.1, 8.0)
    print(f"The weighted mean is {weighted_mean:.6f}.")

    weighted_mean = get_weighted_mean(0.078643, 6.6, 8.0)
    print(f"The weighted mean is {weighted_mean:.6f}.")

    weighted_mean = get_weighted_mean(0.173176, 1.5, 18.1)
    print(f"The weighted mean is {weighted_mean:.6f}.")

    weighted_mean = get_weighted_mean(0.387803, 10.7, 18.1)
    print(f"The weighted mean is {weighted_mean:.6f}.")


if __name__ == '__main__':
    main()
