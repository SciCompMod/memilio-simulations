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
import time
import matplotlib.pyplot as plt

import memilio.simulation as mio
import memilio.simulation.oseirdb as oseirdb
from memilio.simulation.oseirdb import interpolate_simulation_result

import logging
mio.set_log_level(mio.LogLevel.Critical)


def simple_seirdb(p, x0, tstart, tend):
    """
    A simple SEIR model with one age group and two parameters:
    - TimeExposed: Duration of the exposed state
    - TimeInfected: Duration of the infected state
    """

    num_groups = 1
    model = oseirdb.Model(num_groups)

    A0 = mio.AgeGroup(0)

    # Compartment transition duration
    model.parameters.TransmissionProbabilityOnContact[A0] = p[0]  # TransmissionProbabilityOnContact
    model.parameters.TimeExposed[A0] =p[1] # TimeExposed
    model.parameters.TimeInfected[A0] = p[2] # TimeInfected
    model.parameters.ProbabilityToRecover[A0] = p[3]  # ProbabilityToRecover
    model.parameters.TransmissionProbabilityFromDead[A0] = p[4]  # TransmissionProbabilityFromDead
    model.parameters.TimeToBurial[A0] = p[5]  # TimeToBurial

    model.populations[A0, oseirdb.InfectionState.Susceptible] = x0[0]
    model.populations[A0, oseirdb.InfectionState.Exposed] = x0[1]
    model.populations[A0, oseirdb.InfectionState.Infected] = x0[2]
    model.populations[A0, oseirdb.InfectionState.Recovered] = x0[3]
    model.populations[A0, oseirdb.InfectionState.Dead] = x0[4]
    model.populations[A0, oseirdb.InfectionState.Buried] = x0[5]

    model.parameters.ContactPatterns.cont_freq_mat[0].baseline = np.ones(
        (num_groups, num_groups)) * 21.0
    model.parameters.ContactPatterns.cont_freq_mat[0].minimum = np.zeros(
        (num_groups, num_groups))

    result = oseirdb.simulate(t0=tstart, tmax=tend, dt=0.1, model=model)
    result_array = np.array(interpolate_simulation_result(result).as_ndarray())


    return result_array

def flow_seirdb(p, x0, tstart, tend):
    """
    A simple SEIRDB model with one age group and 6 parameters:
    """

    num_groups = 1
    model = oseirdb.Model(num_groups)

    A0 = mio.AgeGroup(0)

    # Compartment transition duration
    model.parameters.TransmissionProbabilityOnContact[A0] = p[0]  # TransmissionProbabilityOnContact
    model.parameters.TimeExposed[A0] =p[1] # TimeExposed
    model.parameters.TimeInfected[A0] = p[2] # TimeInfected
    model.parameters.ProbabilityToRecover[A0] = p[3]  # ProbabilityToRecover
    model.parameters.TransmissionProbabilityFromDead[A0] = p[4]  # TransmissionProbabilityFromDead
    model.parameters.TimeToBurial[A0] = p[5]  # TimeToBurial

    model.populations[A0, oseirdb.InfectionState.Susceptible] = x0[0]
    model.populations[A0, oseirdb.InfectionState.Exposed] = x0[1]
    model.populations[A0, oseirdb.InfectionState.Infected] = x0[2]
    model.populations[A0, oseirdb.InfectionState.Recovered] = x0[3]
    model.populations[A0, oseirdb.InfectionState.Dead] = x0[4]
    model.populations[A0, oseirdb.InfectionState.Buried] = x0[5]

    model.parameters.ContactPatterns.cont_freq_mat[0].baseline = np.ones(
        (num_groups, num_groups)) * 21.0
    model.parameters.ContactPatterns.cont_freq_mat[0].minimum = np.zeros(
        (num_groups, num_groups))

    (result, flows) = oseirdb.simulate_flows(t0=tstart, tmax=tend, dt=0.1, model=model)
    result_array = np.array(interpolate_simulation_result(result).as_ndarray())
    flows_array = np.array(interpolate_simulation_result(flows).as_ndarray())

    return result_array, flows_array


