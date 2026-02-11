import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

import memilio.simulation as mio
import memilio.simulation.ssirs as ssirs
from memilio.simulation.ssirs import InfectionState as State
from memilio.simulation.ssirs import Season

from memilio.simulation import AgeGroup, Damping

import logging
mio.set_log_level(mio.LogLevel.Critical)

def ssirs_model(p, u0, tstart, tend):
    """
    A simple SDE SIRS model with one age group, but 3 years with changing seasonality.
    """
    num_groups = 1
    model = ssirs.Model(Season(3))

    # A0 = mio.AgeGroup(0)

    # Parameters fixed
    model.parameters.TransmissionProbabilityOnContact.value = p[0]
    model.parameters.TimeInfected.value = p[1]
    model.parameters.TimeImmune.value = p[2]

    # Parameters per Season
    # 2016-2017
    model.parameters.Seasonality[Season(0)].value = p[3]
    model.parameters.SeasonalitySigma[Season(0)].value = 24
    model.parameters.SeasonalityPeak[Season(0)].value = 173
    # 2017-2018
    model.parameters.Seasonality[Season(1)].value = p[4]
    model.parameters.SeasonalitySigma[Season(1)].value = 24
    model.parameters.SeasonalityPeak[Season(1)].value = 188
    # 2018-2019
    model.parameters.Seasonality[Season(2)].value = p[5]
    model.parameters.SeasonalitySigma[Season(2)].value = 60
    model.parameters.SeasonalityPeak[Season(2)].value = 172



    # fixed contact patterns
    model.parameters.ContactPatterns.baseline = np.ones(
     (num_groups, num_groups)) * 7.95
    model.parameters.ContactPatterns.minimum = np.zeros(
     (num_groups, num_groups))



    # damping per season, 1-d is the multiplicative effect on TransmissionProbabilityOnContact
    model.parameters.ContactPatterns.add_damping(Damping(coeffs=np.r_[0.1], t=365, level=0, type=0))
    model.parameters.ContactPatterns.add_damping(Damping(coeffs=np.r_[-0.1], t=730, level=0, type=0))
    model.parameters.ContactPatterns.add_damping(Damping(coeffs=np.r_[-0.1], t=730, level=0, type=0))

    # initialize populations
    model.populations[State.Susceptible] = u0[0]
    model.populations[State.Infected] = u0[1]
    model.populations[State.Recovered] = u0[2]

    # Intialize season ends
    model.initialize_season_ends()

    # Check parameter constraints
    model.check_constraints()

  
    result = ssirs.simulate_stochastic(t0=tstart, tmax=tend, dt=0.1, model=model)
    result_array = np.array(result.as_ndarray())

    return result_array
