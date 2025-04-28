/* 
* Copyright (C) 2020-2025 MEmilio
*
* Authors: Lena Ploetzke
*
* Contact: Martin J. Kuehn <Martin.Kuehn@DLR.de>
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include "lct_secir/model.h"
#include "lct_secir/infection_state.h"
#include "lct_secir/initializer_flows.h"

#include "memilio/config.h"
#include "memilio/epidemiology/contact_matrix.h"
#include "memilio/epidemiology/uncertain_matrix.h"
#include "memilio/epidemiology/lct_infection_state.h"
#include "memilio/math/eigen.h"
#include "memilio/utils/logging.h"
#include "memilio/utils/time_series.h"
#include "memilio/compartments/simulation.h"
#include "memilio/data/analyze_result.h"
#include "memilio/io/result_io.h"
#include "memilio/io/io.h"

#include "boost/numeric/odeint/stepper/runge_kutta_cash_karp54.hpp"
#include <string>
#include <iostream>
#include <vector>

namespace params
{
// num_subcompartments is used as a template argument and has to be a constexpr.
constexpr size_t num_subcompartments = NUM_SUBCOMPARTMENTS;

// Define (non-age-resolved) parameters.
const ScalarType dt                             = 0.01;
const ScalarType seasonality                    = 0.;
const ScalarType relativeTransmissionNoSymptoms = 1.;
const ScalarType riskOfInfectionFromSymptomatic = 0.3;
const ScalarType total_population               = 83155031.0;

ScalarType timeExposed                            = 3.335;
const ScalarType timeInfectedNoSymptoms           = 2.58916;
const ScalarType timeInfectedSymptoms             = 6.94547;
const ScalarType timeInfectedSevere               = 7.28196;
const ScalarType timeInfectedCritical             = 13.066;
const ScalarType transmissionProbabilityOnContact = 0.07333;
const ScalarType recoveredPerInfectedNoSymptoms   = 0.206901;
const ScalarType severePerInfectedSymptoms        = 0.07864;
const ScalarType criticalPerSevere                = 0.17318;
const ScalarType deathsPerCritical                = 0.21718;

using InfState = mio::lsecir::InfectionState;
using LctState =
    std::conditional<(num_subcompartments == 0), mio::LctInfectionState<InfState, 1, 3, 3, 7, 7, 13, 1, 1>,
                     mio::LctInfectionState<InfState, 1, num_subcompartments, num_subcompartments, num_subcompartments,
                                            num_subcompartments, num_subcompartments, 1, 1>>::type;
} // namespace params

/** 
* @brief Constructs an initial value vector such that the initial infection dynamic is constant.
*   
*   The initial value vector is constructed based on a value of approximately 4050 for the daily new transmissions. 
*   This value is based on some official reporting numbers for Germany 
*   (see Robert Koch-Institut, Täglicher Lagebericht am 15.10.2020,
*   https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/Situationsberichte/Okt_2020/2020-10-15-de.pdf?__blob=publicationFile).
*   The vector is constructed such that the daily new transmissions remain constant if the effective
*   reproduction number is equal to 1. 
*   Derived numbers for compartments are distributed uniformly to the subcompartments.
*   To define the vector, we assume that individuals behave exactly as defined by the epidemiological parameters 
*   (that means that they stay exactly the mean stay time in each compartment and that the transition probabilities are accurate) 
*   and that the new transmissions are constant over time. 
*   The value for the Recovered compartment is also set according to the reported numbers 
*   and the number of dead individuals is set to zero.
*   
* @param[in] tReff TODO
* @returns The initial value vector including subcompartments.
*/
std::vector<ScalarType> get_initial_values(ScalarType tReff = 2.)
{
    using namespace params;

    if (tReff > 0.) {
        const ScalarType dailyNewTransmissionsReported = (34.1 / 7) * total_population / 100000;

        // Firstly, we calculate an initial value vector without division in subcompartments.
        // Assume that individuals behave exactly as defined by the epidemiological parameters.
        std::vector<ScalarType> init_compartments((size_t)InfState::Count);
        // If the number of daily new transmissions was constant within the last days and individuals remain exactly
        // the average time in the Exposed compartment, we currently have timeExposed * dailyNewTransmissionsReported individuals.
        init_compartments[(size_t)InfState::Exposed] = timeExposed * dailyNewTransmissionsReported;
        init_compartments[(size_t)InfState::InfectedNoSymptoms] =
            timeInfectedNoSymptoms * dailyNewTransmissionsReported;
        // Same argument as for Exposed but we have to take into account the probability to become symptomatic.
        init_compartments[(size_t)InfState::InfectedSymptoms] =
            (1 - recoveredPerInfectedNoSymptoms) * timeInfectedSymptoms * dailyNewTransmissionsReported;
        init_compartments[(size_t)InfState::InfectedSevere] = (1 - recoveredPerInfectedNoSymptoms) *
                                                              severePerInfectedSymptoms * timeInfectedSevere *
                                                              dailyNewTransmissionsReported;
        init_compartments[(size_t)InfState::InfectedCritical] = (1 - recoveredPerInfectedNoSymptoms) *
                                                                severePerInfectedSymptoms * criticalPerSevere *
                                                                timeInfectedCritical * dailyNewTransmissionsReported;
        // Number is from official RKI data as stated above.
        init_compartments[(size_t)InfState::Recovered] = 275292.;
        // Set initial Dead compartment to zero for this experiment. This way, we can visualize the new deaths in the simulation time.
        init_compartments[(size_t)InfState::Dead] = 0.;
        // Set the number of Susceptibles to the remaining number of people in the population.
        init_compartments[(size_t)InfState::Susceptible] = total_population;
        for (size_t i = (size_t)InfState::Exposed; i < (size_t)InfState::Count; i++) {
            init_compartments[(size_t)InfState::Susceptible] -= init_compartments[i];
        }

        // Now, we construct an initial value vector with division in subcompartments.
        // Compartment sizes are distributed uniformly to the subcompartments.
        std::vector<ScalarType> initial_value_vector;
        initial_value_vector.push_back(init_compartments[(size_t)InfState::Susceptible]);
        for (size_t i = 0; i < LctState::get_num_subcompartments<InfState::Exposed>(); i++) {
            initial_value_vector.push_back(init_compartments[(size_t)InfState::Exposed] /
                                           LctState::get_num_subcompartments<InfState::Exposed>());
        }
        for (size_t i = 0; i < LctState::get_num_subcompartments<InfState::InfectedNoSymptoms>(); i++) {
            initial_value_vector.push_back(init_compartments[(size_t)InfState::InfectedNoSymptoms] /
                                           LctState::get_num_subcompartments<InfState::InfectedNoSymptoms>());
        }
        for (size_t i = 0; i < LctState::get_num_subcompartments<InfState::InfectedSymptoms>(); i++) {
            initial_value_vector.push_back(init_compartments[(size_t)InfState::InfectedSymptoms] /
                                           LctState::get_num_subcompartments<InfState::InfectedSymptoms>());
        }
        for (size_t i = 0; i < LctState::get_num_subcompartments<InfState::InfectedSevere>(); i++) {
            initial_value_vector.push_back(init_compartments[(size_t)InfState::InfectedSevere] /
                                           LctState::get_num_subcompartments<InfState::InfectedSevere>());
        }
        for (size_t i = 0; i < LctState::get_num_subcompartments<InfState::InfectedCritical>(); i++) {
            initial_value_vector.push_back(init_compartments[(size_t)InfState::InfectedCritical] /
                                           LctState::get_num_subcompartments<InfState::InfectedCritical>());
        }
        initial_value_vector.push_back(init_compartments[(size_t)InfState::Recovered]);
        initial_value_vector.push_back(init_compartments[(size_t)InfState::Dead]);
        return initial_value_vector;
    }
    else {
        std::vector<ScalarType> initial_value_vector(LctState::Count, 0.);
        initial_value_vector[LctState::get_first_index<InfState::Susceptible>()] = total_population - 500.;
        for (size_t i = 0; i < LctState::get_num_subcompartments<InfState::Exposed>(); i++) {
            initial_value_vector[LctState::get_first_index<InfState::Exposed>() + i] =
                500. / LctState::get_num_subcompartments<InfState::Exposed>();
        }
        return initial_value_vector;
    }
}

/** 
* @brief Perform simulation to examine the impact of the distribution assumption. 
*
*   The simulation uses LCT models with Covid-19 inspired parameters and an initial contact rate such that the 
*   effective reproduction number is initially approximately equal to one. The initial values are chosen such that the
*   daily new transmissions remain constant. The contact rate is changed at simulation day tReff such that the effective 
*   reproduction number at simulation day tReff is equal to the input Reff. 
*   Therefore, we simulate a change point at day tReff.
*   
* @param[in] Reff The effective reproduction number to be set at simulation time tReff. Please use a number greater zero.
* @param[in] tReff The time of the change point from an effective reproduction number of one to Reff.
* @param[in] tmax Time horizon of the simulation.
* @param[in] save_dir Specifies the directory where the results should be stored.
*    Provide an empty string if results should not be saved.
* @param[in] save_subcompartments If true, the result will be saved with division in subcompartments. Default is false.
* @param[in] scale_TimeExposed The value for TimeExposed (=3.335) is scaled by this value before the simulation.
*   Default is 1 (no scaling).
* @param[in] print_final_size If true, the final size will be printed. Default is false. 
*   The printed values refer to the final size only if tmax is chosen large enough.
* @returns Any IO errors that occur when saving the results.
*/
mio::IOResult<void> simulate(ScalarType Reff, ScalarType tReff, ScalarType tmax, std::string save_dir = "",
                             bool save_subcompartments = false, ScalarType scale_TimeExposed = 1.,
                             bool print_final_size = false)
{
    using namespace params;
    std::cout << "Simulation with " << num_subcompartments << " subcompartments and reproduction number " << Reff
              << " from day " << tReff << " on." << std::endl;

    // Initialize LCT model.
    using Model = mio::lsecir::Model<LctState>;
    Model model;

    // Set parameters.
    // Scale TimeExposed for some numerical experiments.
    timeExposed                                                              = scale_TimeExposed * timeExposed;
    model.parameters.get<mio::lsecir::TimeExposed>()[0]                      = timeExposed;
    model.parameters.get<mio::lsecir::TimeInfectedNoSymptoms>()[0]           = timeInfectedNoSymptoms;
    model.parameters.get<mio::lsecir::TimeInfectedSymptoms>()[0]             = timeInfectedSymptoms;
    model.parameters.get<mio::lsecir::TimeInfectedSevere>()[0]               = timeInfectedSevere;
    model.parameters.get<mio::lsecir::TimeInfectedCritical>()[0]             = timeInfectedCritical;
    model.parameters.get<mio::lsecir::TransmissionProbabilityOnContact>()[0] = transmissionProbabilityOnContact;

    model.parameters.get<mio::lsecir::RelativeTransmissionNoSymptoms>()[0] = relativeTransmissionNoSymptoms;
    model.parameters.get<mio::lsecir::RiskOfInfectionFromSymptomatic>()[0] = riskOfInfectionFromSymptomatic;

    model.parameters.get<mio::lsecir::RecoveredPerInfectedNoSymptoms>()[0] = recoveredPerInfectedNoSymptoms;
    model.parameters.get<mio::lsecir::SeverePerInfectedSymptoms>()[0]      = severePerInfectedSymptoms;
    model.parameters.get<mio::lsecir::CriticalPerSevere>()[0]              = criticalPerSevere;
    model.parameters.get<mio::lsecir::DeathsPerCritical>()[0]              = deathsPerCritical;

    // Determine the contact rate such that the effective reproduction number is initially approximately equal to one.
    ScalarType contacts_R1 =
        1. / (transmissionProbabilityOnContact *
              (timeInfectedNoSymptoms * relativeTransmissionNoSymptoms +
               (1 - recoveredPerInfectedNoSymptoms) * timeInfectedSymptoms * riskOfInfectionFromSymptomatic));
    std::cout << "Initial contacts: " << contacts_R1 << std::endl;

    mio::ContactMatrixGroup contact_matrix = mio::ContactMatrixGroup(1, 1);
    if (Reff <= 1.) {
        // Perform a simulation with a decrease in the effective reproduction number on day 2.
        contact_matrix[0] = mio::ContactMatrix(Eigen::MatrixXd::Constant(1, 1, contacts_R1));
        // This is necessary as otherwise, the damping is implemented over a time span instead of a break.
        contact_matrix[0].add_damping(0., mio::SimulationTime(tReff - 0.1));
        contact_matrix[0].add_damping(Reff, mio::SimulationTime(tReff));
    }
    else {
        // Perform a simulation with an increase in the effective reproduction number on day 2.
        contact_matrix[0] = mio::ContactMatrix(Eigen::MatrixXd::Constant(1, 1, Reff * contacts_R1));
        contact_matrix[0].add_damping(1 - 1. / Reff, mio::SimulationTime(-1.));
        contact_matrix[0].add_damping(1 - 1. / Reff, mio::SimulationTime(tReff - 0.1));
        contact_matrix[0].add_damping(0., mio::SimulationTime(tReff));
    }

    model.parameters.get<mio::lsecir::ContactPatterns>() = mio::UncertainContactMatrix<ScalarType>(contact_matrix);
    model.parameters.get<mio::lsecir::Seasonality>()     = seasonality;

    // Set initial values.
    std::vector<ScalarType> initial_values = get_initial_values(tReff);
    for (size_t i = 0; i < model.populations.get_num_compartments(); i++) {
        model.populations[i] = initial_values[i];
    }

    // Set integrator of fifth order with fixed step size and perform simulation.
    auto integrator =
        std::make_shared<mio::ControlledStepperWrapper<ScalarType, boost::numeric::odeint::runge_kutta_cash_karp54>>();
    // Choose dt_min = dt_max to get a fixed step size.
    integrator->set_dt_min(dt);
    integrator->set_dt_max(dt);
    mio::TimeSeries<ScalarType> result = mio::simulate<ScalarType, Model>(0, tmax, dt, model, integrator);

    // Save results and print desired information.
    if (!save_dir.empty()) {
        std::string Reffstring  = std::to_string(Reff);
        std::string tReffstring = std::to_string(tReff);
        std::string filename    = save_dir + "lct_Reff" + Reffstring.substr(0, Reffstring.find(".") + 2) + "_t" +
                               tReffstring.substr(0, tReffstring.find(".") + 2) + "_subcomp" +
                               std::to_string(num_subcompartments);
        if (save_subcompartments) {
            filename = filename + "_subcompartments.h5";
            // Just store daily results in this case.
            auto result_interpolated   = mio::interpolate_simulation_result(result, dt / 2);
            mio::IOResult<void> status = mio::save_result({result_interpolated}, {0}, 1, filename);
            if (!status) {
                return status;
            }
        }
        else {
            // Calculate result without division in subcompartments.
            mio::TimeSeries<ScalarType> populations = model.calculate_compartments(result);
            filename                                = filename + ".h5";
            mio::IOResult<void> status              = mio::save_result({populations}, {0}, 1, filename);
            if (!status) {
                return status;
            }
        }
    }
    if (print_final_size) {
        std::cout << "Final size: " << std::fixed << std::setprecision(6)
                  << total_population - result.get_last_value()[0] << std::endl;
        std::cout << std::endl;
    }
    return mio::success();
}

/** 
* Usage: lct_impact_distribution_assumption <Reff> <tReff> <tmax> <save_dir> <save_subcompartments> <scale_TimeExposed> 
*           <print_final_size>
*   All command line arguments are optional. Simple default values are provided if not specified.
*   All parameters are passed to the simulation() function. See the documentation for a description of the parameters.
*
*   The numbers of subcompartments used in the LCT model is determined by the preprocessor macro NUM_SUBCOMPARTMENTS.
*   You can set the number via the flag -DNUM_SUBCOMPARTMENTS=... . 
*/
int main(int argc, char** argv)
{
    ScalarType Reff              = 1.;
    ScalarType tReff             = 0.;
    ScalarType tmax              = 40;
    std::string save_dir         = "";
    bool save_subcompartments    = false;
    ScalarType scale_TimeExposed = 1.;
    bool print_final_size        = false;

    switch (argc) {
    case 8:
        print_final_size = std::stoi(argv[7]);
        [[fallthrough]];
    case 7:
        scale_TimeExposed = std::stod(argv[6]);
        [[fallthrough]];
    case 6:
        save_subcompartments = std::stoi(argv[5]);
        [[fallthrough]];
    case 5:
        save_dir = argv[4];
        [[fallthrough]];
    case 4:
        tmax = std::stod(argv[3]);
        [[fallthrough]];
    case 3:
        tReff = std::stod(argv[2]);
    case 2:
        Reff = std::stod(argv[1]);
    }
    auto result = simulate(Reff, tReff, tmax, save_dir, save_subcompartments, scale_TimeExposed, print_final_size);
    if (!result) {
        printf("%s\n", result.error().formatted_message().c_str());
        return -1;
    }

    return 0;
}
