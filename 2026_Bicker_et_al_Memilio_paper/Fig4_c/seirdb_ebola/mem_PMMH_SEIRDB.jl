using Libdl

# 1) Preload Julia's OpenSSL (3.3) globally
import OpenSSL_jll
Libdl.dlopen(OpenSSL_jll.libcrypto_path, Libdl.RTLD_GLOBAL)
Libdl.dlopen(OpenSSL_jll.libssl_path,    Libdl.RTLD_GLOBAL)

# 2) Now preload an OpenSSL-flavored libcurl to satisfy Python/HDF5
# Prefer the libcurl that belongs to the Python you use with PyCall.
# If you're using a conda env:
#    curlpath = joinpath(ENV["CONDA_PREFIX"], "lib", "libcurl.so.4")
# If you're using system Python, adjust as needed:
# curlpath = "/usr/lib/x86_64-linux-gnu/libcurl.so.4"  # <- adjust if different
# Libdl.dlopen(curlpath, Libdl.RTLD_GLOBAL)



# # Then everything corresponding to PyCall including adjust opneSSL paths
# using Libdl
# Libdl.dlopen("/usr/lib/x86_64-linux-gnu/libcurl.so.4", Libdl.RTLD_GLOBAL)  # adjust path if needed


# task_id_str = get(ENV, "SLURM_ARRAY_TASK_ID", 0)
# task_id = parse(Int64, task_id_str)

#---------------------------------------------------------
# # use correct python version for PyCall
# using Pkg 
# ENV["PYTHON"] = "/home/vincent/memilio_pf/ebola_seirdb/memilio_ebola_env/bin/python3"
# Pkg.build("PyCall")


# using PyCall



# py"""
# import sys
# sys.path.insert(0, "/home/vincent/memilio_pf/ebola_seirdb")
# """

# use correct python version for PyCall
using Pkg 
ENV["PYTHON"] = "/home/vincent/PhD/Projects/Memilio_PF/seirdb_ebola/seirdb_env/bin/python"
Pkg.build("PyCall")


using PyCall



py"""
import sys
sys.path.insert(0, "/home/vincent/PhD/Projects/Memilio_PF/seirdb_ebola")
"""

# Then all generic julia packages, note nothing that needs different OpenSSL works!!! So no Plots or HDF5 for example.
using MCMCChains
using MCMCDiagnosticTools
using AdvancedMH
using Distributions
using Random
using LinearAlgebra
using LoopVectorization
using DataFrames
using CSV
using Printf
using HDF5
using MCMCChainsStorage
using StatsPlots

using Base.Threads


# Now the local package that needs the PyObject type from PyCall.
include(joinpath(@__DIR__, "../SimParticleFilter", "ParticleFilter_Sim.jl"))
include(joinpath(@__DIR__, "../SimParticleFilter", "PythonSimulator.jl"))
include(joinpath(@__DIR__, "../SimParticleFilter", "ResamplingAlgorithms.jl"))
include(joinpath(@__DIR__, "../SimParticleFilter", "BootstrapFilter_Sim.jl"))
include(joinpath(@__DIR__, "../SimParticleFilter", "Utilities_Sim.jl"))

simulation_function = pyimport("seirdb")["seirdb_flow"]

# set default start parameters and initial conditions
p = [0.0055, 5.0, 4.0, 0.53, 0.025, 2.0]
N = 10000000.0
E0 = 0.0
I0 = 10.0
R0 = 0.0
D0 = 0.0
B0 = 0.0
S0 = N-(E0+I0+R0+D0+B0)
t_initial = 0.0
u_initial = [0.0, 0.0, 0.0, 0.0, 0.0]  # Initial flow values
x_initial = [S0, E0, I0, R0, D0, B0]  # Example initial state vector S, E, I, R, D, B

# initialize the simulator
seirdb_simulator = MemFlowSEIRSimulator(simulation_function, p, t_initial, u_initial, x_initial)

# get data
data = CSV.read(joinpath(@__DIR__, "data", "cut_early_guinea_conf_cases.csv"), DataFrame)
tobs = data.Date
observations = data.new_cases

# define the log-likelihood function based on the BootstrapFilter

nparticles = 10 # only ODE so not many needed
dispersion = 5.0 # not estimated


function nb_log_obs_prob(x, y, t) # needs to be able to take missing values
    case_flow = x[2]  # flow value for E -> I
    prob_cases = min(0.99999, dispersion / (dispersion + case_flow))
    if (y === missing) || (ismissing(y[1])) 
        return 0.0 # log(1)
    end
    # y = convert(Vector{Float64}, y) # convert from Union{Float64, Missing} to Float64
    y = collect(skipmissing(y))
    return sum(logpdf.(NegativeBinomial(dispersion, prob_cases), y))
end

function log_likelihood(θ)
    
    # Update the parameters in the simulator
    seirdb_simulator = MemFlowSEIRSimulator(simulation_function, θ, t_initial, u_initial, x_initial)

    # Initialize the BootstrapFilter
    bf = BootstrapFilter(
        seirdb_simulator, nb_log_obs_prob, tobs, observations, nparticles
    )

    # Run the filter
    log_likeli = run_filter(bf)

    # Return the log-likelihood
    return log_likeli
end

# denstiy model
insupport(θ) = θ[1] > 0 && θ[2] > 0 && θ[3] > 0
density(θ) = insupport(θ) ? log_likelihood(θ) : -Inf

pypesto = pyimport("pypesto")

# convert PyPesto result to MCMCChains.jl chain type
function Chains_from_pypesto(result; kwargs...)
    trace_x = result.sample_result["trace_x"] # parameter values
    trace_neglogp = result.sample_result["trace_neglogpost"] # posterior values
    samples = Array{Float64}(undef, size(trace_x, 2), size(trace_x, 3) + 1, size(trace_x, 1))
    samples[:, begin:end-1, :] .= PermutedDimsArray(trace_x, (2, 3, 1))
    samples[:, end, :] = .-PermutedDimsArray(trace_neglogp, (2, 1))
    param_names = Symbol.(result.problem.x_names)
    chain = Chains(
        samples,
        vcat(param_names, :lp),
        (parameters = param_names, internals = [:lp]);
        kwargs...
    )
    return chain
end

neg_llh(θ) = -density(θ)

# transform to pypesto objective
objective = pypesto.Objective(fun=neg_llh)

# create pypesto problem
pypesto_problem = pypesto.Problem(
    objective,
    x_names=["rhoI", "TE", "TI", "pR", "rhoD", "TB"],
    lb=[0.0001, 1.0, 1.0, 0.01, 0.0001, 1.0], # parameter bounds
    ub=[0.5, 30.0, 50.0, 0.99, 0.5, 20.0], # NB for sampling it is usually better if you remap parameters to (-∞, ∞)
    copy_objective=false, # important
)

# specify sampler
pypesto_sampler = pypesto.sample.AdaptiveMetropolisSampler();

nsamples = 100
x0 = Vector([0.007, 7, 7.0, 0.53, 0.025, 2.0])

function chain()
    result = pypesto.sample.sample(
                        pypesto_problem,
                        n_samples=nsamples,
                        x0=x0, # starting point
                        sampler=pypesto_sampler,
                        )
    return Chains_from_pypesto(result)
end

nchains = 1 # Number of chains to run, one per worker

chains_list = [chain() for i in 1:nchains]

chain_values = chains_list[1].value.data

# get the chains
for j in 2:nchains
    global chain_values
    chain_values = cat(chain_values, chains_list[j].value.data, dims=(3,3))
end

chs = MCMCChains.Chains(chain_values, [:rI, :TE, :TI, :pR, :rD, :TB, :lp])
complete_chain = set_section(chs, Dict(:parameters => [:rI, :TE, :TI, :pR, :rD, :TB], :internals => [:lp]))

h5open("./output/seirdb_early_flow_$(dispersion)disp_10p_$(task_id)ch_$(nsamples)s.h5", "w") do f
    write(f, complete_chain)
end

plot(complete_chain)
# Save the plot
savefig("./figures/seirdb_early_flow_$(dispersion)disp_10p_$(task_id)ch_$(nsamples)s.png")
