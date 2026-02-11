
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


# if used  on a cluster within a slurm array
task_id_str = get(ENV, "SLURM_ARRAY_TASK_ID", 0)
task_id = parse(Int64, task_id_str)

# if used locally
# task_id = 1
#---------------------------------------------------------
# use correct python version for PyCall
using Pkg 
ENV["PYTHON"] = "/home/vincent/memilio_pf/influenza_ssirs/memilio_influenza_env/bin/python3"
Pkg.build("PyCall")


using PyCall


py"""
import sys
sys.path.insert(0, "/home/vincent/memilio_pf/influenza_ssirs")
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


simulation_function = pyimport("mem_ssirs")["ssirs_model"]

# p in ordering as TransmissionProbabilityOnContact, TimeInfected, TimeImmune, season_influence, seasonpeak day
p = [0.1, 7, 16, 0.28235, 0.35, 0.23]
sigma = 0.5 # noise parameter for normal noise, we might estimate later.
N0 = 100000.0
I0 = 3280
R0 = 0.0
# R0 = 10000.0 # for synthetic data example
t_initial = 0.0
# u0 in the ordering [S, I, R]
u_initial = [N0-I0-R0, I0, R0]

ssirs_simulator = MemSSIRSSimulator(simulation_function, p, t_initial, u_initial)


# get observation function
function calc_log_obs_prob(x, y, t)
    y = collect(skipmissing(y))
    return logpdf(Normal(x[2], x[2]*sigma), y)[1]
end

# get data
year = "2016_2019"
data = CSV.read(joinpath(@__DIR__, "data", "ILI_germany_$(year).csv"), DataFrame)
tobs = data.time.*7 # make sure that time is in weeks in data sucht that this transforms to days.
observations = data.Inzidenz_sum

# define the log-likelihood function based on the BootstrapFilter

nparticles = 50

function log_likelihood(θ)
    # Update the parameters in the simulator
    ssirs_simulator = MemSSIRSSimulator(simulation_function, θ, t_initial, u_initial)

    # Initialize the BootstrapFilter
    bf = BootstrapFilter(
        ssirs_simulator, calc_log_obs_prob, tobs, observations, nparticles
    )

    # Run the filter
    log_likeli = run_filter(bf)

    # Return the log-likelihood
    return log_likeli
end

bf = BootstrapFilter(
    ssirs_simulator, calc_log_obs_prob, tobs, observations, nparticles
    )

insupport(θ) = θ[1] > 0 && θ[2] > 0
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
    x_names=["beta", "tinfc", "timm", "season_1", "season_2", "season_3"], # parameter names
    lb=[0.01, 1.0, 1.0, 0.1, 0.1, 0.1], # parameter bounds
    ub=[0.5, 14, 50, 0.5, 0.5, 0.5], # NB for sampling it is usually better if you remap parameters to (-∞, ∞)
    copy_objective=false, # important
)

# specify sampler
pypesto_sampler = pypesto.sample.AdaptiveMetropolisSampler();

nsamples = 100000

x0 = Vector([0.05, 7, 14, 0.2, 0.2, 0.2])

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

chs = MCMCChains.Chains(chain_values, [:beta, :tinfc, :timm, :season1, :season2, :season3, :lp])
complete_chain = set_section(chs, Dict(:parameters => [:beta, :tinfc, :timm, :season1, :season2, :season3], :internals => [:lp]))

h5open("./output/SSIRS_joined_$(sigma)sig_pypesto_$(task_id)ch_$(nparticles)p_$(nsamples)s.h5", "w") do f
    write(f, complete_chain)
end

plot(complete_chain)
# Save the plot
savefig("./output/pypesto_joined_$(sigma)sig_$(task_id)ch_$(nparticles)p_$(nsamples)s.png")
