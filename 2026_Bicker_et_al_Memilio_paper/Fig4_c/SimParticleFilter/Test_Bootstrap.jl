using StochasticDiffEq
using Random
using Distributions
using Plots

include(joinpath(@__DIR__, "..", "src", "DiffEqParticleFilter.jl"))
#=
    What should be the job of PEtab.jl here.
    Provide a SDEProblem with correct parameters, the simulation time-points, struct that
    for a time-point (measurement) it is possible to compute the likelihood given an
    input vector u
    Then the particle filter can just create an appropriate number of integrators, use
    the step function, and at each step reset in an appropriate manner. The resampling
    etc is just fixed by the particle filters.
    Ask on Slack if possible for me to provide random numbers to algorithms with a
    fixed step-size
=#

function f(du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = -3 * u[2] + u[1] * u[2]
end
function g(du, u, p, t)
    du[1] = p[3] * u[1]
    du[2] = p[4] * u[2]
end

p = [1.5, 1.0, 0.1, 0.1]
u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
prob = SDEProblem(f, g, u0, tspan, p)

solve_alg = LambaEM()
solve_kwargs = NamedTuple()

sol = solve(prob, solve_alg, solve_kwargs...)

plot(sol)

# get observation function
function calc_log_obs_prob(x, y, t)
    return logpdf(MvNormal(x, 0.1), y)
end

tobs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
observations = [rand(MvNormal(sol(t), 0.1)) for t in tobs]

scatter!(tobs, transpose(reduce(hcat, observations)))

# Initialize the BootstrapFilter
bf = DiffEqParticleFilter.BootstrapFilter(prob, calc_log_obs_prob, tobs, observations, 100)

log_likeli = DiffEqParticleFilter.run_filter(bf)
