using MCMCChains
using Random
using StatsBase: sample, quantile
using Plots

"""
    posterior_predictive_plot(simfun,
                              chain::Chains,
                              obs_t::AbstractVector,
                              obs_y::AbstractVector;
                              n_ens::Int = 500,
                              burn_in::Int = 0,
                              quantiles = (0.1, 0.25, 0.5, 0.75, 0.9),
                              state::Union{Int,Nothing} = 1,
                              sum_states::Union{AbstractVector{Int},Nothing} = nothing,
                              rng::AbstractRNG = Random.GLOBAL_RNG,
                              plot_kwargs...)

Draw an ensemble of parameter vectors from the MCMC `chain`, run the simulator,
compute pointwise quantiles of the simulated trajectories, and plot them against
the observed data.

Assumes `Array(chain)` has size `(n_samples, n_params)`, i.e. rows are samples.

# Arguments

- `simfun(params, times)` → array:
    User-supplied simulator. It must accept a parameter NamedTuple and a vector
    of times, and return either a `states × T` or `T × states` array.

- `chain::Chains`: MCMCChains object approximating the posterior.

- `obs_t`: observation times (vector of length `T`).
- `obs_y`: observed values at `obs_t` (same length as `obs_t`).

# Keyword arguments

- `n_ens`: number of posterior samples / trajectories in ensemble.
- `burn_in`: **number of initial rows of `Array(chain)` to discard**.
- `quantiles`: collection of quantile levels in (0,1). Will be sorted.
- `state`: which state index to plot (ignored if `sum_states` is given).
- `sum_states`: vector of state indices to sum over before taking quantiles.
- `rng`: random number generator.
- `plot_kwargs...`: forwarded to the initial `plot` call (e.g. `title`, `ylabel`, etc.).

# Returns

A NamedTuple:
- `plot`     – the `Plots.Plot` object
- `times`    – `obs_t`
- `quantiles` – sorted quantile levels
- `qtraj`    – `nq × T` matrix of quantile trajectories
- `ensemble` – `n_ens × T` matrix of simulated trajectories used to compute quantiles
"""
function posterior_predictive_plot(
    simfun,
    chain::Chains,
    obs_t::AbstractVector,
    obs_y::AbstractVector;
    n_ens::Int = 500,
    burn_in::Int = 0,
    quantiles = (0.1, 0.25, 0.5, 0.75, 0.9),
    state::Union{Int,Nothing} = 1,
    sum_states::Union{AbstractVector{Int},Nothing} = nothing,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    plot_kwargs...
)

    @assert length(obs_t) == length(obs_y) "obs_t and obs_y must have the same length"
    @assert (state !== nothing) || (sum_states !== nothing) "Provide either `state` or `sum_states`"

    # Array(chain) is 2D: (n_samples, n_params)
    arr = Array(chain)
    n_samples, n_param = size(arr)

    @assert 0 ≤ burn_in < n_samples "burn_in must be between 0 and n_samples-1"
    first_idx = burn_in + 1
    n_kept = n_samples - burn_in

    # How many draws do we actually simulate?
    n_draw = min(n_ens, n_kept)


    # Sample row indices from the post–burn-in part of the flattened chain
    inds = sample(rng, first_idx:n_samples, n_draw; replace = false)

    T = length(obs_t)
    ensemble = Matrix{Float64}(undef, n_draw, T)

    for (e, k) in enumerate(inds)
        # parameter vector for this draw (as a plain Vector)
        θ_vec = Array(arr[k, :])

        # Run simulator on the observation time grid
        sim = simfun(θ_vec, obs_t)

        # interpret sim as states × time
        sim_st_t =
            if ndims(sim) == 2 && size(sim, 2) == T
                sim
            elseif ndims(sim) == 2 && size(sim, 1) == T
                permutedims(sim)  # make it states × time
            else
                throw(ArgumentError("Simulator output must be (states×T) or (T×states) with T = length(obs_t)"))
            end

        vals =
            if sum_states !== nothing
                # sum over given state indices
                vec(sum(sim_st_t[sum_states, :], dims = 1))
            else
                sim_st_t[state, :]
            end

        ensemble[e, :] = vals
    end

    # ------ Quantiles over ensemble at each time point ------
    q = sort(collect(float.(quantiles)))
    nq = length(q)
    qtraj = Array{Float64}(undef, nq, T)

    for (iq, qq) in enumerate(q)
        for j in 1:T
            qtraj[iq, j] = quantile(@view(ensemble[:, j]), qq)
        end
    end

    # ------ Plotting ------
    plt = plot(; xlabel = "time", ylabel = "value", legend = :topright, plot_kwargs...)

    # Ribbons for symmetric quantile pairs around the median
    median_pos = findfirst(x -> isapprox(x, 0.5; atol = 1e-8), q)

    # symmetric pairs: (q[1], q[end]), (q[2], q[end-1]), ...
    for i in 1:floor(Int, nq / 2)
        low  = qtraj[i, :]
        high = qtraj[nq + 1 - i, :]
        mid   = (low .+ high) ./ 2
        width = (high .- low) ./ 2
        label = "q=$(q[i]), $(q[nq + 1 - i])"
        plot!(plt, obs_t, mid; ribbon = width, fillalpha = 0.2, linealpha=0, label = label)
    end

    # Median line (if present in quantiles)
    if median_pos !== nothing
        plot!(plt, obs_t, qtraj[median_pos, :]; color = :black, lw = 2, label = "median")
    end

    # Observed data
    scatter!(plt, obs_t, obs_y; color = :black, ms = 4, label = "data")

    return (plot = plt, times = obs_t, quantiles = q, qtraj = qtraj, ensemble = ensemble)
end
