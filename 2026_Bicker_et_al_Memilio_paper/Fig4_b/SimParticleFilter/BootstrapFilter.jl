mutable struct BootstrapFilterCache{T_X,
    T1 <: AbstractFloat,
    INTEGRATOR <: DiffEqBase.DEIntegrator,
} <: ParticleFilterCache{T_X}

    x::T_X
    w_unnormalised::Vector{T1} # unnormalised particle filter weights
    w_normalised::Vector{T1} # Normalised particle filter weights
    particles::Matrix{T1} # Current value for particles (dim_x × n_particles)
    particles_tmp::Matrix{T1} # Convenience field to store particles temporarily in a memor efiicient manner
    index_resample::Vector{UInt32}
    integrator::INTEGRATOR

    # inner constructor
    function BootstrapFilterCache(x::T_X, w_unnormalised::Vector{T1}, w_normalised::Vector{T1}, particles::Matrix{T1}, particles_tmp::Matrix{T1}, index_resample::Vector{UInt32}, integrator::DiffEqBase.DEIntegrator) where {T_X, T1}
        if size(particles)[1] != size(x)[1]
            throw(ArgumentError("Number of states in particles should be the same as in x"))
        end
        if size(particles)[2] != length(w_unnormalised)
            throw(ArgumentError("Number of particles should be the same as in w_unnormalised"))
        end
        if size(particles)[2] != length(w_normalised)
            throw(ArgumentError("Number of particles should be the same as in w_normalised"))
        end
        if length(index_resample) != length(w_unnormalised)
            throw(ArgumentError("Number of particles should be the same as in index_resample"))
        end
        return new{T_X, T1, typeof(integrator)}(
            x, w_unnormalised, w_normalised, particles, particles_tmp, index_resample, integrator)
    end
end


"""
    BootstrapFilter{T1, T2, MODEL, CACHE, SCHEME}(model, observations, n_particles, cache, resampling_scheme)


Constructor for bootstrap filter for any StateSpaceModel. Without correlation of Particles

!!! note
The step-length and number of particles should be set as low as possible to reduce run-time.
"""
mutable struct BootstrapFilter{T_X,
    T_Y,
    F1 <: Function,
    PROB <: Union{DiffEqBase.SDEProblem, DiffEqBase.AbstractJumpProblem},
    ALG <: DiffEqBase.DEAlgorithm,
    CACHE <: Union{Nothing, ParticleFilterCache{T_X}},
    SCHEME <: Union{Nothing, ResamplingAlg}
} <: ParticleFilter{T_X}

    prob::PROB
    solve_alg::ALG
    solve_kwargs::Union{Nothing, NamedTuple}
    calc_log_obs_prob::F1 # take x, t, y and return the likelihood of the observation
    tobs::Vector{<:Real}
    observations::Vector{T_Y}
    n_particles::Int64
    cache::CACHE
    resampling_scheme::SCHEME



    # inner constructor
    function BootstrapFilter(prob::Union{DiffEqBase.SDEProblem, DiffEqBase.AbstractJumpProblem}, solve_alg::DiffEqBase.DEAlgorithm, solve_kwargs::Union{Nothing, NamedTuple}, calc_log_obs_prob::Function, tobs::Vector, observations::Vector{T_Y}, n_particles::Int64, cache, resampling_scheme::Union{Nothing, ResamplingAlg}) where{T_Y}
        if n_particles < 1
            throw(ArgumentError("Number of particles should be at least be 1"))
        end
        if length(observations) != length(tobs)
            throw(ArgumentError("Number of observations must be the same as number of observation times."))
        end
        if isnothing(resampling_scheme)
            resampling_scheme = SystematicResampling()
        end
        if isnothing(solve_kwargs)
            solve_kwargs = (maxiters = 1e6, dt=1e-3)
        end
        if isnothing(cache)
            dim_states = size(prob.u0)[1]
            x = Vector{Float64}(undef, dim_states)
            w_unormalised = Vector{Float64}(undef, n_particles)
            w_normalised = Vector{Float64}(undef, n_particles)
            particles = Matrix{Float64}(undef, dim_states, n_particles)
            particles_sq = Matrix{Float64}(undef, dim_states, n_particles)
            index_resample = Vector{UInt32}(undef, n_particles)
            integrator = init(prob, solve_alg; save_everystep=false, solve_kwargs...)
            cache =  BootstrapFilterCache(x, w_unormalised, w_normalised, particles, particles_sq, index_resample, integrator)
        end
        T_X = typeof(cache.x)
        return new{T_X, T_Y, typeof(calc_log_obs_prob), typeof(prob), typeof(solve_alg), typeof(cache), typeof(resampling_scheme)}(
            prob, solve_alg, solve_kwargs, calc_log_obs_prob, tobs, observations, n_particles, cache, resampling_scheme
        )
    end
end


# ToDo some more outer constructors for the BootstrapFilter and a reset method for new data, x0...!!

"""
constructor without cache
"""
function BootstrapFilter(prob::Union{DiffEqBase.SDEProblem, DiffEqBase.AbstractJumpProblem}, solve_alg::DiffEqBase.DEAlgorithm, solve_kwargs::Union{Nothing, NamedTuple}, calc_log_obs_prob::Function, tobs::Vector, observations::Vector, n_particles::Int64, resampling_scheme::Union{Nothing, ResamplingAlg})
    return BootstrapFilter(prob, solve_alg, solve_kwargs, calc_log_obs_prob, tobs, observations, n_particles, nothing, resampling_scheme)
end

"""
constructor whithout solver arguments
"""
function BootstrapFilter(prob::Union{DiffEqBase.SDEProblem, DiffEqBase.AbstractJumpProblem}, solve_alg::DiffEqBase.DEAlgorithm, calc_log_obs_prob::Function, tobs::Vector, observations::Vector, n_particles::Int64, resampling_scheme::Union{Nothing, ResamplingAlg})

    return BootstrapFilter(prob, solve_alg, nothing, calc_log_obs_prob, tobs, observations, n_particles, nothing, resampling_scheme)
end

"""
constructor where we infer problem type and choose default algorithms
"""
function BootstrapFilter(prob::Union{DiffEqBase.SDEProblem, DiffEqBase.AbstractJumpProblem}, calc_log_obs_prob::Function, tobs::Vector, observations::Vector, n_particles::Int64, resampling_scheme::Union{Nothing, ResamplingAlg})
    if typeof(prob) <: DiffEqBase.SDEProblem
        solve_alg = EM()
    elseif typeof(prob) <: DiffEqBase.AbstractJumpProblem
        solve_alg = SSAStepper()
    end
    return BootstrapFilter(prob, solve_alg, nothing, calc_log_obs_prob, tobs, observations, n_particles, nothing, resampling_scheme)
end

"""
Minimal constructor where we also omit the resampling scheme to be given.
"""
function BootstrapFilter(prob::Union{DiffEqBase.SDEProblem, DiffEqBase.AbstractJumpProblem}, calc_log_obs_prob::Function, tobs::Vector, observations::Vector, n_particles::Int64)
    return BootstrapFilter(prob, calc_log_obs_prob, tobs, observations, n_particles, nothing)
end


"""
Function to reset the filter with new data
"""
function reset_filter!(pf::BootstrapFilter, tobs::Vector{Float64}, observations::Vector, n_particles::Int64)
    new_cache = create_cache(BootstrapFilter, Val(size(pf.prob.u0)[1]), Val(length(tobs)))
    pf.tobs = tobs
    pf.observations = observations
    pf.n_particles = n_particles
    pf.cache = new_cache
    return nothing
end


"""
Function to reset the cache of the filter.
"""
function reset_cache!(filter::BootstrapFilter)
    dim_states = size(filter.prob.u0)[1]
    x = Vector{Float64}(undef, dim_states)
    w_unormalised = Vector{Float64}(undef, filter.n_particles)
    w_normalised = Vector{Float64}(undef, filter.n_particles)
    particles = Matrix{Float64}(undef, dim_states, filter.n_particles)
    particles_sq = Matrix{Float64}(undef, dim_states, filter.n_particles)
    index_resample = Vector{UInt32}(undef, filter.n_particles)
    integrator = init(filter.prob, filter.solve_alg; save_everystep=false, filter.solve_kwargs...)
    filter.cache = BootstrapFilterCache(x, w_unormalised, w_normalised, particles, particles_sq, index_resample, integrator)
    return nothing
end


"""
Function to simulate a trajectory of the model.
"""
function simulate_trajectory!(pf)
    sol = solve(pf.prob, pf.solve_alg, pf.solve_kwargs...)
    return sol
end

### Functions related to filtering
"""
Function to simulate forward the model.
"""
function simulate_forward!(integrator, xp, t, dt)
    # TODO: reinitialize integrator otherwise the following error will occur.
    # if integrator.u != xp || integrator.t != t
    #     throw(DomainError("The integrator is not at the correct state or timepoint."))
    # end

    # reinitialize integrator
    reinit!(integrator, xp, t0=t)

    # simulate forward
    step!(integrator, dt, true)
    if check_error(integrator) != ReturnCode.Success
        throw(error("Error in simulating forward the model."))
    end
    return nothing
end


"""
Function to run one step to next timepoint, for one particle
"""
function step_filter!(pf::BootstrapFilter,
    t::Float64,
    dt::Float64
    )
    # simulate the particles forward
    simulate_forward!(pf.cache.integrator, pf.cache.x, t, dt)
    pf.cache.x .= pf.cache.integrator.u
end


"""
function to propagate all particles to next timepoint
"""
function propagate_filter!(pf::BootstrapFilter, t::Float64, dt::Float64)
    # simulate all particles forward to t+dt
    @inbounds for i in 1:pf.n_particles
        # copy the particles as current state
        @views copyto!(pf.cache.x, pf.cache.particles[:,i])
        # simulate forward
        step_filter!(pf, t, dt)
        # copy the particles back
        pf.cache.particles[:,i] .= pf.cache.x
    end
end


"""
Run one instance of the particle filter (offlinefiltering) for given data.
"""
function run_filter(pf::BootstrapFilter,
    )
    # initialise the filter
    n_particles::Int64 = pf.n_particles
    t_vec::Vector{Float64} = pf.tobs
    n_obs::Int64 = length(t_vec)
    y_mat::Matrix{Float64} = reduce(hcat, pf.observations) # get from vector of vectors to matrix with dimension (dim_y x n_obs)

    reset_cache!(pf) # also resets the integrator

    log_lik::Float64 = 0.0

    # initialise the particles
    @inbounds for i in 1:n_particles
        pf.cache.particles[:,i] .= pf.prob.u0
    end

    # Propagate if t = 0 is not observed
    if t_vec[1] != 0.0
        try
            propagate_filter!(pf, 0.0, t_vec[1])
        catch
            @debug "Error in propagating the particles from timepoint 0 to $(t_vec[1])."
            return -Inf
        end
    end

    # Update likelihood first time
    max_w_unnormalised, sum_w_unnormalised = calc_weights!(pf, 1, t_vec, y_mat, n_particles)
    log_lik += max_w_unnormalised + log(sum_w_unnormalised * 1/n_particles)

    # Go over remaining time-steps
    for t_idx in 2:1:n_obs
        # capture if all weights are 0 and return -Inf
        if sum_w_unnormalised == 0.0
            @debug "All weights are zero for SDE parameters $(pf.prob.p)."
            return -Inf
        end
        # check if nans are present in the weights, if it happens, then this needs debugging.
        if any(isnan.(pf.cache.w_normalised))
            @debug "NaNs in the weights for unnormalised $(pf.cache.w_unnormalised), normalised $(pf.cache.w_normalised) and parameter $(pf.prob.p)."
            return -Inf
        end
        # Resample
        resample!(TaskLocalRNG(), pf.cache.index_resample, pf.resampling_scheme, pf.cache.w_normalised, nothing)
        # Store resampled particles in memory efficient manner temporarily in particles_tmp
        pf.cache.particles_tmp .= @view pf.cache.particles[:, pf.cache.index_resample]
        copyto!(pf.cache.particles, pf.cache.particles_tmp)

        # Propagate forward resampled particles
        try
            propagate_filter!(pf, t_vec[t_idx-1], t_vec[t_idx] - t_vec[t_idx-1])
        catch
            @debug "Error in propagation from timepoint  $(t_vec[t_idx-1]) to  $(t_vec[t_idx])."
            return -Inf
        end

        # Update weights and likelihood
        try
            max_w_unnormalised, sum_w_unnormalised = calc_weights!(pf, 1, t_vec, y_mat, n_particles)
        catch e 
            if e isa ValueError
                @debug "Value error in calculating the weights for parameters $(pf.prob.p)."
                return -Inf
            else
                rethrow(e)
            end
        end
        max_w_unnormalised, sum_w_unnormalised = calc_weights!(pf, t_idx, t_vec, y_mat, n_particles)
        log_mean_weight = max_w_unnormalised + log(sum_w_unnormalised * 1/n_particles)
        log_lik += log_mean_weight
    end

    return log_lik
end


"""
Compute the weights of the particles at a specific timepoint.
"""

# new calculate weight function with different weight transformation for numerical stability
function calc_weights!(pf::BootstrapFilter, t_idx::Int64, t_vec::Vector{Float64}, y_mat::Matrix{Float64}, n_particles::Int64)
    t = t_vec[t_idx]
    y_obs_sub = @view y_mat[:, t_idx] # get the observation at time t
    log_w_unnormalised = zeros(n_particles)
    for i in 1:n_particles
        x = @view pf.cache.particles[:, i]
        # calculate the log-weights based on the log observation probability
        log_w_unnormalised[i] = pf.calc_log_obs_prob(x, y_obs_sub, t)
    end
    max_w = maximum(log_w_unnormalised)
    for i in 1:n_particles
        pf.cache.w_unnormalised[i] = exp(log_w_unnormalised[i] - max_w)
    end
    sum_w_unnormalised = sum(pf.cache.w_unnormalised)
    pf.cache.w_normalised .= pf.cache.w_unnormalised ./ sum_w_unnormalised

    return max_w, sum_w_unnormalised
end
