# code taken from Contento, Particles.jl -> Resampling algorithms

abstract type ResamplingAlg end

make_cache(s::ResamplingAlg, nweights::Integer, nchoices::Integer) = nothing
make_cache(s::ResamplingAlg, n::Integer) = make_cache(s, n, n)

function resample!(rng::AbstractRNG, ancestors::AbstractVector{<:Integer}, scheme::ResamplingAlg, weights::Vector{<:Real})
    nweights = length(weights)
    nchoices = length(ancestors)
    cache_r = make_cache(scheme, nweights, nchoices)
    return resample!(rng, ancestors, scheme, weights, cache_r)
end

####################################################################################################

"""
inverse_categorical_cdf!(A::Vector{<:Integer}, w::Vector{<:Real}, u::Vector{<:Real})
Evaluate the inverse CDF of the categorical distribution defined by the normalized (non-negative and sum to one) weights `w` at the ordered points `u` in the unit interval `[0, 1]`.
The vectors `w` and `u` need not have the same length, but `A` must have the same length as `u` and `w` must be non-empty.
The elements of `A` will belong to the set `{1, 2, ..., length(w)}`.
"""
function inverse_categorical_cdf!(A::AbstractVector{<:Integer}, w::AbstractVector{<:Real}, u::AbstractVector{<:Real})
    isempty(w) && throw(ArgumentError("weights vector is empty"))
    sum(w) ≈ 1 || throw(ArgumentError("weights vector is not normalized."))    
    all( w .== 0) && throw(ValueError("weights vector is all zeros"))
    any(w .< 0) && throw(ArgumentError("weights vector contains negative elements"))
    length(A) == length(u) || throw(ArgumentError("A and u vectors have different lengths"))
    issorted(u) || throw(ArgumentError("u vector is not sorted"))
    any(u .< 0) && throw(ArgumentError("u vector contains negative elements"))
    any(u .> 1) && throw(ArgumentError("u vector contains elements greater than one"))

    s = @inbounds w[1]
    m = 1
    @inbounds for n in eachindex(A, u)
        while u[n] > s
            m += 1
            s += w[m]
        end
        A[n] = m
    end
    return A
end


####################################################################################################

struct SystematicResampling <: ResamplingAlg end

function resample!(rng::AbstractRNG, ancestors::AbstractVector{<:Integer}, ::SystematicResampling, weights::Vector{<:Real}, cache_r::Nothing)
    M = length(ancestors)
    spacings = (rand(rng) .+ (0:M-1)) ./ M # uniformly spaced points in [0, 1) with random start
    inverse_categorical_cdf!(ancestors, weights, spacings)
    return true
end

####################################################################################################

struct MultinomialResampling <: ResamplingAlg end

make_cache(::MultinomialResampling, nparticles::Integer, nchoices::Integer) = Vector{Float64}(undef, nchoices)

"""
    uniform_spacing!(rng::AbstractRNG, v::AbstractVector)
Generate \$N\$ ordered uniform variates (where `N = length(v)`) in \$O(N)\$ time.
Equivalent to `sort!(rand!(v))` which has \$O(N\\log(N))\$ complexity.
"""
function uniform_spacing!(rng::AbstractRNG, v::AbstractVector{T}) where {T}
    ifirst, ilast = firstindex(v), lastindex(v)
    v0 = -log(rand(rng, T))
    @turbo for i in axes(v, 1)
        v[i] = -log(rand(rng, T))
    end
    for i in ifirst+1:ilast
        v[i] += v[i-1]
    end
    vsum = @inbounds v[ilast] + v0
    @turbo @. v /= vsum
    return v
end

function resample!(rng::AbstractRNG, ancestors::AbstractVector{<:Integer}, ::MultinomialResampling, weights::Vector{<:Real}, cache_r::Vector{Float64})
    spacings = uniform_spacing!(rng, cache_r)
    inverse_categorical_cdf!(ancestors, weights, spacings)
    return true
end


####################################################################################################

## ToDo add stratified resampling or others



####################################################################################################

"""
AdaptiveResampling
Only does resampling if the ESS (effective sample size) is below a certain threshold.
The ESS is computed from the weights, and is defined as the reciprocal of the sum of the squares of the weights.
"""

function ess_of_normalized_weights(weights::Vector{<:Real})
    return inv(sum(abs2, weights))
end


struct AdaptiveResampling{T <: ResamplingAlg} <: ResamplingAlg
    scheme::T
    ESSrmin::Float64
    function AdaptiveResampling(scheme::ResamplingAlg, ESSrmin::Real=0.5)
        0 ≤ ESSrmin ≤ 1 || throw(ArgumentError("ESSrmin must be in [0, 1]"))
        return new{typeof(scheme)}(scheme, ESSrmin)
    end
end

make_cache(ar::AdaptiveResampling, nparticles::Integer) = make_cache(ar.scheme, nparticles)

function resample!(rng::AbstractRNG, ancestors::AbstractVector{<:Integer}, ascheme::AdaptiveResampling, weights::Vector{<:Real}, cache_r)
    ESS = ess_of_normalized_weights(weights)
    if ESS < ascheme.ESSrmin * length(weights)
        return resample!(rng, ancestors, ascheme.scheme, weights, cache_r)
    else
        M = length(ancestors)
        @assert length(weights) == M # This is true at the moment, but could fail if the number of particles is allowed to change between steps
        @turbo ancestors .= Base.OneTo(M)
        return false
    end
end