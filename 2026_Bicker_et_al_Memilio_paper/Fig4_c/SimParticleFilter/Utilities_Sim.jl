#= 
    Common functions used by all filters. 
    Remake filter object
    Create ind-data struct 
    Create and update random numbers 
    Create model-parameters struct 
    Init filter struct 
    Change option for filter struct 
=# 


# ToDo create filter with different type for x, so somehow infer it from the model in the filter?
function create_cache(filter::BootstrapFilter, ::Val{dim_states}, ::Val{dim_obs}) where {dim_states, dim_obs}
    
    x = Vector{Float64}(undef, dim_states)
    w_unormalised = Vector{Float64}(undef, filter.n_particles)
    w_normalised = Vector{Float64}(undef, filter.n_particles)
    particles = Matrix{Float64}(undef, dim_states, filter.n_particles)
    particles_sq = Matrix{Float64}(undef, dim_states, filter.n_particles)
    index_resample = Vector{UInt32}(undef, filter.n_particles)
    
    return BootstrapFilterCache(x, w_unormalised, w_normalised, particles, particles_sq, index_resample)
end
