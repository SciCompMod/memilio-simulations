# calculate optimal sizes in pixels for latex documents textwidth
dpi = 100 # use the Plots.jl default
width_pts = 455.244
inches_per_points = 1.0/72.27
width_inches = width_pts*inches_per_points
width_px = width_inches*dpi
# note that default in julia is (600,400)

function plot_llh_vs_nparticles(simulator, calc_obs_prob, tobs, observations, nparticles::AbstractVector{<:Integer}; nruns::Integer=100)
    x = Vector{String}(undef, length(nparticles) * nruns)
    y = Vector{Float64}(undef, length(nparticles) * nruns)
    variances = Vector{Float64}(undef, length(nparticles))
    k = 1
    i = 1
    for n in nparticles
        bf = BootstrapFilter(simulator, calc_obs_prob, tobs, observations, n)
        for _ in 1:nruns
            @inbounds x[k] = string(convert(Int, n))
            @inbounds y[k] = run_filter(bf)
            k += 1
        end
        variances[i] = round(var(y[((i-1)*nruns+1):i*nruns]); digits=3)
        i += 1
    end
    y = reshape(y, (nruns,length(nparticles)))
    x_names = reshape(string.(nparticles), (1,length(nparticles)))
    labels_list = ["$(nparticles[j])p var: $(variances[j])" for j in eachindex(nparticles)] 
    labels = reshape(labels_list, (1,length(nparticles)))
    return Plots.violin(x_names, y, xlabel="particle number",
        trim=false, labels=labels, legend=:outerright,
        size=(width_px, width_px*2/3))
end

    
colors = palette(:Accent_8) #set some color scheme, because I like lighter colors for the following plot


# # TO BE CHANGED; NEEDS TO WORK WITH GENERIC BF FROM ME

# function unique_ancestors_at_previous_times(ancestors::AbstractVector{<:AbstractVector{Int}}; check::Bool=true)
#     # NB thanks to the fact that ancestor indices are always ordered,
#     #    we only need the history for the first and last particle
#     !check || all(issorted, ancestors) || error("ancestors do not satisfy ordering assumptions")
#     unique = Vector{Int}(undef, length(ancestors) + 1)
#     k = length(ancestors) + 1
#     t = lastindex(ancestors)::Int
#     B = collect(Int, axes(last(ancestors), 1))
#     @inbounds while true
#         unique[k] = length(Set(B))
#         t ≥ firstindex(ancestors) || break
#         B .= getindex.(Ref(ancestors[t]), B)
#         t -= 1
#         k -= 1
#     end
#     return unique
# end