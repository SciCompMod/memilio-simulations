# calculate optimal sizes in pixels for latex documents textwidth
dpi = 100 # use the Plots.jl default
width_pts = 455.244
inches_per_points = 1.0/72.27
width_inches = width_pts*inches_per_points
width_px = width_inches*dpi
# note that default in julia is (600,400)

function plot_llh_vs_nparticles(prob, calc_obs_prob, tobs, observations, nparticles::AbstractVector{<:Integer}; nruns::Integer=100)
    x = Vector{String}(undef, length(nparticles) * nruns)
    y = Vector{Float64}(undef, length(nparticles) * nruns)
    variances = Vector{Float64}(undef, length(nparticles))
    k = 1
    i = 1
    for n in nparticles
        bf = DiffEqParticleFilters.BootstrapFilter(prob, calc_obs_prob, tobs, observations, n)
        for _ in 1:nruns
            @inbounds x[k] = string(convert(Int, n))
            @inbounds y[k] = DiffEqParticleFilters.run_filter(bf)
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

function my_plot_filter(ssm, parameters, components::AbstractVector{<:Integer}, 
                        tobs; nparticles::Integer, nsigmas::Real=2)
    
    length(components) > length(colors) && error("too many components, not enough colors")
    ncomp = length(components)
    hidden, obs=rand(ssm, parameters, length(tobs));
    bf = BootstrapFilter(ssm, obs)
    pf = SMC(
        bf, parameters, nparticles,
        (filter=RunningSummary(MeanAndVariance(), FullHistory()), ),
    )
    offlinefilter!(pf);
    hist = pf.history_run.filter
    comp_means = Vector{Vector{Float64}}(undef, ncomp)
    comp_variances = Vector{Vector{Float64}}(undef, ncomp)
    for i in 1:ncomp
        mean = [hist[j].mean[i] for j in 1:length(tobs)]
        variance = [hist[j].var[i] for j in 1:length(tobs)]
        comp_means[i] = mean
        comp_variances[i] = variance
    end
    plt = Plots.plot(size=(width_px/2,width_px/3))
    x = 1:length(tobs)
    for i in components
        scatter!(plt, x, [hidden[j][i] for j in 1:length(tobs)], color=colors[i], label="$i component")
        plot!(plt, x, comp_means[i], color=:blue, label="")
        plot!(plt, x, comp_means[i]+nsigmas*sqrt.(comp_variances[i]),
            fillrange=comp_means[i]-nsigmas*sqrt.(comp_variances[i]),
            alpha=0.35, color=colors[i], label="") 
    end
    return plt
end;