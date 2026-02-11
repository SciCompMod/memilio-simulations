using StochasticDiffEq: StochasticDiffEqAlgorithm
using StochasticDiffEq: StochasticDiffEqConstantCache, StochasticDiffEqMutableCache
using StochasticDiffEq: @unpack, @.., @muladd, is_split_step, is_diagonal_noise

struct PosEM <: StochasticDiffEqAlgorithm end

StochasticDiffEq.alg_order(alg::PosEM) = 1 // 2
StochasticDiffEq.alg_compatible(prob::StochasticDiffEq.DiffEqBase.AbstractSDEProblem, alg::PosEM) = true
StochasticDiffEq.is_split_step(::PosEM) = false

macro cache(expr)
    name = expr.args[2].args[1].args[1]
    fields = expr.args[3].args[2:2:end]
    cache_vars = Expr[]
    rand_vars = Expr[]
    jac_vars = Pair{Symbol,Expr}[]
    ratenoise_vars = Expr[]
    for x in fields
        if x.args[2] == :uType || x.args[2] == :rateType ||
            x.args[2] == :kType || x.args[2] == :uNoUnitsType #|| x.args[2] == :possibleRateType
            push!(cache_vars,:(c.$(x.args[1])))
        elseif x.args[2] == :JCType
            push!(cache_vars,:(c.$(x.args[1]).duals...))
        elseif x.args[2] == :GCType
            push!(cache_vars,:(c.$(x.args[1]).duals))
        elseif x.args[2] == :DiffCacheType
            push!(cache_vars,:(c.$(x.args[1]).du))
            push!(cache_vars,:(c.$(x.args[1]).dual_du))
        elseif x.args[2] == :JType || x.args[2] == :WType
            push!(jac_vars,x.args[1] => :(c.$(x.args[1])))
        elseif x.args[2] == :randType
            push!(rand_vars,:(c.$(x.args[1])))
        elseif x.args[2] == :rateNoiseType || x.args[2] == :rateNoiseCollectionType
            # Should be a pair for handling non-diagonal
            push!(ratenoise_vars,:(c.$(x.args[1])))
        end
    end
    quote
        $expr
        $(esc(:(StochasticDiffEq.full_cache)))(c::$name) = tuple($(cache_vars...))
        $(esc(:(StochasticDiffEq.jac_iter)))($(esc(:c))::$name) = tuple($(jac_vars...))
        $(esc(:(StochasticDiffEq.rand_cache)))($(esc(:c))::$name) = tuple($(rand_vars...))
        $(esc(:(StochasticDiffEq.ratenoise_cache)))($(esc(:c))::$name) = tuple($(ratenoise_vars...))
    end
end

struct PosEMConstantCache <: StochasticDiffEqConstantCache end
@cache struct PosEMCache{uType,rateType,rateNoiseType} <: StochasticDiffEqMutableCache
  u::uType
  uprev::uType
  tmp::uType
  rtmp1::rateType
  rtmp2::rateNoiseType
end

StochasticDiffEq.alg_cache(alg::PosEM,prob,u,ΔW,ΔZ,p,rate_prototype,noise_rate_prototype,jump_rate_prototype,::Type{uEltypeNoUnits},::Type{uBottomEltypeNoUnits},::Type{tTypeNoUnits},uprev,f,t,dt,::Type{Val{false}}) where {uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits} = PosEMConstantCache()

function StochasticDiffEq.alg_cache(alg::PosEM,prob,u,ΔW,ΔZ,p,rate_prototype,noise_rate_prototype,jump_rate_prototype,::Type{uEltypeNoUnits},::Type{uBottomEltypeNoUnits},::Type{tTypeNoUnits},uprev,f,t,dt,::Type{Val{true}}) where {uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits}
  tmp = zero(u); rtmp1 = zero(rate_prototype);
  if noise_rate_prototype !== nothing
    rtmp2 = zero(noise_rate_prototype)
  else
    rtmp2 = nothing
  end
  PosEMCache(u,uprev,tmp,rtmp1,rtmp2)
end

@muladd function StochasticDiffEq.perform_step!(integrator, cache::PosEMConstantCache)
    @unpack t,dt,uprev,u,W,P,c,p,f = integrator

    # CHANGE clamp the expectation at zero
    K = max.(uprev .+ dt .* integrator.f(uprev,p,t), zero(eltype(uprev)))

    if is_split_step(integrator.alg)
        u_choice = K
    else
        u_choice = uprev
    end

    if !is_diagonal_noise(integrator.sol.prob) || typeof(W.dW) <: Number
        noise = integrator.g(u_choice,p,t)*W.dW
    else
        noise = integrator.g(u_choice,p,t).*W.dW
    end

    if P !== nothing
        tmp = c(uprev, p, t, P.dW, nothing)
        u = K + noise + tmp
    else
        u = K + noise
    end
    # CHANGE clamp next value at zero
    integrator.u = max.(u, zero(eltype(u)))
end

@muladd function StochasticDiffEq.perform_step!(integrator, cache::PosEMCache)
    @unpack tmp,rtmp1,rtmp2 = cache
    @unpack t,dt,uprev,u,W,P,c,p = integrator
    integrator.f(rtmp1,uprev,p,t)

    # CHANGE clamp the expectation at zero
    let zero = zero(eltype(uprev))
        @.. u = max(uprev + dt * rtmp1, zero)
    end

    if is_split_step(integrator.alg)
        u_choice = u
    else
        u_choice = uprev
    end

    integrator.g(rtmp2,u_choice,p,t)

    if P !== nothing
        c(tmp, uprev, p, t, P.dW, nothing)
    end

    if is_diagonal_noise(integrator.sol.prob)
        @.. rtmp2 *= W.dW
        if P !== nothing
            @.. u += rtmp2 + tmp
        else
            @.. u += rtmp2
        end
    else
        mul!(rtmp1,rtmp2,W.dW)
        if P !== nothing
            @.. u += rtmp1 + tmp
        else
            @.. u += rtmp1
        end
    end

    # CHANGE clamp next value at zero
    let zero = zero(eltype(u))
        @.. u = max(u, zero)
    end
end
