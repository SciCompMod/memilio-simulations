abstract type PythonSimulator{T_X} end

"""
    PythonSimulator{T_X}
A type that represents a simulator for the particles in the particle filter.
`T_X` is the type of the state vector.
"""

mutable struct MemSEIRSimulator{T_X} <: PythonSimulator{T_X}
    sim_function::PyObject
    p::Vector{Float64} # Parameters for the simulation function
    t_initial::Float64
    t::Float64
    u::T_X
    u_initial::T_X

    function MemSEIRSimulator(sim_function::PyObject, p::Vector{Float64}, t_initial::Float64, u_initial::T_X) where{T_X}
        new{T_X}(sim_function, p, t_initial, t_initial, u_initial, u_initial)
    end
end

function reset_simulator!(simulator::MemSEIRSimulator)
    simulator.t = simulator.t_initial
    simulator.u = simulator.u_initial
    return nothing
end

function reinit_simulator!(simulator::MemSEIRSimulator, xp, t)
    simulator.t = t 
    simulator.u = xp
    return nothing
end

function simulation_step!(simulator::MemSEIRSimulator, dt::Float64)
    tend = simulator.t + dt
    sim = simulator.sim_function(simulator.p, simulator.u, simulator.t, tend)
    simulator.u = sim[2:end,end] # Important: Keep ordering as in python function, also of the parameters.
    simulator.t = sim[1,end] # Update time
    return nothing
end

function full_simulation(simulator::MemSEIRSimulator, tend::Float64)
    sim = simulator.sim_function(simulator.p, simulator.u, simulator.t, tend)
    return sim
end



mutable struct MemSSIRSSimulator{T_X} <: PythonSimulator{T_X}
    sim_function::PyObject
    p::Vector{Float64} # Parameters for the simulation function
    t_initial::Float64
    t::Float64
    u::T_X
    u_initial::T_X

    function MemSSIRSSimulator(sim_function::PyObject, p::Vector{Float64}, t_initial::Float64, u_initial::T_X) where{T_X}
        new{T_X}(sim_function, p, t_initial, t_initial, u_initial, u_initial)
    end
end

function reset_simulator!(simulator::MemSSIRSSimulator)
    simulator.t = simulator.t_initial
    simulator.u = simulator.u_initial
    return nothing
end

function reinit_simulator!(simulator::MemSSIRSSimulator, xp, t)
    simulator.t = t 
    simulator.u = xp
    return nothing
end

function simulation_step!(simulator::MemSSIRSSimulator, dt::Float64)
    tend = simulator.t + dt
    sim = simulator.sim_function(simulator.p, simulator.u, simulator.t, tend)
    simulator.u = sim[2:end,end] # Important: Keep ordering as in python function, also of the parameters.
    simulator.t = sim[1,end] # Update time
    return nothing
end

function full_simulation(simulator::MemSSIRSSimulator, tend::Float64)
    sim = simulator.sim_function(simulator.p, simulator.u, simulator.t, tend)
    return sim
end




"""
The flow is now the state u, such that we can access it in the observation function and BootstrapFilter do work with it. 
But we need now to explicitly keep track also of the model states.

Important: Data is assumed to be the cumulative flow since the last timepoint!

"""

mutable struct MemFlowSEIRSimulator{T_X, T_F} <: PythonSimulator{T_X}
    sim_function::PyObject
    p::Vector{Float64} # Parameters for the simulation function
    t_initial::Float64
    t::Float64
    u::T_F
    states::T_X
    states_initial::T_X

    function MemFlowSEIRSimulator(sim_function::PyObject, p::Vector{Float64}, t_initial::Float64, u_initial::T_F, x_initial::T_X) where{T_X, T_F}
        new{T_X, T_F}(sim_function, p, t_initial, t_initial, u_initial, x_initial, x_initial)
    end
end

function reset_simulator!(simulator::MemFlowSEIRSimulator)
    simulator.t = simulator.t_initial
    simulator.u .= 0.0 # reset flows
    simulator.states = simulator.states_initial
    return nothing
end

function reinit_simulator!(simulator::MemFlowSEIRSimulator, xp, t)
    simulator.t = t 
    simulator.u .= xp # reinit flows
    # simulator.states = xp # reinit states as well 
    return nothing
end

function simulation_step!(simulator::MemFlowSEIRSimulator, dt::Float64)
    tend = simulator.t + dt
    sim = simulator.sim_function(simulator.p, simulator.states, simulator.t, tend)
    result = sim[1]
    flows = sim[2]
    simulator.states = result[2:end,end] # Important: Keep ordering as in python function, also of the parameters.
    simulator.t = flows[1,end] # Update time
    simulator.u = flows[2:end,end]
    return nothing
end

function full_simulation(simulator::MemFlowSEIRSimulator, tend::Float64)
    reset_simulator!(simulator)
    sim = simulator.sim_function(simulator.p, simulator.states_initial, simulator.t, tend)
    return sim
end