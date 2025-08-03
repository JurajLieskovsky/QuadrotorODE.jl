module QuadrotorODE

using LinearAlgebra
using Parameters

include("quaternions.jl")
using .Quaternions

# Dimensions
const nx = 13
const nz = 12
const nu = 4

# System's properties

struct System
    g::Vector # gravitation acceleration
    m::Real   # mass
    J::Matrix # moment of inertia
    W::Matrix # moment matrix

    function System(gravitation_accelaration, mass, moment_of_inertia, arm_length, torque_coefficient)
        rotor_radius_vectors = [
            [1, -1, 0],
            [1, 1, 0],
            [-1, 1, 0],
            [-1, -1, 0]
        ]

        z_axis = [0, 0, 1]
        direction_of_rotation = [1, -1, 1, -1]

        moment_matrix = mapreduce(
            arg -> arg[1] × z_axis * arm_length - arg[2] * z_axis * torque_coefficient,
            hcat,
            zip(rotor_radius_vectors, direction_of_rotation)
        )

        return new(gravitation_accelaration, mass, moment_of_inertia, moment_matrix)
    end
end

# Dynamics (accelerations)

"""
Calculates the quadrotor's angular acceleration.

arguments:
    system - properties of the quadrotor
    ω - angular velocity (in the frame of the quadrotor)
    u - control inputs

returns:
    ω̇ - angular accelaration

"""
function angular_acceleration(system::System, ω, u)
    @unpack J, W = system
    return J \ (W * u - ω × (J * ω))
end


"""
Calculates the quadrotor's linear acceleration.

arguments:
    system - properties of the quadrotor
    q - orientation of the quadrotor (quaternion)
    u - control inputs

returns:
    ω̇ - angular accelaration

"""
function linear_acceleration(system::System, q, u)
    @unpack g, m = system
    F = [0, 0, sum(u)]
    return g + Quaternions.rot(q, F) / m
end

# State incrementation utility

"""
Calculates the system's state as incremented by dz.

arguments:
x₀ - system's state (x₀ = [r, q, v, ω])
dz - incrementation of the state (dz = [dr, dθ, dv, dω]) 

returns:
x₀ + dx(dz) - incremented state (dx = [dr, q̇(q,dθ), dv, dω])

"""
function δx(x₀, δz)
    @assert length(x₀) == 13
    @assert length(δz) == 12

    r, q, v, ω = x₀[1:3], x₀[4:7], x₀[8:10], x₀[11:13]
    δr, δθ, δv, δω = δz[1:3], δz[4:6], δz[7:9], δz[10:12]

    return vcat(r + δr, Quaternions.multiply(Quaternions.δq(δθ), q), v + δv, ω + δω)
end

# State difference utility

"""
Calculates the difference between the current and reference state in the tangential direction of the reference state.

arguments:
    x  - current state
    x₀ - reference state

returns:
    dz - state difference (dz = [dr, dθ, dv, dω]) 
    
The approximation x ≈ x₀ + dx(dz), where dx = [dr, q̇(q,dθ), dv, dω], should be accurate for very small values of dθ
    
"""
function state_difference(x, x₀)
    @assert length(x) == 13
    @assert length(x₀) == 13

    dr = x[1:3] - x₀[1:3]
    dθ = Quaternions.dθ(Quaternions.multiply(x[4:7], Quaternions.conjugate(x₀[4:7])))
    dv = x[8:10] - x₀[8:10]
    dω = x[11:13] - x₀[11:13]

    return vcat(dr, dθ, dv, dω)
end

# State normalization utility

"""
Normalizes the quaternion, that represents the quadrotors orientation, within the state vector.
"""
function normalize_state!(x)
    q = view(x, 4:7)
    q ./= norm(q)
end

# State space descriptions

"""
Calculates the rate of change of the state according to the state description ẋ = f(x,u).

arguments:
system - properties of the quadrotor
x - system's state (x = [r, q, v, ω])
u - control inputs

returns:
ẋ - rate of change of the state (ẋ = [v, q̇, v̇, ω̇])

"""
function dynamics(system, x, u)
    @assert length(x) == 13
    @assert length(u) == 4

    _, q, v, ω = x[1:3], x[4:7], x[8:10], x[11:13]
    ω̇ = angular_acceleration(system, ω, u)
    v̇ = linear_acceleration(system, q, u)

    return vcat(v, Quaternions.q̇(q, ω), v̇, ω̇)
end


"""
Calculates the rate of change of the state in the tangential direction.

arguments:
system - properties of the quadrotor
x₀ - system's state (x₀ = [r, q, v, ω])
dz - increment of the state (dz = [dr, dθ, dv, dω]) 
u  - control inputs

returns:
dż - rate of change of the state in tangential direction (dż = [v, ω, v̇, ω̇])

"""
function tangential_dynamics(system, x₀, dz, u)
    @assert length(x₀) == 13
    @assert length(dz) == 12
    @assert length(u) == 4

    x = δx(x₀, dz)
    _, q, v, ω = x[1:3], x[4:7], x[8:10], x[11:13]

    ω̇ = angular_acceleration(system, ω, u)
    v̇ = linear_acceleration(system, q, u)

    return vcat(v, ω, v̇, ω̇)
end

end
