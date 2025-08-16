module QuadrotorODE

using LinearAlgebra
using Parameters
using BlockDiagonals

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
function linear_acceleration(system::System, q, v, ω, u)
    @unpack g, m = system
    F = [0, 0, sum(u)]
    return Quaternions.rot(Quaternions.conjugate(q), g) + F / m - ω × v
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
    dθ = Quaternions.q2rp(Quaternions.multiply(Quaternions.conjugate(x₀[4:7]), x[4:7]))
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
    v̇ = linear_acceleration(system, q, v, ω, u)

    return vcat(Quaternions.rot(q, v), Quaternions.multiply(q, Quaternions.q̇(ω)), v̇, ω̇)
end

motion_jacobian(x) = BlockDiagonal(
    [Matrix{Float64}(I, 3, 3), Quaternions.G(x[4:7]), Matrix{Float64}(I, 6, 6)]
)

end
