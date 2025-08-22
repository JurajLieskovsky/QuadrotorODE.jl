module QuadrotorODE

using LinearAlgebra
using Parameters
using StaticArrays

include("quaternions.jl")
using .Quaternions: conjugate, multiply, rot, dqdt, G, q2rp, q2qv

# Dimensions
const nx = 13
const nz = 12
const nu = 4

# System's properties

struct System
    g::Real # gravitation acceleration
    m::Real   # mass
    J::Matrix # moment of inertia
    a::Real   # moment arm of propellers
    kₜ::Real  # propeller thrust coefficient
    kₘ::Real  # propeller torque coefficient
end

# Dynamics (accelerations)

"""
Calculates the quadrotor's body-frame accelerations.

arguments:
    system - properties of the quadrotor
    q - orientation of the quadrotor (quaternion)
    v - linear velocity (in the frame of the quadrotor)
    ω - angular velocity (in the frame of the quadrotor)
    u - control inputs

returns:
    v̇ - linear acceleration
    ω̇ - angular accelaration

"""
function body_frame_acceleration(system::System, q, v, ω, u)
    @unpack g, m, J, a, kₘ, kₜ = system

    G = @SVector [0, 0, -g]
    F = @SVector [0, 0, sum(u)]
    W = @SMatrix [
        -a*kₜ +a*kₜ +a*kₜ -a*kₜ
        -a*kₜ -a*kₜ +a*kₜ +a*kₜ
        +kₘ -kₘ +kₘ -kₘ
    ]

    v̇ = rot(conjugate(q), G) + F / m - ω × v
    ω̇ = J \ (W * u - ω × (J * ω))

    return v̇, ω̇
end

# State space descriptions

"""
Calculates the rate of change of the state according to the state description ẋ = f(x,u).
The state of the system x = [r, q, v, ω] uses a combination of position r and orientation q
expressed in the global frame and the linear translational velocity v and angular velocity ω
expressed in the local frame of the quadrotor's body.

arguments:
    system - properties of the quadrotor
    x - system's state (, where v and ω are expressed in the frame of the quadrotor)
    u - control inputs

returns:
    ẋ - rate of change of the state (ẋ = [v, q̇, v̇, ω̇])

"""
function dynamics(system, x, u)
    @assert length(x) == 13
    @assert length(u) == 4

    _, q, v, ω = x[1:3], x[4:7], x[8:10], x[11:13]

    ṙ = rot(q, v)
    q̇ = multiply(q, dqdt(ω))
    v̇, ω̇ = body_frame_acceleration(system, q, v, ω, u)

    return Vector(vcat(ṙ, q̇, v̇, ω̇))
end

# Jacobian

"""
Calculates E(x) where ∂x/∂z = E(x) and ∂x/∂z = E(x)ᵀ.

"""
function jacobian(x)
    E = zeros(eltype(x), 13, 12)
    E[1:3, 1:3] .= Matrix{Float64}(I, 3, 3)
    E[4:7, 4:6] .= G(x[4:7])
    E[8:13, 7:12] .= Matrix{Float64}(I, 6, 6)
    return E
end

# State difference utility

"""
Calculates the difference between the current and reference state. The relative rotation is expressed
using Rodrigues parameters.

arguments:
    x  - current state
    x₀ - reference state

returns:
    dz - state difference (dz = [dr, dθ, dv, dω]) 
  
"""
function state_difference(x, x₀, rep=:rp)
    @assert length(x) == 13
    @assert length(x₀) == 13

    dq = multiply(conjugate(x₀[4:7]), x[4:7])

    dθ = if rep == :rp
        q2rp(dq)
    elseif rep == :qv
        q2qv(dq)
    end

    dr = x[1:3] - x₀[1:3]
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

end
