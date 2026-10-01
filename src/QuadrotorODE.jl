module QuadrotorODE

using LinearAlgebra
using Parameters
using StaticArrays

include("quaternions.jl")
using .Quaternions: conjugate, multiply, rot, dqdt, G, q2rp, rp2q, q2qv, qv2q

# Dimensions
const nx = 13
const nd = 12
const nu = 4
const nw = 3

# System's properties

struct System
    g::Vector # gravitation acceleration
    m::Real   # mass
    h::Vector # first moment of mass
    Σ::Matrix # second moment of mass
    a::Real   # moment arm of propellers
    kₜ::Real  # propeller thrust coefficient
    kₘ::Real  # propeller torque coefficient
end

# Dynamics (accelerations)

skew(a) = @SMatrix [
    0 -a[3] a[2]
    a[3] 0 -a[1]
    -a[2] a[1] 0
]

function mass_matrix(system::System)
    @unpack m, h, Σ = system

    return [
        m*I(3) -skew(h)
        skew(h) tr(Σ)*I(3)-Σ
    ]
end

"""
coriolis and centrifugal terms contributing to v̇, i.e the acceleration of the body
relative to the moving reference frame.
"""
function reference_frame_bias(system::System, v, ω)
    @unpack m, h, Σ = system

    return vcat(
        skew(ω) * (m * v + skew(ω) * h),
        -skew(ω) * Σ * ω + h × (skew(ω) * v)
    )
end

function external_torques(system::System, q, u, w)
    @unpack g, m, h, a, kₘ, kₜ = system

    G = rot(conjugate(q), g)

    F = @SVector [0, 0, sum(u)]
    W = @SMatrix [
        -a*kₜ +a*kₜ +a*kₜ -a*kₜ
        -a*kₜ -a*kₜ +a*kₜ +a*kₜ
        +kₘ -kₘ +kₘ -kₘ
    ]

    τ = vcat(
        m * G + F + w,
        skew(h) * G + W * u
    )
end

# State space descriptions

"""
Calculates the rate of change of the state according to the state description ẋ = f(x,u).
The state of the system x = [r, q, v, ω] where
    r - position relative to the origin of the world (inertial) frame expressed in world cooridnates
    q - attitude to the inertial frame expressed in world cooridnates
    v - linear translational velocity relative to the moving the body (reference) frame in body cooridinates
    ω - angular velocity of the quadrotor's body in local coordinates.

arguments:
    system - properties of the quadrotor
    x - system's state (, where v and ω are expressed in the frame of the quadrotor)
    u - control inputs

returns:
    ẋ - rate of change of the state (ẋ = [v, q̇, v̇, ω̇])

"""
function dynamics(system, x, u, w=zeros(3))
    @assert length(x) == 13
    @assert length(u) == 4

    _, q, v, ω = x[1:3], x[4:7], x[8:10], x[11:13]

    ṙ = rot(q, v)
    q̇ = multiply(q, dqdt(ω))

    H = mass_matrix(system)
    c = reference_frame_bias(system, v, ω)
    τ = external_torques(system, q, u, w)

    a = inv(H) * (-c + τ)

    return Vector(vcat(ṙ, q̇, a[1:3], a[4:6]))
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
Calculates the difference between the current and reference state. The relative rotation can be expressed
using:
    - the vector part of a quaternion - :qv
    - Rodrigues parameters            - :rp
    - quaternion                      - :q

arguments:
    x   - current state
    x₀  - reference state
    rep - relative rotation representation (default = :qv)

returns:
    dz - state difference (dz = [dr, dθ, dv, dω]) 
  
"""
function state_difference(x, x₀, rep=:qv)
    @assert length(x) == 13
    @assert length(x₀) == 13

    dq = multiply(conjugate(x₀[4:7]), x[4:7])

    dθ = if rep == :qv
        q2qv(dq)
    elseif rep == :rp
        q2rp(dq)
    elseif rep == :q
        dq
    end

    dr = x[1:3] - x₀[1:3]
    dv = x[8:10] - x₀[8:10]
    dω = x[11:13] - x₀[11:13]

    return vcat(dr, dθ, dv, dω)
end

"""
Composes state x from x₀ and dz.

arguments:
    x₀ - reference state
    dz - state difference (dz = [dr, dθ, dv, dω]) 

returns:
    x  - new state
  
"""
function state_composition(x₀, dz, rep=:rp; normalize=true)
    @assert length(x₀) == 13
    @assert length(dz) == 12

    dθ = dz[4:6]

    dq = if rep == :rp
        rp2q(dθ)
    elseif rep == :qv
        qv2q(dθ)
    end

    r = x₀[1:3] + dz[1:3]
    q = multiply(x₀[4:7], dq)
    v = x₀[8:10] + dz[7:9]
    ω = x₀[11:13] + dz[10:12]

    x = vcat(r, q, v, ω)
    normalize && normalize_state!(x)

    return x
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
