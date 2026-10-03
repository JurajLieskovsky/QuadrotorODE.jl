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

const I33 = LinearAlgebra.I(3)

# System's properties

struct System
    g::Vector # gravitation acceleration
    m::Real   # mass
    h::Vector # first mass moment 
    I::Matrix # rotational inertia
    a::Real   # moment arm of propellers
    kₜ::Real  # propeller thrust coefficient
    kₘ::Real  # propeller torque coefficient
end

"""
Calculates the pseudo-inertial matrix of the system from the mass m, first mass moment h, and rotational inertia I.
"""
function pseudo_inertial_matrix(m, h, I)
    Σ = 0.5 * tr(I) * I33 - I
    return [m h'; h Σ]
end

# Dynamics (accelerations)

"""
Calculates the skew-symmetric matrix of vector a, such that skew(a) * b == a × b.
"""
skew(a) = @SMatrix [
    0 -a[3] a[2]
    a[3] 0 -a[1]
    -a[2] a[1] 0
]

"""
Calculates the system's mass matrix H, such that H * [v̇; α] equals the net force and torque acting on the
quadrotor, expressed in body coordinates.
"""
function mass_matrix(system::System)
    @unpack m, h, I = system

    return [
        m*I33 -skew(h)
        skew(h) I
    ]
end

"""
Calculates the bias force and torque (the gyroscopic and Coriolis terms stemming from the angular velocity ω),
excluding the contributions of gravity and the control inputs.
"""
function bias_torque(system::System, ω)
    @unpack h, I = system

    return vcat(
        skew(ω) * (skew(ω) * h),
        skew(ω) * I * ω
    )
end

"""
Calculates the force and torque due to gravity, expressed in body coordinates, given the quadrotor's
attitude q.
"""
function gravitational_torque(system::System, q)
    @unpack g, m, h = system

    G = rot(conjugate(q), g)

    return vcat(m * G, h × G)
end

"""
Calculates the force and torque produced by the propellers' thrusts in response to the control inputs u
in body coodrinates.
"""
function input_torque(system::System, u)
    @unpack a, kₘ, kₜ = system

    F = @SVector [0, 0, sum(u)]
    W = @SMatrix [
        -a*kₜ +a*kₜ +a*kₜ -a*kₜ
        -a*kₜ -a*kₜ +a*kₜ +a*kₜ
        +kₘ -kₘ +kₘ -kₘ
    ]

    return vcat(F, W * u)
end

# State space description

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
    w - disturbance

returns:
    ẋ - rate of change of the state (ẋ = [v, q̇, v̇, ω̇])

"""
function dynamics(system, x, u, w=zeros(6))
    @assert length(x) == 13
    @assert length(u) == 4
    @assert length(w) == 6

    q, v, ω = x[4:7], x[8:10], x[11:13]

    ṙ = rot(q, v)
    q̇ = multiply(q, dqdt(ω))

    H = mass_matrix(system)
    c = bias_torque(system, ω)
    τ_g = gravitational_torque(system, q)
    τ_u = input_torque(system, u)

    acc = inv(H) * (-c + τ_g + τ_u + w)

    v̇ = acc[1:3] - skew(ω) * v
    α = acc[4:6]

    return vcat(ṙ, q̇, v̇, α)
end

"""
Calculates the IMU observation y = [ω, s] that would be measured by a gyroscope and accelerometer located at the
origin of the body frame, according to the observation description y = h(x,u).

arguments:
    system - properties of the quadrotor
    x - system's state (, where v and ω are expressed in the frame of the quadrotor)
    u - control inputs
    w - disturbance

returns:
    y - IMU observation (y = [ω, s]), where s is the specific force sensed by the accelerometer

"""
function imu_observation(system, x, u, w=zeros(6))
    @assert length(x) == 13
    @assert length(u) == 4
    @assert length(w) == 6

    ω = x[11:13]

    H = mass_matrix(system)
    c = bias_torque(system, ω)
    τ_u = input_torque(system, u)

    res = inv(H) * (-c + τ_u + w)

    s = res[1:3]

    return vcat(ω, s)
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

# State utilities

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

"""
Normalizes the quaternion, that represents the quadrotors orientation, within the state vector.
"""
function normalize_state!(x)
    q = view(x, 4:7)
    q ./= norm(q)
end

end
