module Quaternions

using LinearAlgebra
using StaticArrays

"""Calculates the complex conjugate of a quaternion."""
conjugate(q) = @SVector [q[1], -q[2], -q[3], -q[4]]


"""
Multiplies quaternions p and q.

s₀ = p₀ * q₀ - p⃗'q⃗
s⃗  = p₀ * q⃗ + q₀ * p⃗ + p⃗ × q⃗

"""
function multiply(p, q)
    p₀, p⃗ = p[1], view(p, 2:4)
    q₀, q⃗ = q[1], view(q, 2:4)

    return @SVector [
        p₀ * q₀ - p⃗[1] * q⃗[1] - p⃗[2] * q⃗[2] - p⃗[3] * q⃗[3],
        p₀ * q⃗[1] + p⃗[1] * q₀ + p⃗[2] * q⃗[3] - p⃗[3] * q⃗[2],
        p₀ * q⃗[2] - p⃗[1] * q⃗[3] + p⃗[2] * q₀ + p⃗[3] * q⃗[1],
        p₀ * q⃗[3] + p⃗[1] * q⃗[2] - p⃗[2] * q⃗[1] + p⃗[3] * q₀
    ]
end


"""
Transforms a 3D vector according to a unit quaternion.

v' = v + 2 * q⃗ × (q⃗ × v + q₀ * v)

"""
function rot(q, v)
    q₀, q⃗ = q[1], view(q, 2:4)

    t = @SVector [
        2 * (q⃗[2] * v[3] - q⃗[3] * v[2])
        2 * (q⃗[3] * v[1] - q⃗[1] * v[3])
        2 * (q⃗[1] * v[2] - q⃗[2] * v[1])
    ]

    return @SVector [
        v[1] + q₀ * t[1] + (q⃗[2] * t[3] - q⃗[3] * t[2])
        v[2] + q₀ * t[2] + (q⃗[3] * t[1] - q⃗[1] * t[3])
        v[3] + q₀ * t[3] + (q⃗[1] * t[2] - q⃗[2] * t[1])
    ]
end

"""
Converts angular velocity to a quaternion rate of change.

q̇ = ∂q/∂θ(0) ω⃗

"""
q̇(ω) = @SVector [0, 0.5 * ω[1], 0.5 * ω[2], 0.5 * ω[3]]


"""
Calculates the jacobian G(p) which maps an angular velocity ω
in the local frame to a quaternion rate of change ṗ

ṗ = p q̇ = p ∂q/∂θ(0) ω = G(p) * ω

"""
function G(q)
    q₀, p⃗ = q[1], view(q, 2:4)

    return @SMatrix [
        -p⃗[1] -p⃗[2] -p⃗[3]
        q₀ -p⃗[3] p⃗[2]
        p⃗[3] q₀ -p⃗[1]
        -p⃗[2] p⃗[1] q₀
    ]
end


"""
Converts a quaternion to a rotation about an axis.

"""
function q2θ(q, ε=eps())
    q₀, q⃗ = q[1], view(q, 2:4)

    q⃗2 = mapreduce(e -> e^2, +, q⃗)  # q⃗⋅q⃗
    nrm = sqrt(q⃗2 + ε)              # ||q⃗||₂

    # return identity (can occur only when ε=0)
    nrm == 0 && return @SVector [0, 0, 0]

    scl = 2 * atan(nrm, q₀) / nrm   # θ/||q⃗||₂

    return @SVector [
        scl * q⃗[1], scl * q⃗[2], scl * q⃗[3]
    ]
end

"""
Converts a quaternion to a rotation about an axis.

"""
function θ2q(θ, ε=eps())
    θ2 = mapreduce(e -> e^2, +, θ)
    nrm = sqrt(θ2 + ε)

    # return identity (can occur only when ε=0)
    nrm == 0 && return @SVector [1, 0, 0, 0]

    scl = 1 / nrm * sin(nrm / 2)

    return @SVector [
        cos(nrm / 2), scl * θ[1], scl * θ[2], scl * θ[3]
    ]
end

"""
Converts a quaternion to a Rodrigues parameter.

"""
function rp2q(ϕ)
    (1 / sqrt(1 + ϕ' * ϕ)) * [1; ϕ]
end

"""
Converts a Rodrigues parameter to a quaternion.

"""
function q2rp(q)
    q[2:4] / q[1]
end

end
