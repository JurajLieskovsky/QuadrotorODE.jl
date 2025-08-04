module Quaternions

using LinearAlgebra
using StaticArrays

"""Calculates the complex conjugate of a quaternion."""
conjugate(q) = @SVector [q[1], -q[2], -q[3], -q[4]]


"""Multiplies quaternions p and q."""
function multiply(p, q)
    p₀, p⃗ = p[1], view(p, 2:4)
    q₀, q⃗ = q[1], view(q, 2:4)
    return vcat(p₀ * q₀ - p⃗'q⃗, p₀ * q⃗ + q₀ * p⃗ + p⃗ × q⃗)
end


"""
Infinitesimal rotation around an arbirary axis as a quaternion

δq = q(δθ), δθ → 0

"""
δq(δθ) = @SVector [1, 0.5 * δθ[1], 0.5 * δθ[2], 0.5 * δθ[3]]


"""
Converts angular velocity to a quaternion rate of change.

q̇ = δq/δθ ω

"""
q̇(ω) = @SVector [0, 0.5 * ω[1], 0.5 * ω[2], 0.5 * ω[3]]


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


"""Approximates a quaternion as a rotation about an arbitrary axis."""
function dθ(dq)
    dq₀, dq⃗ = dq[1], view(dq, 2:4)

    magnitude = sqrt(dq⃗'dq⃗ + eps(Float64))

    θ = 2 * atan(magnitude, dq₀)
    u = dq⃗ / magnitude
    return θ * u
end

end
