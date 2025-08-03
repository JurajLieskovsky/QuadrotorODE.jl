module Quaternions

using LinearAlgebra

"""
Calculates the complex conjugate of a quaternion

"""
conjugate(q) = [q[1], -q[2], -q[3], -q[4]]


"""
Multiplies quaternions p and q.

"""
function multiply(p, q)
    p₀, p⃗ = p[1], p[2:4]
    q₀, q⃗ = q[1], q[2:4]
    return vcat(p₀ * q₀ - p⃗'q⃗, p₀ * q⃗ + q₀ * p⃗ + p⃗ × q⃗)
end


"""
Infinitesimal rotation around an arbirary axis as a quaternion

δq = q(δθ), δθ → 0

"""
δq(δθ) = vcat(1, 0.5 * δθ)


"""
Converts angular velocity to a quaternion rate of change.

q̇ = δq/δθ ω

"""
q̇(ω) = vcat(0, 0.5 * ω)


"""
Transforms a 3D vector according to a unit quaternion.

"""
function rot(q, v)
    q₀, q⃗ = q[1], q[2:4]
    return v + 2 * q⃗ × (q⃗ × v + q₀ * v)
end


"""
Approximates a quaternion as a rotation about an arbitrary axis.

"""
function dθ(dq)
    dq₀, dq⃗ = dq[1], dq[2:4]

    magnitude = sqrt(dq⃗'dq⃗ + eps(Float64))

    θ = 2 * atan(magnitude / dq₀)
    u = dq⃗ / magnitude
    return θ * u
end

end
