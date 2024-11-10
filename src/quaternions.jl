module Quaternions

using LinearAlgebra

"""
Calculates the complex conjugate of a quaternion

arguments:
    q - quaternion

returns:
    q* - complex conjugate of q

"""
conjugate(q) = [q[1], -q[2], -q[3], -q[4]]


"""
Multiplies two quaternion p and q.

arguments:
    p - quaternion
    q - quaternion

returns:
    pq - product of quaternion mulitiplication

"""
function multiply(p, q)
    p₀, p⃗ = p[1], p[2:4]
    q₀, q⃗ = q[1], q[2:4]
    return vcat(p₀ * q₀ - p⃗'q⃗, p₀ * q⃗ + q₀ * p⃗ + p⃗ × q⃗)
end


"""
Converts angular velocity to a quaternion's rate of change.

arguments:
    q - quaternion
    ω - angular velocity (in the coordinates of q)

returns:
    q̇ - quaternion's rate of change

"""
function q̇(q, ω)
    q₀, q⃗ = q[1], q[2:4]
    return 0.5 * vcat(-q⃗'ω, q₀ * ω + q⃗ × ω)
end


"""
Approximates a quaternion as a rotation about an arbitrary axis.

arguments:
    dq - quaternion

returns:
    dθ - angle * axis of rotation

"""
function dθ(dq)
    dq₀, dq⃗ = dq[1], dq[2:4]

    magnitude = sqrt(dq⃗'dq⃗ + eps(Float64))

    θ = 2 * atan(magnitude / dq₀)
    u = dq⃗ / magnitude
    return θ * u
end


"""
Transforms a 3D vector according to a unit quaternion.

arguments:
    q - unit quaternion
    v - 3D vector

returns:
    v′ - transformed 3D vector

"""
function rot(q, v)
    q₀, q⃗ = q[1], q[2:4]
    return v + 2 * q⃗ × (q⃗ × v + q₀ * v)
end

end
