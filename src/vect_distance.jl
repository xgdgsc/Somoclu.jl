# Feature-space distances for BMU search — from denseCpuKernels.cpp
# (EuclideanDistance / NormPDistance / NormInfDistance).
# C++ only wires these into the dense CPU kernel; sparse keeps Euclidean.

"""
    parse_vect_distance(s) -> (kind::Symbol, p::Float32)

Parse a C++/CLI-style vector distance string: `"euclidean"`, `"norm-inf"`,
or `"norm-p"` with any `p > 0` (e.g. `"norm-1"`, `"norm-2.5"`).
Returns `(:euclidean, 2)`, `(:inf, 0)`, or `(:p, p)`.
"""
function parse_vect_distance(s::AbstractString)
    s == "euclidean" && return (:euclidean, 2.0f0)
    s == "norm-inf" && return (:inf, 0.0f0)
    if startswith(s, "norm-")
        p = tryparse(Float32, SubString(s, 6))
        if p !== nothing && p > 0
            return (:p, Float32(p))
        end
    end
    error("Unknown vect_distance \"$s\" (use \"euclidean\", \"norm-inf\", or \"norm-p\" with p>0)")
end

"""Integer kind for KernelAbstractions: 0=euclidean, 1=inf, 2=p-norm."""
function vect_distance_kind(spec::Tuple{Symbol,Float32})
    kind, p = spec
    kind === :euclidean && return (0, 2.0f0)
    kind === :inf && return (1, 0.0f0)
    kind === :p && return (2, p)
    error("Invalid vect_distance spec $spec")
end

"""
    feature_distance(codebook, neuron, data, n, nDimensions, dist_kind, p)

Host-side feature distance between codebook column `neuron` and data column `n`.
`dist_kind`: `0` Euclidean, `1` infinity, `2` p-norm (matches denseCpuKernels.cpp).
"""
@inline function feature_distance(codebook::AbstractMatrix{Float32}, neuron::Int,
                                  data::AbstractMatrix{Float32}, n::Int,
                                  nDimensions::Int, dist_kind::Int, p::Float32)
    if dist_kind == 1  # infinity
        m = 0.0f0
        @inbounds for d in 1:nDimensions
            Δ = abs(codebook[d, neuron] - data[d, n])
            Δ > m && (m = Δ)
        end
        return m
    elseif dist_kind == 2  # p-norm
        acc = 0.0
        @inbounds for d in 1:nDimensions
            acc += abs(Float64(codebook[d, neuron]) - Float64(data[d, n]))^Float64(p)
        end
        return Float32(acc^(1 / Float64(p)))
    else  # euclidean
        acc = 0.0f0
        @inbounds for d in 1:nDimensions
            Δ = codebook[d, neuron] - data[d, n]
            acc += Δ * Δ
        end
        return sqrt(acc)
    end
end
