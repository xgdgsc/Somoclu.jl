# Map-space distances — ported from somoclu/src/mapDistanceFunctions.cpp
# Feature-space distances — ported from somoclu/src/denseCpuKernels.cpp

@inline function euclidean_distance_planar(som_x::Integer, som_y::Integer, x::Integer, y::Integer)
    xdist = abs(som_x - x)
    ydist = abs(som_y - y)
    return sqrt(Float32(xdist * xdist + ydist * ydist))
end

@inline function euclidean_distance_toroid(som_x::Integer, som_y::Integer, x::Integer, y::Integer,
                                          nSomX::Integer, nSomY::Integer)
    x1, x2 = minmax(som_x, x)
    y1, y2 = minmax(som_y, y)
    xdist = min(x2 - x1, x1 + nSomX - x2)
    ydist = min(y2 - y1, y1 + nSomY - y2)
    return sqrt(Float32(xdist * xdist + ydist * ydist))
end

# Hexagonal distances use float xdist so ±0.5 offsets are preserved
# (C++ declares unsigned int but applies ±0.5 — float is the mathematical intent).
@inline function euclidean_distance_hex_planar(som_x::Integer, som_y::Integer, x::Integer, y::Integer)
    x1, x2 = minmax(som_x, x)
    y1, y2 = minmax(som_y, y)
    xdist = Float32(x2 - x1)
    ydist = Float32(y2 - y1)
    if (y2 - y1) & 1 != 0
        xdist += ((y1 & 1) != 0) ? -0.5f0 : 0.5f0
    end
    return sqrt(xdist * xdist + ydist * ydist * 0.75f0)
end

@inline function euclidean_distance_hex_toroid(som_x::Integer, som_y::Integer, x::Integer, y::Integer,
                                              nSomX::Integer, nSomY::Integer)
    x1, x2 = minmax(som_x, x)
    y1, y2 = minmax(som_y, y)
    xdist = Float32(min(x2 - x1, x1 + nSomX - x2))
    ydist_i = min(y2 - y1, y1 + nSomY - y2)
    ydist = Float32(ydist_i)
    if (ydist_i & 1) != 0
        xdist += ((y1 & 1) != 0) ? -0.5f0 : 0.5f0
    end
    return sqrt(xdist * xdist + ydist * ydist * 0.75f0)
end

@inline function map_distance(som_x, som_y, bmu_x, bmu_y, nSomX, nSomY, rectangular::Bool, toroid::Bool)
    if rectangular
        return toroid ? euclidean_distance_toroid(som_x, som_y, bmu_x, bmu_y, nSomX, nSomY) :
                        euclidean_distance_planar(som_x, som_y, bmu_x, bmu_y)
    else
        return toroid ? euclidean_distance_hex_toroid(som_x, som_y, bmu_x, bmu_y, nSomX, nSomY) :
                        euclidean_distance_hex_planar(som_x, som_y, bmu_x, bmu_y)
    end
end

@inline function euclidean_feature_distance(codebook, data, neuron::Integer, sample::Integer, nDimensions::Integer)
    dist = 0.0f0
    @inbounds for d in 1:nDimensions
        Δ = codebook[d, neuron] - data[d, sample]
        dist += Δ * Δ
    end
    return sqrt(dist)
end
