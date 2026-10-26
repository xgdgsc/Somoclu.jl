# Neighborhood weights and cooling — ported from
# somoclu/src/mapDistanceFunctions.cpp and somoclu/src/training.cpp

@inline function gaussian_neighborhood(distance::Float32, radius::Float32, std_coeff::Float32)
    std = radius * std_coeff
    return exp((-distance * distance) / (2 * std * std))
end

@inline function get_weight(distance::Float32, radius::Float32, scaling::Float32,
                            compact_support::Bool, gaussian::Bool, std_coeff::Float32)
    result = 0.0f0
    if gaussian
        if !compact_support || distance <= radius
            result = gaussian_neighborhood(distance, radius, std_coeff)
        end
    else
        if distance <= radius
            result = 1.0f0
        end
    end
    return scaling * result
end

@inline function linear_cooling(start::Float32, finish::Float32, nEpoch::Float32, epoch::Float32)
    diff = (start - finish) / (nEpoch - 1.0f0)
    return start - (epoch * diff)
end

@inline function exponential_cooling(start::Float32, finish::Float32, nEpoch::Float32, epoch::Float32)
    diff = if finish == 0.0f0
        -log(0.1f0) / nEpoch
    else
        -log(finish / start) / nEpoch
    end
    return start * exp(-epoch * diff)
end
