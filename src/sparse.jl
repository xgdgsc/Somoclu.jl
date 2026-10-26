# Sparse CPU training — ported from somoclu/src/sparseCpuKernels.cpp (non-MPI path).
# Data layout: SparseMatrixCSC with size (nDimensions, nVectors); each column is a sample.
# Codebook stays dense. Missing features in a sample contribute 0 to the numerator
# (C++ behaviour: those codebook dimensions become 0 when denom ≠ 0).

"""Squared L2 norms of sparse sample columns."""
function sparse_sample_sqnorms(data::SparseMatrixCSC{<:Real})
    nVectors = size(data, 2)
    X2 = Vector{Float32}(undef, nVectors)
    @inbounds for n in 1:nVectors
        acc = 0.0f0
        for k in nzrange(data, n)
            v = Float32(nonzeros(data)[k])
            acc += v * v
        end
        X2[n] = acc
    end
    return X2
end

"""Squared L2 norms of dense codebook columns (neurons)."""
function codebook_sqnorms!(W2::AbstractVector{Float32}, codebook::AbstractMatrix{Float32})
    nDim, nNeurons = size(codebook)
    length(W2) == nNeurons || throw(DimensionMismatch("W2 length"))
    @inbounds for j in 1:nNeurons
        acc = 0.0f0
        for d in 1:nDim
            w = codebook[d, j]
            acc += w * w
        end
        W2[j] = acc
    end
    return W2
end

"""Dense–sparse dot product: ⟨w, x_n⟩ for sparse column `n` (1-based feature indices)."""
@inline function dense_sparse_dot(codebook::AbstractMatrix{Float32}, neuron::Int,
                                  data::SparseMatrixCSC{<:Real}, n::Int)
    acc = 0.0f0
    @inbounds for k in nzrange(data, n)
        row = rowvals(data)[k]
        acc += codebook[row, neuron] * Float32(nonzeros(data)[k])
    end
    return acc
end

"""
Find BMU for sparse sample `n` using squared Euclidean distance
`x² + ‖w‖² − 2⟨w,x⟩` (matches C++ `get_bmu_coord`).
"""
function find_bmu_sparse!(bmu_x::AbstractVector{Int32}, bmu_y::AbstractVector{Int32},
                          codebook::AbstractMatrix{Float32}, data::SparseMatrixCSC{<:Real},
                          X2::AbstractVector{Float32}, W2::AbstractVector{Float32},
                          nSomX::Int, nSomY::Int, n::Int)
    mindist = typemax(Float32)
    best_x = 0
    best_y = 0
    @inbounds for som_y in 0:(nSomY - 1)
        for som_x in 0:(nSomX - 1)
            neuron = som_y * nSomX + som_x + 1
            dist = X2[n] + W2[neuron] - 2f0 * dense_sparse_dot(codebook, neuron, data, n)
            dist < 0f0 && (dist = 0f0)
            if (som_y == 0 && som_x == 0) || dist < mindist
                mindist = dist
                best_x = som_x
                best_y = som_y
            end
        end
    end
    bmu_x[n] = Int32(best_x)
    bmu_y[n] = Int32(best_y)
    return nothing
end

function find_all_bmus_sparse!(bmu_x, bmu_y, codebook, data, X2, W2, nSomX, nSomY)
    nVectors = size(data, 2)
    Threads.@threads for n in 1:nVectors
        find_bmu_sparse!(bmu_x, bmu_y, codebook, data, X2, W2, nSomX, nSomY, n)
    end
    return nothing
end

"""
One sparse epoch (C++ `trainOneEpochSparseCPU` non-MPI): in-place codebook update.
Only nonzero features of each sample accumulate into the numerator.
"""
function train_epoch_sparse!(codebook::AbstractMatrix{Float32},
                             data::SparseMatrixCSC{<:Real},
                             bmu_x::AbstractVector{Int32}, bmu_y::AbstractVector{Int32},
                             X2::AbstractVector{Float32}, W2::AbstractVector{Float32},
                             nSomX::Int, nSomY::Int, nDimensions::Int,
                             radius::Float32, scale::Float32,
                             compact_support::Bool, gaussian::Bool, std_coeff::Float32,
                             rectangular::Bool, toroid::Bool; only_bmus::Bool=false)
    codebook_sqnorms!(W2, codebook)
    find_all_bmus_sparse!(bmu_x, bmu_y, codebook, data, X2, W2, nSomX, nSomY)
    only_bmus && return nothing

    nNeurons = nSomX * nSomY
    nVectors = size(data, 2)
    # Parallel over neurons; each has a private numerator buffer (like C++ OpenMP).
    Threads.@threads for idx in 1:nNeurons
        som_y = (idx - 1) ÷ nSomX
        som_x = (idx - 1) % nSomX
        local_num = zeros(Float32, nDimensions)
        local_den = 0.0f0
        @inbounds for n in 1:nVectors
            bx = Int(bmu_x[n])
            by = Int(bmu_y[n])
            dist = map_distance(som_x, som_y, bx, by, nSomX, nSomY, rectangular, toroid)
            w = get_weight(dist, radius, scale, compact_support, gaussian, std_coeff)
            local_den += w
            for k in nzrange(data, n)
                row = rowvals(data)[k]
                local_num[row] += w * Float32(nonzeros(data)[k])
            end
        end
        if local_den != 0.0f0
            @inbounds for d in 1:nDimensions
                codebook[d, idx] = local_num[d] / local_den
            end
        end
    end
    return nothing
end

function train_som_sparse!(codebook::AbstractMatrix{Float32},
                           data::SparseMatrixCSC{<:Real},
                           nSomX::Int, nSomY::Int;
                           nEpoch::Int=10,
                           radius0::Float32=0.0f0,
                           radiusN::Float32=1.0f0,
                           radius_cooling::AbstractString="linear",
                           scale0::Float32=0.1f0,
                           scaleN::Float32=0.01f0,
                           scale_cooling::AbstractString="linear",
                           compact_support::Bool=true,
                           gaussian::Bool=true,
                           std_coeff::Float32=0.5f0,
                           rectangular::Bool=true,
                           toroid::Bool=false,
                           verbose::Int=0)
    nDimensions, nVectors = size(data)
    size(codebook, 1) == nDimensions || throw(DimensionMismatch("codebook/data dimensions"))
    size(codebook, 2) == nSomX * nSomY || throw(DimensionMismatch("codebook/map size"))

    r0 = radius0 == 0.0f0 ? Float32(min(nSomX, nSomY) / 2) : radius0
    rN = radiusN == 0.0f0 ? 1.0f0 : radiusN
    s0 = scale0 == 0.0f0 ? 0.1f0 : scale0
    sN = scaleN

    X2 = sparse_sample_sqnorms(data)
    W2 = Vector{Float32}(undef, nSomX * nSomY)
    bmu_x = zeros(Int32, nVectors)
    bmu_y = zeros(Int32, nVectors)

    N = Float32(nEpoch)
    for currentEpoch in 0:(nEpoch - 1)
        radius = if radius_cooling == "linear"
            linear_cooling(r0, rN, N, Float32(currentEpoch))
        elseif radius_cooling == "exponential"
            exponential_cooling(r0, rN, N, Float32(currentEpoch))
        else
            error("Unknown radius cooling: $radius_cooling")
        end
        scale = if scale_cooling == "linear"
            linear_cooling(s0, sN, N, Float32(currentEpoch))
        elseif scale_cooling == "exponential"
            exponential_cooling(s0, sN, N, Float32(currentEpoch))
        else
            error("Unknown scale cooling: $scale_cooling")
        end
        if verbose > 0
            @info "Sparse epoch $(currentEpoch + 1)/$nEpoch radius=$radius scale=$scale"
        end
        train_epoch_sparse!(codebook, data, bmu_x, bmu_y, X2, W2,
                            nSomX, nSomY, nDimensions,
                            radius, scale, compact_support, gaussian, std_coeff,
                            rectangular, toroid)
    end

    train_epoch_sparse!(codebook, data, bmu_x, bmu_y, X2, W2,
                        nSomX, nSomY, nDimensions,
                        rN, sN, compact_support, gaussian, std_coeff,
                        rectangular, toroid; only_bmus=true)

    return bmu_x, bmu_y
end
