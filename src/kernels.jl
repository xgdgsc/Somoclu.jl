# KernelAbstractions kernels for BMU search and batch codebook update.
# Numerics follow somoclu denseCpuKernels.cpp (non-MPI path).
# dist_kind: 0=euclidean, 1=norm-inf, 2=norm-p (p passed separately).

@kernel inbounds=true function find_bmu_kernel!(bmu_x, bmu_y, @Const(codebook), @Const(data),
                                                nSomX::Int, nSomY::Int, nDimensions::Int,
                                                dist_kind::Int, p::Float32)
    n = @index(Global, Linear)
    nVectors = size(data, 2)
    if n <= nVectors
        mindist = typemax(Float32)
        best_x = 0
        best_y = 0
        for som_y in 0:(nSomY - 1)
            for som_x in 0:(nSomX - 1)
                neuron = som_y * nSomX + som_x + 1
                dist = 0.0f0
                if dist_kind == 1
                    m = 0.0f0
                    for d in 1:nDimensions
                        Δ = abs(codebook[d, neuron] - data[d, n])
                        Δ > m && (m = Δ)
                    end
                    dist = m
                elseif dist_kind == 2
                    # Match C++ NormPDistance (double accumulator)
                    acc = 0.0
                    for d in 1:nDimensions
                        acc += abs(Float64(codebook[d, neuron]) - Float64(data[d, n]))^Float64(p)
                    end
                    dist = Float32(acc^(1.0 / Float64(p)))
                else
                    for d in 1:nDimensions
                        Δ = codebook[d, neuron] - data[d, n]
                        dist += Δ * Δ
                    end
                    dist = sqrt(dist)
                end
                if (som_y == 0 && som_x == 0) || dist < mindist
                    mindist = dist
                    best_x = som_x
                    best_y = som_y
                end
            end
        end
        bmu_x[n] = best_x
        bmu_y[n] = best_y
    end
end

@kernel inbounds=true function accumulate_update_kernel!(numerator, denominator,
                                                         @Const(data), @Const(bmu_x), @Const(bmu_y),
                                                         nSomX::Int, nSomY::Int, nDimensions::Int,
                                                         radius::Float32, scale::Float32,
                                                         compact_support::Bool, gaussian::Bool,
                                                         std_coeff::Float32,
                                                         rectangular::Bool, toroid::Bool)
    idx = @index(Global, Linear)
    nNeurons = nSomX * nSomY
    if idx <= nNeurons
        som_y = (idx - 1) ÷ nSomX
        som_x = (idx - 1) % nSomX
        nVectors = size(data, 2)
        denom = 0.0f0
        for d in 1:nDimensions
            numerator[d, idx] = 0.0f0
        end
        for n in 1:nVectors
            bx = Int(bmu_x[n])
            by = Int(bmu_y[n])
            dist = map_distance(som_x, som_y, bx, by, nSomX, nSomY, rectangular, toroid)
            w = get_weight(dist, radius, scale, compact_support, gaussian, std_coeff)
            denom += w
            for d in 1:nDimensions
                numerator[d, idx] = numerator[d, idx] + w * data[d, n]
            end
        end
        denominator[idx] = denom
    end
end

@kernel inbounds=true function apply_update_kernel!(codebook, @Const(numerator), @Const(denominator),
                                                    nDimensions::Int)
    idx = @index(Global, Linear)
    if idx <= size(codebook, 2)
        denom = denominator[idx]
        if denom != 0.0f0
            for d in 1:nDimensions
                codebook[d, idx] = numerator[d, idx] / denom
            end
        end
    end
end

function launch_find_bmus!(backend, bmu_x, bmu_y, codebook, data, nSomX, nSomY, nDimensions;
                           dist_kind::Int=0, p::Float32=2.0f0)
    kernel = find_bmu_kernel!(backend)
    kernel(bmu_x, bmu_y, codebook, data, Int(nSomX), Int(nSomY), Int(nDimensions),
           Int(dist_kind), Float32(p);
           ndrange = size(data, 2))
    KernelAbstractions.synchronize(backend)
    return nothing
end

function launch_update_codebook!(backend, codebook, data, bmu_x, bmu_y, numerator, denominator,
                                 nSomX, nSomY, nDimensions,
                                 radius, scale, compact_support, gaussian, std_coeff,
                                 rectangular, toroid)
    nNeurons = nSomX * nSomY
    acc = accumulate_update_kernel!(backend)
    acc(numerator, denominator, data, bmu_x, bmu_y,
        Int(nSomX), Int(nSomY), Int(nDimensions),
        Float32(radius), Float32(scale), compact_support, gaussian, Float32(std_coeff),
        rectangular, toroid;
        ndrange = nNeurons)
    KernelAbstractions.synchronize(backend)

    apply = apply_update_kernel!(backend)
    apply(codebook, numerator, denominator, Int(nDimensions); ndrange = nNeurons)
    KernelAbstractions.synchronize(backend)
    return nothing
end
