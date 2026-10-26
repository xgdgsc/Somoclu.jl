# Training loop — ported from somoclu/src/training.cpp and denseCpuKernels.cpp

"""
Legacy Julia binding sent `gridtype="square"` to C++ as the string `"square"`.
The C++ kernels only special-case `"rectangular"`, so `"square"` used hexagonal
map distances. Keep that behaviour for numerical identity with Somoclu.jl tests.
`"rectangular"` selects true rectangular geometry (Python / CLI).
"""
function rectangular_grid(gridtype::AbstractString)
    if gridtype == "rectangular"
        return true
    elseif gridtype == "square" || gridtype == "hexagonal"
        return false
    else
        error("Unknown grid type: $gridtype (use \"square\", \"rectangular\", or \"hexagonal\")")
    end
end

function toroid_map(maptype::AbstractString)
    if maptype == "planar"
        return false
    elseif maptype == "toroid"
        return true
    else
        error("Unknown map type: $maptype")
    end
end

function initialize_codebook_random!(codebook::AbstractMatrix{Float32}, rng::AbstractRNG)
    nDim, nNeurons = size(codebook)
    @inbounds for j in 1:nNeurons
        for d in 1:nDim
            w = Int(0xFFF & rand(rng, UInt32)) - 0x800
            codebook[d, j] = Float32(w) / 4096.0f0
        end
    end
    return codebook
end

function initialize_codebook_pca!(codebook::AbstractMatrix{Float32}, data::AbstractMatrix{<:Real},
                                  ncolumns::Int, nrows::Int)
    nDimensions, nVectors = size(data)
    nNeurons = ncolumns * nrows
    size(codebook) == (nDimensions, nNeurons) || throw(DimensionMismatch("codebook size"))

    coord = zeros(Float32, nNeurons, 2)
    @inbounds for i in 1:nNeurons
        coord[i, 1] = (i - 1) ÷ ncolumns
        coord[i, 2] = (i - 1) % ncolumns
    end
    if nrows > 1 && ncolumns > 1
        coord ./= Float32[nrows - 1  ncolumns - 1]
    end
    coord .= 2f0 .* (coord .- 0.5f0)

    data32 = Float32.(data)
    me = mean(data32; dims=2)
    M = fit(PCA, data32 .- me; maxoutdim=2)
    eigval = principalvars(M)
    eigvec = projection(M)
    ncomp = size(eigvec, 2)
    norms = [norm(view(eigvec, :, i)) for i in 1:ncomp]
    scaled = similar(eigvec)
    @inbounds for i in 1:ncomp
        scaled[:, i] .= (eigvec[:, i] ./ norms[i]) .* eigval[i]
    end

    codebook .= me
    @inbounds for j in 1:nNeurons
        for i in 1:ncomp
            codebook[:, j] .+= coord[j, i] .* scaled[:, i]
        end
    end
    return codebook
end

function train_epoch!(backend, codebook, data, bmu_x, bmu_y, numerator, denominator,
                      nSomX::Int, nSomY::Int, nDimensions::Int,
                      radius::Float32, scale::Float32,
                      compact_support::Bool, gaussian::Bool, std_coeff::Float32,
                      rectangular::Bool, toroid::Bool; only_bmus::Bool=false)
    launch_find_bmus!(backend, bmu_x, bmu_y, codebook, data, nSomX, nSomY, nDimensions)
    only_bmus && return nothing
    launch_update_codebook!(backend, codebook, data, bmu_x, bmu_y, numerator, denominator,
                            nSomX, nSomY, nDimensions,
                            radius, scale, compact_support, gaussian, std_coeff,
                            rectangular, toroid)
    return nothing
end

function train_som!(codebook::AbstractMatrix{Float32}, data::AbstractMatrix{Float32},
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
                    verbose::Int=0,
                    rng::AbstractRNG=Random.default_rng())
    nDimensions, nVectors = size(data)
    size(codebook, 1) == nDimensions || throw(DimensionMismatch("codebook/data dimensions"))
    size(codebook, 2) == nSomX * nSomY || throw(DimensionMismatch("codebook/map size"))

    r0 = radius0 == 0.0f0 ? Float32(min(nSomX, nSomY) / 2) : radius0
    rN = radiusN == 0.0f0 ? 1.0f0 : radiusN
    s0 = scale0 == 0.0f0 ? 0.1f0 : scale0
    sN = scaleN

    backend = KernelAbstractions.get_backend(codebook)
    bmu_x = KernelAbstractions.zeros(backend, Int32, nVectors)
    bmu_y = KernelAbstractions.zeros(backend, Int32, nVectors)
    nNeurons = nSomX * nSomY
    numerator = KernelAbstractions.zeros(backend, Float32, nDimensions, nNeurons)
    denominator = KernelAbstractions.zeros(backend, Float32, nNeurons)

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
            @info "Epoch $(currentEpoch + 1)/$nEpoch radius=$radius scale=$scale"
        end
        train_epoch!(backend, codebook, data, bmu_x, bmu_y, numerator, denominator,
                     nSomX, nSomY, nDimensions,
                     radius, scale, compact_support, gaussian, std_coeff,
                     rectangular, toroid)
    end

    # Final BMU-only pass (matches C++ train())
    train_epoch!(backend, codebook, data, bmu_x, bmu_y, numerator, denominator,
                 nSomX, nSomY, nDimensions,
                 rN, sN, compact_support, gaussian, std_coeff,
                 rectangular, toroid; only_bmus=true)

    return bmu_x, bmu_y
end
