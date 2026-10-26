"""
    Somoclu

Pure-Julia port of [Somoclu](https://github.com/peterwittek/somoclu) self-organizing maps,
using KernelAbstractions.jl for dense CPU (PoCL) / GPU backends, plus a sparse CPU path
matching C++ `sparseCpuKernels.cpp`.

The public API mirrors the historical Somoclu.jl C++ wrapper (`Som`, `train!`) plus
Python-style visualization helpers via PlotlyLight (`view_umatrix`, etc.).
"""
module Somoclu

using KernelAbstractions
using LinearAlgebra
using MultivariateStats: PCA, fit, principalvars, projection
using PlotlyLight
using Random
using SparseArrays
using Statistics

export Som, train!
export codebook_plane, get_surface_state
export view_umatrix, view_component_planes, view_activation_map, view_similarity_matrix

include("distances.jl")
include("neighborhood.jl")
include("kernels.jl")
include("umatrix.jl")
include("train.jl")
include("sparse.jl")

"""
    Som(ncolumns, nrows; kwargs...)

Self-organizing map of size `ncolumns` × `nrows`.

# Arguments
- `compactsupport::Bool=true`: Cut off Gaussian updates beyond the training radius.
- `gridtype::String="square"`: `"square"` (legacy; hexagonal distances), `"rectangular"`,
  or `"hexagonal"`.
- `initialcodebook`: Optional `Matrix{Float32}` of size `(nDimensions, ncolumns*nrows)`.
- `initialization::String="random"`: `"random"` or `"pca"`.
- `maptype::String="planar"`: `"planar"` or `"toroid"`.
- `neighborhood::String="gaussian"`: `"gaussian"` or `"bubble"`.
- `stdcoeff::Float32=0.5f0`: Gaussian neighborhood coefficient.
- `verbose::Integer=0`: Verbosity level 0–2.
- `backend`: KernelAbstractions backend (default `CPU()`).
"""
mutable struct Som
    ncolumns::Int
    nrows::Int
    maptype::String
    gridtype::String
    compactsupport::Bool
    neighborhood::String
    stdcoeff::Float32
    verbose::Int
    initialization::String
    backend::KernelAbstractions.Backend
    codebook::Union{Nothing,AbstractMatrix{Float32}}
    bmus::Union{Nothing,Matrix{Int32}}
    umatrix::Union{Nothing,Matrix{Float32}}
    data::Union{Nothing,AbstractMatrix}
    activation_map::Union{Nothing,Matrix{Float64}}
    clusters::Union{Nothing,Matrix{Int}}

    function Som(ncolumns::Integer, nrows::Integer;
                 maptype::String="planar",
                 gridtype::String="square",
                 compactsupport::Bool=true,
                 neighborhood::String="gaussian",
                 stdcoeff::Real=0.5,
                 verbose::Integer=0,
                 initialization::String="random",
                 initialcodebook=nothing,
                 backend=CPU())
        maptype in ("planar", "toroid") || error("Unknown map type!")
        gridtype in ("square", "rectangular", "hexagonal") || error("Unknown grid type!")
        neighborhood in ("gaussian", "bubble") || error("Unknown neighborhood function!")
        initialization in ("random", "pca") || error("Unknown initialization!")
        0 <= verbose <= 2 || error("Unsupported verbosity level!")

        codebook = if initialcodebook === nothing
            nothing
        else
            Matrix{Float32}(initialcodebook)
        end
        new(Int(ncolumns), Int(nrows), maptype, gridtype, compactsupport,
            neighborhood, Float32(stdcoeff), Int(verbose), initialization,
            backend, codebook, nothing, nothing, nothing, nothing, nothing)
    end
end

"""
    train!(som::Som, data; kwargs...)

Train `som` on `data` of size `nDimensions × nVectors` (columns are samples).

- Dense `AbstractMatrix`: KernelAbstractions path (`som.backend`, default PoCL `CPU()`).
- `SparseMatrixCSC`: C++-compatible sparse CPU kernel (ignores `som.backend`; codebook
  stays dense). Prefer this for high-dimensional sparse inputs.

# Arguments
- `epochs::Integer=10`
- `radius0::Float32=0` — `0` means `min(ncolumns, nrows)/2`
- `radiusN::Float32=1`
- `radiuscooling::String="linear"` — `"linear"` or `"exponential"`
- `scale0::Float32=0.1`
- `scaleN::Float32=0.01`
- `scalecooling::String="linear"` — `"linear"` or `"exponential"`
"""
function train!(som::Som, data::AbstractMatrix{<:Real};
                epochs::Integer=10,
                radius0::Real=0,
                radiusN::Real=1,
                radiuscooling::String="linear",
                scale0::Real=0.1,
                scaleN::Real=0.01,
                scalecooling::String="linear",
                rng::AbstractRNG=Random.default_rng())
    _check_cooling(radiuscooling, scalecooling)
    data32 = Matrix{Float32}(data)
    nDimensions, nVectors = size(data32)
    nSomX, nSomY, nNeurons = _map_shape(som)
    codebook_host = _init_codebook(som, data32, nDimensions, nNeurons, rng)

    backend = som.backend
    codebook = adapt_storage(backend, codebook_host)
    data_dev = adapt_storage(backend, data32)
    rectangular = rectangular_grid(som.gridtype)
    toroid = toroid_map(som.maptype)
    gaussian = som.neighborhood == "gaussian"

    bmu_x, bmu_y = train_som!(codebook, data_dev, nSomX, nSomY;
                              nEpoch=Int(epochs),
                              radius0=Float32(radius0),
                              radiusN=Float32(radiusN),
                              radius_cooling=radiuscooling,
                              scale0=Float32(scale0),
                              scaleN=Float32(scaleN),
                              scale_cooling=scalecooling,
                              compact_support=som.compactsupport,
                              gaussian=gaussian,
                              std_coeff=som.stdcoeff,
                              rectangular=rectangular,
                              toroid=toroid,
                              verbose=som.verbose,
                              rng=rng)

    _finish_train!(som, Array(codebook), Array(bmu_x), Array(bmu_y), data32,
                   nSomX, nSomY, nDimensions, nVectors, rectangular, toroid)
    return nothing
end

function train!(som::Som, data::SparseMatrixCSC{<:Real};
                epochs::Integer=10,
                radius0::Real=0,
                radiusN::Real=1,
                radiuscooling::String="linear",
                scale0::Real=0.1,
                scaleN::Real=0.01,
                scalecooling::String="linear",
                rng::AbstractRNG=Random.default_rng())
    _check_cooling(radiuscooling, scalecooling)
    # Keep sparse structure; cast values to Float32 view via SparseMatrixCSC{Float32}
    data_f32 = SparseMatrixCSC{Float32,Int}(data)
    nDimensions, nVectors = size(data_f32)
    nSomX, nSomY, nNeurons = _map_shape(som)

    # PCA init densifies once (same as using dense features for PCA only)
    init_data = som.initialization == "pca" ? Matrix{Float32}(data_f32) : data_f32
    codebook_host = _init_codebook(som, init_data, nDimensions, nNeurons, rng)

    rectangular = rectangular_grid(som.gridtype)
    toroid = toroid_map(som.maptype)
    gaussian = som.neighborhood == "gaussian"

    bmu_x, bmu_y = train_som_sparse!(codebook_host, data_f32, nSomX, nSomY;
                                     nEpoch=Int(epochs),
                                     radius0=Float32(radius0),
                                     radiusN=Float32(radiusN),
                                     radius_cooling=radiuscooling,
                                     scale0=Float32(scale0),
                                     scaleN=Float32(scaleN),
                                     scale_cooling=scalecooling,
                                     compact_support=som.compactsupport,
                                     gaussian=gaussian,
                                     std_coeff=som.stdcoeff,
                                     rectangular=rectangular,
                                     toroid=toroid,
                                     verbose=som.verbose)

    _finish_train!(som, codebook_host, bmu_x, bmu_y, data_f32,
                   nSomX, nSomY, nDimensions, nVectors, rectangular, toroid)
    return nothing
end

function _check_cooling(radiuscooling, scalecooling)
    radiuscooling in ("linear", "exponential") || error("Unknown radius cooling")
    scalecooling in ("linear", "exponential") || error("Unknown scale cooling")
    return nothing
end

_map_shape(som::Som) = (som.nrows, som.ncolumns, som.ncolumns * som.nrows)

function _init_codebook(som::Som, data, nDimensions::Int, nNeurons::Int, rng::AbstractRNG)
    if som.codebook !== nothing && size(som.codebook) == (nDimensions, nNeurons)
        return Matrix{Float32}(som.codebook)
    elseif som.initialization == "random"
        codebook = Matrix{Float32}(undef, nDimensions, nNeurons)
        return initialize_codebook_random!(codebook, rng)
    elseif som.initialization == "pca"
        codebook = Matrix{Float32}(undef, nDimensions, nNeurons)
        return initialize_codebook_pca!(codebook, Matrix{Float32}(data), som.ncolumns, som.nrows)
    else
        error("Unknown initialization method")
    end
end

function _finish_train!(som::Som, codebook_host, bmu_x_h, bmu_y_h, data_store,
                        nSomX, nSomY, nDimensions, nVectors, rectangular, toroid)
    nNeurons = nSomX * nSomY
    umatrix_flat = Vector{Float32}(undef, nNeurons)
    calculate_umatrix!(umatrix_flat, codebook_host, nSomX, nSomY, nDimensions,
                       rectangular, toroid)
    som.codebook = codebook_host
    som.data = data_store
    som.activation_map = nothing
    som.bmus = Matrix{Int32}(undef, 2, nVectors)
    @inbounds for n in 1:nVectors
        som.bmus[1, n] = Int32(bmu_y_h[n] + 1)
        som.bmus[2, n] = Int32(bmu_x_h[n] + 1)
    end
    som.umatrix = reshape(umatrix_flat, nSomX, nSomY)
    return nothing
end

"""Copy `A` onto storage for `backend` (PoCL / GPU device arrays)."""
function adapt_storage(backend::KernelAbstractions.Backend, A::AbstractArray{Float32})
    B = KernelAbstractions.allocate(backend, Float32, size(A)...)
    copyto!(B, A)
    return B
end

include("plot.jl")

end # module
