# Getting started

## Creating a map

```julia
using Somoclu

som = Som(
    40, 30;                    # ncolumns × nrows
    maptype = "planar",        # or "toroid"
    gridtype = "rectangular",  # "rectangular", "hexagonal", or legacy "square"
    neighborhood = "gaussian", # or "bubble"
    initialization = "random", # or "pca"
    compactsupport = true,
)
```

Internal training uses the historical Julia axis convention
(`nSomX = nrows`, `nSomY = ncolumns`) so results stay aligned with the old
C++ wrapper and Python package when map sizes and options match.

## Training

`data` must be `nDimensions × nVectors` (columns are samples):

```julia
data = rand(Float32, 3, 500)
train!(som, data;
       epochs = 20,
       radius0 = 0,      # 0 → min(ncolumns, nrows)/2
       radiusN = 1,
       scale0 = 0.1f0,
       scaleN = 0.01f0,
)
```

### Sparse data

Pass a `SparseMatrixCSC` to use the C++-compatible sparse CPU kernel
(`sparseCpuKernels.cpp`):

- Layout: `nDimensions × nVectors` (each **column** is one sample)
- Codebook stays **dense**; neighbourhood update only touches nonzero features
- Runs on CPU with `Threads` — **`som.backend` is ignored**
- Prefer this for high-dimensional sparse inputs (e.g. bag-of-words)

```julia
using SparseArrays
data = sprand(Float32, 10_000, 500, 0.01)  # features × samples
som = Som(40, 30; gridtype="rectangular")
train!(som, data; epochs=10)
```

See [Compared to C++](vs_cpp.md) for how this maps to the upstream `-k 2` path.

After training:

| Field | Shape | Meaning |
|-------|-------|---------|
| `som.codebook` | `nDim × (ncolumns*nrows)` | Neuron weight vectors |
| `som.bmus` | `2 × nVectors` | Best-matching unit coords (1-based) |
| `som.umatrix` | `nrows × ncolumns` | Unified distance matrix |

## KernelAbstractions backends

`Som` takes any [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl)
backend via `backend=...`. Default is `CPU()` (multithreaded / PoCL).
Install the matching GPU package first; APIs below follow current KA conventions.

### CPU (default)

```julia
using Somoclu
som = Som(50, 50)                          # backend = CPU()
# som = Som(50, 50; backend = CPU())
train!(som, data)
```

### NVIDIA — CUDA.jl

```julia
using CUDA, Somoclu
som = Som(50, 50; backend = CUDABackend())
train!(som, data)
```

### AMD — AMDGPU.jl

```julia
using AMDGPU, Somoclu
som = Som(50, 50; backend = ROCBackend())
train!(som, data)
```

### Intel — oneAPI.jl

```julia
using oneAPI, Somoclu
som = Som(50, 50; backend = oneAPIBackend())
train!(som, data)
```

### Apple Silicon — Metal.jl

```julia
using Metal, Somoclu
som = Som(50, 50; backend = MetalBackend())
train!(som, data)
```

Codebook / data are moved onto the chosen backend during `train!`; results
(`som.codebook`, `som.bmus`, `som.umatrix`) are copied back to host arrays.

## Numerical notes

Dense training follows Somoclu’s dense CPU kernel; the Python interface test
vector is reproduced to about `1e-5`. Sparse training follows
`sparseCpuKernels.cpp` (squared-Euclidean BMUs). The legacy `"square"` grid uses
hexagonal map distances (C++ only special-cased `"rectangular"`); that quirk is
preserved.
