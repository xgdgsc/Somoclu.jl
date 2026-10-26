# Somoclu.jl

[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://xgdgsc.github.io/Somoclu.jl/stable)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://xgdgsc.github.io/Somoclu.jl/dev)
[![CI](https://github.com/xgdgsc/Somoclu.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/xgdgsc/Somoclu.jl/actions/workflows/CI.yml)

Pure-Julia port of [Somoclu](https://github.com/peterwittek/somoclu) — massively parallel
self-organizing maps — using [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
for dense CPU (PoCL) / GPU backends, plus a sparse CPU path matching C++
`sparseCpuKernels.cpp`. No C++ dependency.

**Documentation:** [stable](https://xgdgsc.github.io/Somoclu.jl/stable) · [dev](https://xgdgsc.github.io/Somoclu.jl/dev) · [vs C++](https://xgdgsc.github.io/Somoclu.jl/dev/man/vs_cpp/)

## Installation

```julia
using Pkg
Pkg.add("Somoclu")   # after General registry merge
# or:
Pkg.add(url="https://github.com/xgdgsc/Somoclu.jl")
```

Requires Julia **1.13+**.

## Features

- Dense batch SOM training (KernelAbstractions; same rule as Somoclu’s dense CPU kernel)
- Sparse CPU training (`SparseMatrixCSC`; same rule as C++ `sparseCpuKernels.cpp`)
- Planar / toroid topologies; rectangular / hexagonal grids (legacy `"square"` → hex distances)
- Gaussian / bubble neighbourhoods; linear / exponential cooling
- Random or PCA codebook initialization
- U-matrix and BMUs
- PlotlyLight visualizations (`view_umatrix`, `view_component_planes`, …)

## Usage

```julia
using Somoclu

ncolumns, nrows = 40, 30
data = rand(Float32, 2, 150)
som = Som(ncolumns, nrows; maptype="toroid", gridtype="rectangular")
train!(som, data; epochs=10)

som.codebook  # nDimensions × (ncolumns*nrows)
som.bmus      # 2 × nVectors (1-based)
som.umatrix   # nrows × ncolumns
```

### Sparse data

```julia
using SparseArrays, Somoclu
data = sprand(Float32, 10_000, 500, 0.01)  # features × samples
som = Som(40, 30; gridtype="rectangular")
train!(som, data; epochs=10)   # CPU sparse kernel; codebook stays dense
```

### Visualization (PlotlyLight)

```julia
view_umatrix(som; bestmatches=true, colorbar=true)
view_component_planes(som; dimensions=1:2)
view_activation_map(som; data_index=1)
view_similarity_matrix(som)
view_umatrix(som; filename="umatrix.html")
```

### GPU (CUDA.jl example; dense only)

```julia
using CUDA, Somoclu
som = Som(40, 30; backend=CUDABackend())
train!(som, data)
```

## Compared to C++ Somoclu

| | C++ | Somoclu.jl |
|--|-----|------------|
| Dense training | CPU (+ optional CUDA) | KA: PoCL `CPU()`, CUDA, AMDGPU, oneAPI, Metal |
| Sparse training | CPU (`-k 2` / libsvm) | `SparseMatrixCSC` CPU (+ Threads) |
| MPI | Optional | **No** |
| CLI / file IO (`.lrn`, libsvm, dumps) | Yes | **No** (in-memory) |
| `norm-inf` / `norm-p` distances | Dense CPU | **No** (Euclidean only) |
| Snapshots / `cluster()` | CLI / Python | **No** |
| Plotly `view_*` | — | **Yes** |

Dense CPU numerics match the C++/Python test vectors (`≈1e-5`). Legacy Julia quirks
kept: `nSomX = nrows`, `nSomY = ncolumns`, and `"square"` → hexagonal map distances.

Full table: [Compared to C++](https://xgdgsc.github.io/Somoclu.jl/dev/man/vs_cpp/).

## Citation

Peter Wittek, Shi Chao Gao, Ik Soo Lim, Li Zhao (2017). Somoclu: An Efficient Parallel
Library for Self-Organizing Maps. *Journal of Statistical Software*, 78(9), 1–21.
https://doi.org/10.18637/jss.v078.i09

## License

MIT — see [LICENSE](LICENSE).
