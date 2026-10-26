# Somoclu.jl

[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://xgdgsc.github.io/Somoclu.jl/stable)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://xgdgsc.github.io/Somoclu.jl/dev)
[![CI](https://github.com/xgdgsc/Somoclu.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/xgdgsc/Somoclu.jl/actions/workflows/CI.yml)

Pure-Julia port of [Somoclu](https://github.com/peterwittek/somoclu) — massively parallel
self-organizing maps — using [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
for CPU (PoCL) and GPU backends. No C++ dependency.

Version **1.7.6** matches the current [Python `somoclu`](https://pypi.org/project/somoclu/) release.

**Documentation:** [stable](https://xgdgsc.github.io/Somoclu.jl/stable) · [dev](https://xgdgsc.github.io/Somoclu.jl/dev)

## Installation

Once registered in the General registry:

```julia
using Pkg
Pkg.add("Somoclu")
```

Until then, install from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/xgdgsc/Somoclu.jl")
```

Requires Julia **1.13+**.

## Features

- Dense batch SOM training (same update rule as Somoclu’s dense CPU kernel)
- Planar / toroid topologies
- Rectangular / hexagonal grids (plus legacy `"square"` = hexagonal distances)
- Gaussian / bubble neighbourhoods
- Random or PCA codebook initialization
- U-matrix and best-matching units (BMUs)
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

### Visualization (PlotlyLight)

```julia
view_umatrix(som; bestmatches=true, colorbar=true)
view_component_planes(som; dimensions=1:2)
view_activation_map(som; data_index=1)
view_similarity_matrix(som)

# save HTML
view_umatrix(som; filename="umatrix.html")
```

### GPU (CUDA.jl example)

```julia
using CUDA, Somoclu
som = Som(40, 30; backend=CUDABackend())
train!(som, data)
```

## Numerical compatibility

Training numerics follow the Somoclu C++ dense CPU kernel
(`denseCpuKernels.cpp`, `mapDistanceFunctions.cpp`, `training.cpp`, `uMatrix.cpp`).
The Python interface test vector is reproduced exactly (to `1e-5`).

The historical Julia wrapper’s axis swap (`nSomX = nrows`, `nSomY = ncolumns`) and
`"square"` → hexagonal map-distance quirk (C++ only special-cased `"rectangular"`) are
preserved.

## Citation

Peter Wittek, Shi Chao Gao, Ik Soo Lim, Li Zhao (2017). Somoclu: An Efficient Parallel
Library for Self-Organizing Maps. *Journal of Statistical Software*, 78(9), 1–21.
https://doi.org/10.18637/jss.v078.i09

## License

MIT — see [LICENSE](LICENSE).
