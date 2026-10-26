# Somoclu.jl

```@docs
Somoclu
```

Pure-Julia port of [Somoclu](https://github.com/peterwittek/somoclu) — massively parallel
self-organizing maps — using
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
for dense CPU (PoCL) / GPU backends, plus sparse CPU training. No C++ dependency.

See [Compared to C++](man/vs_cpp.md) for a feature matrix versus the upstream library.

## Installation

```julia
using Pkg
Pkg.add("Somoclu")          # after General registry merge
# or:
Pkg.add(url="https://github.com/xgdgsc/Somoclu.jl")
```

Requires Julia **1.13+**.

## Quick example

```julia
using Somoclu, Random
Random.seed!(42)

data = rand(Float32, 2, 200)
som = Som(40, 30; maptype="toroid", gridtype="rectangular")
train!(som, data; epochs=10)

som.umatrix   # nrows × ncolumns
som.bmus      # 2 × nVectors (1-based)
som.codebook  # nDimensions × (ncolumns * nrows)
```

Sparse:

```julia
using SparseArrays, Somoclu
train!(Som(40, 30; gridtype="rectangular"), sprand(Float32, 5000, 200, 0.02); epochs=10)
```

## Features

- Dense batch SOM (KernelAbstractions; same rule as C++ dense CPU kernel)
- Sparse CPU SOM (`SparseMatrixCSC`; same rule as C++ `sparseCpuKernels.cpp`)
- Planar / toroid; rectangular / hexagonal (legacy `"square"` → hex distances)
- Gaussian / bubble; random or PCA init; U-matrix / BMUs
- PlotlyLight visualizations (`view_umatrix`, …)

## Citation

Peter Wittek, Shi Chao Gao, Ik Soo Lim, Li Zhao (2017).
Somoclu: An Efficient Parallel Library for Self-Organizing Maps.
*Journal of Statistical Software*, 78(9), 1–21.
[https://doi.org/10.18637/jss.v078.i09](https://doi.org/10.18637/jss.v078.i09)

## Package overview

```@contents
Pages = ["man/getting_started.md", "man/visualization.md", "man/vs_cpp.md", "api.md"]
Depth = 2
```
