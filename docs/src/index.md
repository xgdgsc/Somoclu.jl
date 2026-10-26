# Somoclu.jl

```@docs
Somoclu
```

Pure-Julia port of [Somoclu](https://github.com/peterwittek/somoclu) — massively parallel
self-organizing maps — using
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
for CPU (PoCL) and GPU backends. No C++ dependency.

Version **1.7.6** matches the current [Python `somoclu`](https://pypi.org/project/somoclu/) release.

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

## Features

- Dense batch SOM training (same update rule as Somoclu’s dense CPU kernel)
- Planar / toroid topologies
- Rectangular / hexagonal grids (legacy `"square"` uses hexagonal distances)
- Gaussian / bubble neighbourhoods
- Random or PCA codebook initialization
- U-matrix and best-matching units (BMUs)
- PlotlyLight visualizations (`view_umatrix`, …)

## Citation

Peter Wittek, Shi Chao Gao, Ik Soo Lim, Li Zhao (2017).
Somoclu: An Efficient Parallel Library for Self-Organizing Maps.
*Journal of Statistical Software*, 78(9), 1–21.
[https://doi.org/10.18637/jss.v078.i09](https://doi.org/10.18637/jss.v078.i09)

## Package overview

```@contents
Pages = ["man/getting_started.md", "man/visualization.md", "api.md"]
Depth = 2
```
