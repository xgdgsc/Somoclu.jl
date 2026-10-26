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

After training:

| Field | Shape | Meaning |
|-------|-------|---------|
| `som.codebook` | `nDim × (ncolumns*nrows)` | Neuron weight vectors |
| `som.bmus` | `2 × nVectors` | Best-matching unit coords (1-based) |
| `som.umatrix` | `nrows × ncolumns` | Unified distance matrix |

## GPU backends

Any KernelAbstractions backend works. Example with CUDA:

```julia
using CUDA, Somoclu
som = Som(50, 50; backend = CUDABackend())
train!(som, data)
```

## Numerical notes

Training follows Somoclu’s dense CPU kernel. The Python interface test vector is
reproduced to about `1e-5`. The legacy `"square"` grid uses hexagonal map
distances (C++ only special-cased `"rectangular"`); that quirk is preserved.
