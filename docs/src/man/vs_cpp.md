# Compared to C++ Somoclu

Somoclu.jl reimplements the C++ [Somoclu](https://github.com/peterwittek/somoclu)
algorithms in pure Julia. There is **no** C++ / binary dependency.

Version numbering tracks the upstream library line (currently aligned with the
Python package’s `1.7.x` series).

## Feature matrix

| Feature | C++ Somoclu | Somoclu.jl |
|---------|-------------|------------|
| Dense batch training | `denseCpuKernels.cpp` | Yes — KernelAbstractions (`CPU()` / PoCL, CUDA, …) |
| Sparse training | `sparseCpuKernels.cpp` (`-k 2`, libsvm) | Yes — `SparseMatrixCSC` CPU path |
| Sparse GPU | No | No |
| Dense GPU | CUDA (`denseGpuKernels.cu`) | Yes — via KA (e.g. `CUDABackend()`), not the C++ `.cu` |
| MPI distributed | Optional `HAVE_MPI` | No (single process; Threads / KA only) |
| OpenMP | Yes (CPU kernels) | Threads on sparse; PoCL/KA on dense |
| Planar / toroid | Yes | Yes |
| Rectangular / hexagonal | Yes | Yes (+ legacy `"square"` → hex distances) |
| Gaussian / bubble | Yes | Yes |
| Radius / scale cooling | linear, exponential | Yes |
| Random init | Yes | Yes (same `0xFFF` formula) |
| PCA init | Python / Julia wrappers | Yes |
| U-matrix | `uMatrix.cpp` | Yes |
| BMUs | Yes | Yes (1-based, legacy axis swap) |
| Feature distances | euclidean, `norm-inf`, `norm-p` (dense CPU) | Yes on dense KA path (`Som(; vect_distance=...)`); sparse stays Euclidean |
| CLI binary | `somoclu` | No (library API only) |
| File IO (`.lrn`, libsvm, codebook dumps) | `io.cpp` | Yes — `read_dense_matrix`, `read_libsvm` / `write_libsvm`, `save_codebook` / `save_bmus` / `save_umatrix` / `save_som_dumps` |
| Training snapshots (`-s`) | Yes | No |
| Clustering API | Python `cluster()` (sklearn) | No (`som.clusters` only for plot colors) |
| Visualizations | Python matplotlib, etc. | PlotlyLight (`view_*`) |

## Sparse path details

- **Input:** `SparseMatrixCSC` of size `nDimensions × nVectors` (columns = samples).
- **Algorithm:** Same as C++ non-MPI `trainOneEpochSparseCPU`: BMU via
  ``‖x‖² + ‖w‖² − 2⟨w,x⟩``, then neighbourhood-weighted update using **only
  nonzero** features. Missing features contribute `0` to the numerator (so those
  codebook dimensions become `0` when the denominator is nonzero).
- **Codebook** remains dense.
- **`som.backend` is ignored** for sparse `train!` (CPU + `Threads.@threads`).
- Dense KA/PoCL and sparse CPU can diverge across epochs even on “fully dense”
  data because dense BMUs use `√ΣΔ²` while sparse BMUs use the squared formula;
  float tie-breaks differ. Prefer one path consistently.

```julia
using SparseArrays, Somoclu
X = sprand(Float32, 20_000, 1_000, 0.005)
som = Som(50, 40; gridtype="rectangular", compactsupport=true)
train!(som, X; epochs=15)
```

## Intentionally preserved quirks

- Internal map size: `nSomX = nrows`, `nSomY = ncolumns` (historical Julia wrapper).
- `gridtype="square"` uses **hexagonal** map distances (C++ only special-cases
  `"rectangular"`).
- Default `compactsupport=true` matches the historical Julia/Python wrapper
  (CLI default is false).

## Distances and file IO

```julia
som = Som(40, 30; vect_distance="norm-inf")   # or "norm-1", "norm-2.5", …
train!(som, data)

X = read_dense_matrix("data.lrn")             # also plain txt / .wts
S = read_libsvm("data.svm")
save_som_dumps("out", som)                    # out.wts / out.bm / out.umx
```

## Not planned as C++ clones

MPI and the `somoclu` CLI binary are omitted for now. MPI is feasible later with
`MPI.jl` (broadcast codebook / reduce numerator–denominator per epoch) if
multi-node scaling is needed.
