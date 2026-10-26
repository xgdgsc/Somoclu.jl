"""Compare Python somoclu vs Julia Somoclu.jl on the same data / C++ geometry."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from somoclu import Somoclu

outdir = Path(__file__).resolve().parent / "figures" / "compare"
data_jl = np.loadtxt(outdir / "data_julia.csv", delimiter=",", dtype=np.float32)
# Julia: (n_dim, n_vec) → Python: (n_vec, n_dim)
data = data_jl.T.copy()
umatrix_jl = np.loadtxt(outdir / "umatrix_julia.csv", delimiter=",", dtype=np.float32)
codebook_jl = np.loadtxt(outdir / "codebook_julia.csv", delimiter=",", dtype=np.float32)
bmus_jl = np.loadtxt(outdir / "bmus_julia.csv", delimiter=",", dtype=np.int32)
colors = np.loadtxt(outdir / "colors.csv", delimiter=",", dtype=np.int32)

# Match Julia's C++ layout: nSomX=nrows_user=30, nSomY=ncolumns_user=40
# Python Somoclu(n_columns, n_rows) → C++ nSomX=n_columns, nSomY=n_rows
n_columns_py, n_rows_py = 30, 40
n_dim = data.shape[1]
codebook0 = np.zeros((n_rows_py * n_columns_py, n_dim), dtype=np.float32)

som = Somoclu(
    n_columns_py,
    n_rows_py,
    initialcodebook=codebook0,
    maptype="planar",
    gridtype="rectangular",
    compactsupport=False,
    initialization=None,
)
som.train(data, epochs=10)

# Align umatrices:
# Julia umatrix shape (nrows, ncolumns) = (nSomX, nSomY) = (30, 40), index [sx+1, sy+1]
# Python umatrix shape (n_rows, n_columns) = (nSomY, nSomX) = (40, 30), index [sy, sx]
# → Julia[sx, sy] == Python[sy, sx] → Julia ≈ Python.T
umatrix_py_aligned = som.umatrix.T.astype(np.float32)

diff = np.abs(umatrix_jl - umatrix_py_aligned)
print(f"Python umatrix shape={som.umatrix.shape}  Julia={umatrix_jl.shape}")
print(f"Aligned diff: max={diff.max():.6g}  mean={diff.mean():.6g}  sum={diff.sum():.6g}")

# Codebook compare: Julia (n_dim, nNeurons), neuron = sy*nSomX + sx
# Python codebook (n_rows, n_columns, n_dim) = (nSomY, nSomX, n_dim)
cb_py = som.codebook.reshape(n_rows_py, n_columns_py, n_dim)
# Build Julia-ordered codebook from Python: col j = sy*30+sx → jl[:, j] vs py[sy, sx, :]
nSomX, nSomY = 30, 40
cb_py_as_jl = np.zeros_like(codebook_jl)
for sy in range(nSomY):
    for sx in range(nSomX):
        j = sy * nSomX + sx
        cb_py_as_jl[:, j] = cb_py[sy, sx, :]
cb_diff = np.abs(codebook_jl - cb_py_as_jl)
print(f"Codebook diff: max={cb_diff.max():.6g}  mean={cb_diff.mean():.6g}")

# --- Side-by-side U-matrix figure ---
cmap = cm.Spectral_r
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

im0 = axes[0].imshow(umatrix_jl, aspect="auto", cmap=cmap, interpolation="nearest")
axes[0].set_title("Julia Somoclu.jl")
axes[0].set_xlabel("column (nSomY)")
axes[0].set_ylabel("row (nSomX)")
# BMUs: Julia bmus[1]=col=sy+1, bmus[2]=row=sx+1 → plot (col-1, row-1) on imshow
axes[0].scatter(bmus_jl[:, 0] - 1, bmus_jl[:, 1] - 1, c=colors, s=12, cmap="tab10",
                 edgecolors="white", linewidths=0.3)

im1 = axes[1].imshow(umatrix_py_aligned, aspect="auto", cmap=cmap, interpolation="nearest")
axes[1].set_title("Python somoclu (aligned)")
axes[1].set_xlabel("column (nSomY)")
axes[1].set_ylabel("row (nSomX)")
# Python bmus are (n_vectors, 2) as (column, row) in Python's n_columns×n_rows coords
# Python: bmu[:,0]=som_x in 0..n_columns-1=0..29, bmu[:,1]=som_y in 0..39
# On aligned (Julia) view: x=sy=bmu[:,1], y=sx=bmu[:,0]
bmus_py = som.bmus
axes[1].scatter(bmus_py[:, 1], bmus_py[:, 0], c=colors, s=12, cmap="tab10",
                edgecolors="white", linewidths=0.3)

im2 = axes[2].imshow(diff, aspect="auto", cmap="magma", interpolation="nearest")
axes[2].set_title(f"|Julia − Python|  max={diff.max():.2e}")
axes[2].set_xlabel("column")
axes[2].set_ylabel("row")

fig.colorbar(im0, ax=axes[0], fraction=0.046)
fig.colorbar(im1, ax=axes[1], fraction=0.046)
fig.colorbar(im2, ax=axes[2], fraction=0.046)
fig.suptitle("U-matrix comparison (same data, matched C++ nSomX×nSomY = 30×40)", fontsize=12)
out_png = outdir / "umatrix_side_by_side.png"
fig.savefig(out_png, dpi=160)
print(f"wrote {out_png}")

# Also save Python raw plots for reference
fig2, ax = plt.subplots(figsize=(5, 5.5))
ax.imshow(som.umatrix, aspect="auto", cmap=cmap)
ax.set_title("Python raw umatrix (n_rows×n_columns)")
fig2.savefig(outdir / "umatrix_python_raw.png", dpi=140)
plt.close("all")

np.savetxt(outdir / "umatrix_python_aligned.csv", umatrix_py_aligned, delimiter=",")
np.savetxt(outdir / "diff_umatrix.csv", diff, delimiter=",")
print("done")
