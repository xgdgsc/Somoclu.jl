# Generate shared data + Julia reference, then call Python for side-by-side plots.
using Somoclu
using Random
using DelimitedFiles

outdir = joinpath(@__DIR__, "figures", "compare")
mkpath(outdir)

Random.seed!(123)
ncolumns_user, nrows_user = 40, 30   # Julia API: Som(ncolumns, nrows)
# C++ sees nSomX=nrows, nSomY=ncolumns → 30 × 40
nv = 50
c1 = rand(Float32, 2, nv) ./ 5f0
c2 = rand(Float32, 2, nv) ./ 5f0 .+ Float32[0.2; 0.5]
c3 = rand(Float32, 2, nv) ./ 5f0 .+ Float32[0.4; 0.1]
data = hcat(c1, c2, c3)  # (2, 150) Julia layout
colors = vcat(fill(0, nv), fill(1, nv), fill(2, nv))  # class ids for BMU colors

# Deterministic zeros init
codebook0 = zeros(Float32, 2, ncolumns_user * nrows_user)
som = Som(ncolumns_user, nrows_user; maptype="planar", gridtype="rectangular",
          compactsupport=false, initialcodebook=copy(codebook0))
train!(som, data; epochs=10)

writedlm(joinpath(outdir, "data_julia.csv"), data, ',')
writedlm(joinpath(outdir, "umatrix_julia.csv"), som.umatrix, ',')
writedlm(joinpath(outdir, "codebook_julia.csv"), som.codebook, ',')
writedlm(joinpath(outdir, "bmus_julia.csv"), Int.(som.bmus'), ',')  # nVectors × 2
writedlm(joinpath(outdir, "colors.csv"), colors, ',')

# meta for Python (matched C++ geometry)
open(joinpath(outdir, "meta.txt"), "w") do io
    println(io, "ncolumns_user=$ncolumns_user")
    println(io, "nrows_user=$nrows_user")
    println(io, "nSomX=$(nrows_user)")
    println(io, "nSomY=$(ncolumns_user)")
    println(io, "epochs=10")
    println(io, "compactsupport=false")
    println(io, "gridtype=rectangular")
    println(io, "maptype=planar")
end

println("Julia umatrix size=$(size(som.umatrix)) range=$(extrema(som.umatrix))")
println("wrote $outdir")
