using Somoclu
using PlotlyLight
using Random

outdir = joinpath(@__DIR__, "figures")
mkpath(outdir)

Random.seed!(42)
ncolumns, nrows = 40, 30
nvectors = 50
c1 = rand(Float32, 2, nvectors) ./ 5
c2 = rand(Float32, 2, nvectors) ./ 5 .+ Float32[0.2; 0.5]
c3 = rand(Float32, 2, nvectors) ./ 5 .+ Float32[0.4; 0.1]
data = hcat(c1, c2, c3)
labels = vcat(fill("A", nvectors), fill("B", nvectors), fill("C", nvectors))
colors = vcat(fill("#e41a1c", nvectors), fill("#377eb8", nvectors), fill("#4daf4a", nvectors))

som = Som(ncolumns, nrows; maptype="toroid", gridtype="rectangular", compactsupport=true)
train!(som, data; epochs=15)

view_umatrix(som; bestmatches=true, bestmatchcolors=colors, colorbar=true,
             filename=joinpath(outdir, "umatrix.html"))
println("wrote umatrix.html")

planes = view_component_planes(som; dimensions=1:2, colorbar=true)
for (i, p) in enumerate(planes)
    PlotlyLight.save(p, joinpath(outdir, "component_$i.html"))
end
println("wrote component_1/2.html")

view_activation_map(som; data_index=1, colorbar=true,
                    filename=joinpath(outdir, "activation.html"))
println("wrote activation.html")

view_similarity_matrix(som; labels=labels,
                       filename=joinpath(outdir, "similarity.html"))
println("wrote similarity.html")

# smaller labeled similarity for readable ticks
sub = data[:, 1:3:150]
sub_labels = labels[1:3:150]
som2 = Som(20, 15; gridtype="rectangular")
train!(som2, sub; epochs=10)
view_similarity_matrix(som2; labels=sub_labels,
                       filename=joinpath(outdir, "similarity_labeled.html"))
println("wrote similarity_labeled.html")

# hexagonal variant
som_hex = Som(20, 15; gridtype="hexagonal", maptype="planar")
train!(som_hex, data; epochs=10)
view_umatrix(som_hex; bestmatches=true, bestmatchcolors=colors, colorbar=true,
             filename=joinpath(outdir, "umatrix_hex.html"))
println("wrote umatrix_hex.html")
println("outdir=$outdir")
