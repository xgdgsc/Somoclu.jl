using Somoclu
using PlotlyLight
using Test

@testset "Somoclu.jl" begin
    @testset "deterministic codebook (C++ dense kernel, Julia data layout)" begin
        # data columns are samples: [0.1,0.3] and [0.2,0.4]
        # Legacy Julia defaults: gridtype="square" → hexagonal map distances, compactsupport=true
        ncolumns, nrows = 2, 2
        initialcodebook = zeros(Float32, 2, ncolumns * nrows)
        som = Som(ncolumns, nrows; initialcodebook=initialcodebook)
        data = Float32[0.1 0.2; 0.3 0.4]
        train!(som, data)
        # Reference from denseCpuKernels.cpp batch update (verified against pure Julia port)
        correct_codebook = Float32[0.18807973  0.1119203  0.2  0.1;
                                   0.38807973  0.31192032 0.4  0.3]
        @test sum(abs, som.codebook .- correct_codebook) < 1e-5
    end

    @testset "deterministic codebook (Python test vector)" begin
        # From somoclu/src/Python/tests/test_somoclu.py
        # Python data is (n_vectors, n_dim); Julia uses (n_dim, n_vectors) so transpose.
        n_rows, n_columns = 2, 2
        codebook = zeros(Float32, 2, n_columns * n_rows)
        data = Float32[0.1 0.3; 0.2 0.4]  # samples [0.1,0.2], [0.3,0.4]
        som = Som(n_columns, n_rows; initialcodebook=codebook,
                  gridtype="rectangular", compactsupport=false)
        train!(som, data)
        # Exact C++/Python reference (neuron order: (sx,sy) = (0,0),(1,0),(0,1),(1,1)
        # with nSomX=nrows, nSomY=ncolumns — for 2×2 same as Python's nSomX=n_columns)
        correct = Float32[0.2        0.10359724  0.29640275  0.2;
                          0.30000001 0.20359723  0.39640275  0.3]
        @test sum(abs, som.codebook .- correct) < 1e-5
    end

    @testset "rectangular compactsupport=false (Julia sample orientation)" begin
        som = Som(2, 2; initialcodebook=zeros(Float32, 2, 4),
                  gridtype="rectangular", compactsupport=false)
        train!(som, Float32[0.1 0.2; 0.3 0.4])
        correct = Float32[0.15       0.10179862  0.19820137  0.15;
                          0.35       0.3017986   0.3982014   0.35]
        @test sum(abs, som.codebook .- correct) < 1e-5
    end

    @testset "BMUs and U-matrix shapes" begin
        som = Som(5, 4; gridtype="rectangular")
        data = rand(Float32, 3, 20)
        train!(som, data; epochs=5)
        @test size(som.codebook) == (3, 20)
        @test size(som.bmus) == (2, 20)
        @test size(som.umatrix) == (4, 5)  # (nrows, ncolumns)
        @test all(som.bmus .>= 1)
    end

    @testset "toroid hexagonal smoke" begin
        som = Som(6, 6; maptype="toroid", gridtype="hexagonal", compactsupport=true)
        data = rand(Float32, 4, 30)
        train!(som, data; epochs=3, radiuscooling="exponential", scalecooling="exponential")
        @test size(som.codebook) == (4, 36)
    end

    @testset "bubble neighborhood" begin
        som = Som(3, 3; neighborhood="bubble", gridtype="rectangular",
                  initialcodebook=zeros(Float32, 2, 9))
        data = Float32[0.1 0.2 0.3; 0.4 0.5 0.6]
        train!(som, data; epochs=5)
        @test all(isfinite, som.codebook)
    end

    @testset "PCA initialization smoke" begin
        data = rand(Float32, 3, 40) .+ Float32[1, 2, 3]
        som = Som(8, 6; initialization="pca", gridtype="rectangular")
        train!(som, data; epochs=2)
        @test size(som.codebook) == (3, 48)
        @test all(isfinite, som.codebook)
    end

    @testset "PlotlyLight view_* smoke" begin
        som = Som(8, 6; gridtype="rectangular", compactsupport=true)
        data = rand(Float32, 2, 40)
        train!(som, data; epochs=3)

        p = view_umatrix(som; bestmatches=true, colorbar=true)
        @test p isa PlotlyLight.Plot

        planes = view_component_planes(som; dimensions=1:2)
        @test length(planes) == 2

        am = get_surface_state(som)
        @test size(am) == (40, 48)
        p2 = view_activation_map(som; data_index=1)
        @test p2 isa PlotlyLight.Plot
        p3 = view_activation_map(som; data_vector=data[:, 1])
        @test p3 isa PlotlyLight.Plot

        p4 = view_similarity_matrix(som)
        @test p4 isa PlotlyLight.Plot

        mktempdir() do dir
            f = joinpath(dir, "umatrix.html")
            view_umatrix(som; filename=f)
            @test isfile(f)
            @test filesize(f) > 100
        end
    end

    @testset "hexagonal view_umatrix smoke" begin
        som = Som(5, 5; gridtype="hexagonal")
        train!(som, rand(Float32, 2, 20); epochs=2)
        p = view_umatrix(som; bestmatches=true)
        @test p isa PlotlyLight.Plot
    end
end
