using Somoclu
using PlotlyLight
using Random
using SparseArrays
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

    @testset "sparse epoch-0 matches dense update" begin
        # First epoch from zero codebook: all BMUs at (0,0), updates identical.
        dense = Float32[0.1 0.3; 0.2 0.4]
        sp = sparse(dense)
        cb_s = zeros(Float32, 2, 4)
        cb_d = zeros(Float32, 2, 4)
        bx = zeros(Int32, 2); by = zeros(Int32, 2)
        X2 = Somoclu.sparse_sample_sqnorms(sp)
        W2 = zeros(Float32, 4)
        Somoclu.train_epoch_sparse!(cb_s, sp, bx, by, X2, W2, 2, 2, 2,
            1.0f0, 0.1f0, false, true, 0.5f0, true, false)
        # dense update with BMUs fixed at (0,0)
        for idx in 1:4
            som_y = (idx - 1) ÷ 2; som_x = (idx - 1) % 2
            local_num = zeros(Float32, 2); local_den = 0.0f0
            for n in 1:2
                dist = Somoclu.map_distance(som_x, som_y, 0, 0, 2, 2, true, false)
                w = Somoclu.get_weight(dist, 1.0f0, 0.1f0, false, true, 0.5f0)
                local_den += w
                local_num .+= w .* dense[:, n]
            end
            cb_d[:, idx] .= local_num ./ local_den
        end
        @test sum(abs, cb_s .- cb_d) < 1e-6
        @test all(bx .== 0) && all(by .== 0)
    end

    @testset "sparse train! reproducibility" begin
        Random.seed!(7)
        data = sprand(Float32, 30, 25, 0.08)
        for j in 1:size(data, 2)
            nnz(data[:, j]) == 0 && (data[1, j] = 1f0)
        end
        cb0 = zeros(Float32, 30, 4 * 3)
        s1 = Som(4, 3; initialcodebook=copy(cb0), gridtype="rectangular")
        s2 = Som(4, 3; initialcodebook=copy(cb0), gridtype="rectangular")
        train!(s1, data; epochs=4)
        train!(s2, data; epochs=4)
        @test s1.codebook == s2.codebook
        @test s1.bmus == s2.bmus
    end

    @testset "sparse smoke (true sparsity)" begin
        Random.seed!(1)
        data = sprand(Float32, 50, 40, 0.05)
        # ensure no empty columns (C++ still processes them; keep nonzero for stability)
        for j in 1:size(data, 2)
            if nnz(data[:, j]) == 0
                data[1, j] = 1.0f0
            end
        end
        som = Som(8, 6; gridtype="rectangular", compactsupport=true)
        train!(som, data; epochs=5)
        @test size(som.codebook) == (50, 48)
        @test size(som.bmus) == (2, 40)
        @test all(isfinite, som.codebook)
        @test som.data isa SparseMatrixCSC
    end
end
