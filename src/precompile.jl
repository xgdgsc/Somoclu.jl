# Representative first-use paths (dense/sparse train, IO, PlotlyLight views).

@setup_workload begin
    Random.seed!(0)
    dense = Float32[0.1 0.3 0.0; 0.2 0.0 0.4]
    sparse_data = sparse(dense)
    cb0 = zeros(Float32, 2, 4)

    @compile_workload begin
        # Dense Euclidean (default KA CPU path)
        som = Som(2, 2; initialcodebook=copy(cb0), gridtype="rectangular",
                  compactsupport=false)
        train!(som, dense; epochs=1)
        get_surface_state(som)
        codebook_plane(som, 1)
        view_umatrix(som; bestmatches=false)
        view_component_planes(som; dimensions=1:1)
        view_activation_map(som; data_index=1)
        view_similarity_matrix(som)

        # Dense non-Euclidean BMU metrics
        som_inf = Som(2, 2; initialcodebook=copy(cb0), gridtype="rectangular",
                      compactsupport=false, vect_distance="norm-inf")
        train!(som_inf, dense; epochs=1)
        som_l1 = Som(2, 2; initialcodebook=copy(cb0), gridtype="rectangular",
                     compactsupport=false, vect_distance="norm-1")
        train!(som_l1, dense; epochs=1)

        # Sparse CPU path
        som_sp = Som(2, 2; initialcodebook=copy(cb0), gridtype="rectangular",
                     compactsupport=false)
        train!(som_sp, sparse_data; epochs=1)

        # File IO (tempdir only during package precompilation)
        mktempdir() do dir
            plain = joinpath(dir, "d.txt")
            open(plain, "w") do io
                println(io, "0.1 0.2")
                println(io, "0.3 0.0")
            end
            read_dense_matrix(plain)
            svm = joinpath(dir, "d.svm")
            write_libsvm(svm, dense)
            read_libsvm(svm)
            save_som_dumps(joinpath(dir, "out"), som)
            read_dense_matrix(joinpath(dir, "out.wts"))
        end

        parse_vect_distance("euclidean")
        parse_vect_distance("norm-inf")
        parse_vect_distance("norm-2")
    end
end
