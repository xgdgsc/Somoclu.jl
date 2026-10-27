# File IO matching somoclu/src/io.cpp (ESOM-style dumps + libsvm sparse).
# Julia matrices are nDimensions × nVectors (columns = samples). C++ flat buffers
# store samples contiguously (row-major nVectors × nDimensions).

"""Format like C++ `%0.10f` (fixed 10 decimals)."""
_fmt10(x::Float32) = string(round(x; digits=10))

"""
    read_dense_matrix(path) -> Matrix{Float32}

Read a whitespace-separated dense matrix file (or `.lrn` / `.wts`).
Returns `nFeatures × nSamples` (Julia column-sample layout).
Comment lines starting with `#` are skipped. `.lrn` / `.wts` `%` headers are parsed
as in C++ `readMatrix` / `readLrnHeader` / `readWtsHeader`.
"""
function read_dense_matrix(path::AbstractString)
    ext = lowercase(splitext(path)[2])
    if ext == ".lrn"
        return _read_lrn(path)
    elseif ext == ".wts"
        return _read_wts(path)
    else
        return _read_plain_dense(path)
    end
end

function _read_plain_dense(path::AbstractString)
    rows = Vector{Vector{Float32}}()
    ncols = 0
    open(path, "r") do io
        for line in eachline(io)
            isempty(line) && continue
            startswith(line, "#") && continue
            vals = Float32.(parse.(Float64, split(line)))
            isempty(vals) && continue
            if ncols == 0
                ncols = length(vals)
            elseif length(vals) != ncols
                error("Inconsistent column count in $path")
            end
            push!(rows, vals)
        end
    end
    isempty(rows) && error("No data in $path")
    # file rows = samples → Julia columns
    nSamples = length(rows)
    data = Matrix{Float32}(undef, ncols, nSamples)
    @inbounds for j in 1:nSamples
        data[:, j] .= rows[j]
    end
    return data
end

function _read_lrn(path::AbstractString)
    nRows = 0
    nAllColumns = 0
    columnMap = Int[]
    open(path, "r") do io
        for line in eachline(io)
            startswith(line, "#") && continue
            if startswith(line, "%")
                # Fourth `%` line is column names (non-numeric) — stop after columnMap
                if nRows > 0 && nAllColumns > 0 && !isempty(columnMap)
                    break
                end
                body = strip(line[2:end])
                nums = parse.(Int, split(body))
                if nRows == 0
                    nRows = nums[1]
                elseif nAllColumns == 0
                    nAllColumns = nums[1]
                else
                    columnMap = nums
                end
            end
        end
    end
    isempty(columnMap) && error("Invalid .lrn header in $path")
    feature_cols = findall(==(1), columnMap)
    nFeat = length(feature_cols)
    data = Matrix{Float32}(undef, nFeat, nRows)
    i = 0
    open(path, "r") do io
        for line in eachline(io)
            startswith(line, "#") && continue
            startswith(line, "%") && continue
            isempty(strip(line)) && continue
            vals = parse.(Float64, split(strip(line)))
            length(vals) >= nAllColumns || error("Short .lrn data row in $path")
            i += 1
            i > nRows && break
            @inbounds for (k, c) in enumerate(feature_cols)
                data[k, i] = Float32(vals[c])
            end
        end
    end
    i == nRows || error("Expected $nRows data rows in $path, got $i")
    return data
end

function _read_wts(path::AbstractString)
    nSomY = 0
    nSomX = 0
    nDim = 0
    open(path, "r") do io
        for line in eachline(io)
            startswith(line, "#") && continue
            if startswith(line, "%")
                body = strip(line[2:end])
                nums = parse.(Int, split(body))
                if nSomY == 0
                    nSomY, nSomX = nums[1], nums[2]
                elseif nDim == 0
                    nDim = nums[1]
                else
                    break
                end
            end
        end
    end
    (nSomX > 0 && nSomY > 0 && nDim > 0) || error("Invalid .wts header in $path")
    nNeurons = nSomX * nSomY
    codebook = Matrix{Float32}(undef, nDim, nNeurons)
    idx = 0
    open(path, "r") do io
        for line in eachline(io)
            startswith(line, "#") && continue
            startswith(line, "%") && continue
            isempty(strip(line)) && continue
            vals = Float32.(parse.(Float64, split(strip(line))))
            length(vals) == nDim || error("Bad .wts weight row in $path")
            idx += 1
            idx > nNeurons && error("Too many weight rows in $path")
            codebook[:, idx] .= vals
        end
    end
    idx == nNeurons || error("Expected $nNeurons weight rows in $path, got $idx")
    return codebook
end

"""
    read_libsvm(path; zerobased=nothing) -> SparseMatrixCSC{Float32,Int}

Read a libsvm / sparse text file (`index:value` per line, optional leading label).
Returns `nFeatures × nSamples`. If `zerobased` is `nothing`, auto-detect from indices.
Matches C++ `readSparseMatrixDimensions` / `readSparseMatrixChunk` layout after
converting to 1-based Julia feature rows.
"""
function read_libsvm(path::AbstractString; zerobased::Union{Nothing,Bool}=nothing)
    rows = Vector{Tuple{Vector{Int},Vector{Float32}}}()
    max_index = -1
    saw_zero = false
    open(path, "r") do io
        for line in eachline(io)
            startswith(line, "#") && continue
            isempty(strip(line)) && continue
            toks = split(strip(line))
            isempty(toks) && continue
            start = 1
            if !occursin(':', toks[1])
                start = 2  # skip label
            end
            idxs = Int[]
            vals = Float32[]
            for t in @view toks[start:end]
                sep = findfirst(':', t)
                sep === nothing && continue
                idx = parse(Int, t[1:prevind(t, sep)])
                val = parse(Float32, t[nextind(t, sep):end])
                idx == 0 && (saw_zero = true)
                max_index = max(max_index, idx)
                push!(idxs, idx)
                push!(vals, val)
            end
            push!(rows, (idxs, vals))
        end
    end
    isempty(rows) && error("No sparse data in $path")
    zb = something(zerobased, saw_zero)
    nFeat = zb ? max_index + 1 : max_index
    nFeat > 0 || error("No features found in $path")
    nSamples = length(rows)
    I = Int[]
    J = Int[]
    V = Float32[]
    for (j, (idxs, vals)) in enumerate(rows)
        for (idx, val) in zip(idxs, vals)
            row = zb ? idx + 1 : idx
            push!(I, row)
            push!(J, j)
            push!(V, val)
        end
    end
    return sparse(I, J, V, nFeat, nSamples)
end

"""
    write_libsvm(path, data::AbstractMatrix; labels=nothing, zerobased=false)

Write dense or sparse data in libsvm `index:value` format (1-based indices by default).
`data` is `nFeatures × nSamples`.
"""
function write_libsvm(path::AbstractString, data::AbstractMatrix{<:Real};
                      labels=nothing, zerobased::Bool=false)
    nFeat, nSamples = size(data)
    open(path, "w") do io
        for j in 1:nSamples
            if labels !== nothing
                print(io, labels[j], ' ')
            end
            first = true
            if data isa SparseMatrixCSC
                for k in nzrange(data, j)
                    row = rowvals(data)[k]
                    idx = zerobased ? row - 1 : row
                    first || print(io, ' ')
                    print(io, idx, ':', Float32(nonzeros(data)[k]))
                    first = false
                end
            else
                @inbounds for d in 1:nFeat
                    v = Float32(data[d, j])
                    v == 0f0 && continue
                    idx = zerobased ? d - 1 : d
                    first || print(io, ' ')
                    print(io, idx, ':', v)
                    first = false
                end
            end
            println(io)
        end
    end
    return nothing
end

"""
    save_codebook(path, som::Som)
    save_codebook(path, codebook, nSomX, nSomY)

Write an ESOM-style `.wts` codebook (`%nSomY %nSomX` / `%nDim` / weight rows).
Uses internal layout `nSomX = nrows`, `nSomY = ncolumns`.
"""
function save_codebook(path::AbstractString, som::Som)
    som.codebook === nothing && error("No codebook; train! first")
    nSomX, nSomY, _ = _map_shape(som)
    return save_codebook(path, som.codebook, nSomX, nSomY)
end

function save_codebook(path::AbstractString, codebook::AbstractMatrix{<:Real},
                       nSomX::Integer, nSomY::Integer)
    nDim, nNeurons = size(codebook)
    nNeurons == nSomX * nSomY || throw(DimensionMismatch("codebook vs map size"))
    open(path, "w") do io
        println(io, "%", Int(nSomY), ' ', Int(nSomX))
        println(io, "%", nDim)
        @inbounds for som_y in 0:(Int(nSomY) - 1), som_x in 0:(Int(nSomX) - 1)
            neuron = som_y * Int(nSomX) + som_x + 1
            for d in 1:nDim
                # C++: fprintf("%0.10f ", ...) including trailing space
                print(io, _fmt10(Float32(codebook[d, neuron])), ' ')
            end
            println(io)
        end
    end
    return nothing
end

"""
    save_bmus(path, som::Som)

Write ESOM-style `.bm` best-matching units (`index som_y som_x`, 0-based),
matching C++ `saveBmus` (ESOM swaps x/y in the file).
"""
function save_bmus(path::AbstractString, som::Som)
    som.bmus === nothing && error("No BMUs; train! first")
    nSomX, nSomY, _ = _map_shape(som)
    nVectors = size(som.bmus, 2)
    open(path, "w") do io
        println(io, "%", nSomY, ' ', nSomX)
        println(io, "%", nVectors)
        @inbounds for i in 1:nVectors
            # som.bmus[1]=y+1, som.bmus[2]=x+1 → file: i-1? C++ writes 0-based sample index i
            # C++: fprintf("%d %d %d\\n", i, bmus[2*i+1], bmus[2*i]) with i from 0..n-1
            yi = Int(som.bmus[1, i]) - 1
            xi = Int(som.bmus[2, i]) - 1
            println(io, i - 1, ' ', yi, ' ', xi)
        end
    end
    return nothing
end

"""
    save_umatrix(path, som::Som)

Write ESOM-style `.umx` U-matrix (`%nSomY %nSomX` then rows).
"""
function save_umatrix(path::AbstractString, som::Som)
    som.umatrix === nothing && error("No U-matrix; train! first")
    nSomX, nSomY, _ = _map_shape(som)
    open(path, "w") do io
        println(io, "%", nSomY, ' ', nSomX)
        # C++: for som_y, for som_x: uMatrix[som_y * nSomX + som_x]
        # Julia umatrix is reshape(..., nSomX, nSomY) → umatrix[som_x+1, som_y+1]
        @inbounds for som_y in 1:nSomY
            for som_x in 1:nSomX
                print(io, ' ', Float32(som.umatrix[som_x, som_y]))
            end
            println(io)
        end
    end
    return nothing
end

"""
    save_som_dumps(prefix, som::Som)

Write `prefix.wts`, `prefix.bm`, and `prefix.umx` (C++ CLI default dumps).
"""
function save_som_dumps(prefix::AbstractString, som::Som)
    save_codebook(string(prefix, ".wts"), som)
    save_bmus(string(prefix, ".bm"), som)
    save_umatrix(string(prefix, ".umx"), som)
    return nothing
end
