# Visualization helpers — PlotlyLight ports of Python somoclu view_* methods.

"""
    codebook_plane(som, dim) -> Matrix{Float32}

Reshape codebook dimension `dim` (1-based) to an `nrows × ncolumns` map.
"""
function codebook_plane(som::Som, dim::Integer)
    som.codebook === nothing && error("The codebook is not available. Train a map first.")
    nDim = size(som.codebook, 1)
    1 <= dim <= nDim || throw(BoundsError(som.codebook, dim))
    # Internal layout: neuron = som_y * nrows + som_x + 1 with nSomX=nrows, nSomY=ncolumns
    plane = Matrix{Float32}(undef, som.nrows, som.ncolumns)
    @inbounds for som_y in 0:(som.ncolumns - 1), som_x in 0:(som.nrows - 1)
        neuron = som_y * som.nrows + som_x + 1
        plane[som_x + 1, som_y + 1] = som.codebook[dim, neuron]
    end
    return plane
end

"""
    get_surface_state(som, data=som.data) -> Matrix{Float64}

Euclidean distances from each sample (rows) to every map neuron (columns),
matching Python `Somoclu.get_surface_state`. `data` is `nDimensions × nVectors`.
"""
function get_surface_state(som::Som, data::Union{Nothing,AbstractMatrix{<:Real}}=nothing)
    som.codebook === nothing && error("The codebook is not available. Train a map first.")
    d = data === nothing ? som.data : Matrix{Float32}(data)
    d === nothing && error("No data available; pass data or train! first.")
    nDim, nVec = size(d)
    size(som.codebook, 1) == nDim || throw(DimensionMismatch("data/codebook dimensions"))
    nNeurons = som.ncolumns * som.nrows
    am = Matrix{Float64}(undef, nVec, nNeurons)
    @inbounds for n in 1:nVec
        for j in 1:nNeurons
            s = 0.0
            for dim in 1:nDim
                Δ = Float64(som.codebook[dim, j]) - Float64(d[dim, n])
                s += Δ * Δ
            end
            am[n, j] = sqrt(s)
        end
    end
    if data === nothing
        som.activation_map = am
    end
    return am
end

"""
    activation_map_matrix(som, data_index) -> Matrix{Float64}

Activation of training sample `data_index` reshaped to `nrows × ncolumns`.
"""
function activation_map_matrix(som::Som, data_index::Integer)
    am = som.activation_map === nothing ? get_surface_state(som) : som.activation_map
    1 <= data_index <= size(am, 1) || throw(BoundsError(am, data_index))
    return reshape(am[data_index, :], som.nrows, som.ncolumns)
end

function activation_map_matrix(som::Som, data_vector::AbstractVector{<:Real})
    som.codebook === nothing && error("The codebook is not available. Train a map first.")
    w = Float32.(data_vector)
    length(w) == size(som.codebook, 1) || throw(DimensionMismatch("vector/codebook dimensions"))
    nNeurons = som.ncolumns * som.nrows
    dists = Vector{Float64}(undef, nNeurons)
    @inbounds for j in 1:nNeurons
        s = 0.0
        for d in eachindex(w)
            Δ = Float64(som.codebook[d, j]) - Float64(w[d])
            s += Δ * Δ
        end
        dists[j] = sqrt(s)
    end
    return reshape(dists, som.nrows, som.ncolumns)
end

# --- Plotly helpers ----------------------------------------------------------

function _default_figsize(som::Som, zoom)
    zr, zc = zoom
    rows = length(zr)
    cols = length(zc)
    width = 640
    height = max(320, round(Int, width * rows / max(cols, 1)))
    return width, height
end

function _resolve_zoom(som::Som, zoom)
    if zoom === nothing
        return (1:som.nrows, 1:som.ncolumns)
    end
    # Accept ((r0,r1),(c0,c1)) as 1-based inclusive or Python-style half-open 0-based.
    (rr, cc) = zoom
    row_range = _as_range(rr, som.nrows)
    col_range = _as_range(cc, som.ncolumns)
    return row_range, col_range
end

function _as_range(lims, n)
    a, b = lims
    if a isa Integer && b isa Integer
        # Python zoom uses half-open 0-based; if b==n and a==0 treat as full map 1:n
        if a == 0 && b == n
            return 1:n
        elseif a >= 1 && b <= n && a <= b
            return a:b
        else
            # half-open 0-based → 1-based inclusive
            return (a + 1):b
        end
    end
    return lims
end

function _filter_bmus(som::Som, row_range, col_range)
    som.bmus === nothing && return Int[], Int[], Int[]
    xs = Int[]
    ys = Int[]
    idxs = Int[]
    @inbounds for n in 1:size(som.bmus, 2)
        # bmus[1]=column (1..ncolumns), bmus[2]=row (1..nrows)
        c = Int(som.bmus[1, n])
        r = Int(som.bmus[2, n])
        if r in row_range && c in col_range
            push!(xs, c - first(col_range) + 1)
            push!(ys, r - first(row_range) + 1)
            push!(idxs, n)
        end
    end
    return xs, ys, idxs
end

function _hex_offsets(n_rows::Int, n_columns::Int)
    offsets = Matrix{Float64}(undef, n_rows * n_columns, 2)
    @inbounds for row in 0:(n_rows - 1), col in 0:(n_columns - 1)
        i = row * n_columns + col + 1
        if row % 2 == 0
            offsets[i, 1] = col + 0.5
            offsets[i, 2] = 2 * n_rows - 2 * row
        else
            offsets[i, 1] = col
            offsets[i, 2] = 2 * n_rows - 2 * row
        end
    end
    return offsets
end

function _view_matrix(som::Som, matrix::AbstractMatrix{<:Real};
                      title::AbstractString="",
                      colorscale::AbstractString="Spectral",
                      reversescale::Bool=true,
                      colorbar::Bool=true,
                      bestmatches::Bool=false,
                      bestmatchcolors=nothing,
                      labels=nothing,
                      zoom=nothing,
                      figsize=nothing,
                      filename::Union{Nothing,AbstractString}=nothing)
    row_range, col_range = _resolve_zoom(som, zoom)
    sub = matrix[row_range, col_range]
    width, height = if figsize === nothing
        _default_figsize(som, (row_range, col_range))
    else
        # figsize like Python (inches) → approximate pixels
        round(Int, figsize[1] * 80), round(Int, figsize[2] * 80)
    end

    hexagonal = som.gridtype == "hexagonal"

    if hexagonal
        nr, nc = size(sub)
        offsets = _hex_offsets(nr, nc)
        # offsets indexed as row * n_columns + col (0-based) matching Python
        colors = Vector{Float64}(undef, nr * nc)
        @inbounds for row in 0:(nr - 1), col in 0:(nc - 1)
            colors[row * nc + col + 1] = Float64(sub[row + 1, col + 1])
        end
        p = plot(
            x = offsets[:, 1],
            y = offsets[:, 2],
            mode = "markers",
            type = "scatter",
            marker = Config(
                symbol = "hexagon",
                size = 28,
                color = colors,
                colorscale = colorscale,
                reversescale = reversescale,
                showscale = colorbar,
                line = Config(width = 0),
            ),
            hovertemplate = "z=%{marker.color:.4g}<extra></extra>",
        )
        if bestmatches && som.bmus !== nothing
            xs, ys, idxs = _filter_bmus(som, row_range, col_range)
            bx = Float64[]
            by = Float64[]
            for (x, y) in zip(xs, ys)
                r0 = y - 1
                c0 = x - 1
                i = r0 * nc + c0 + 1
                if 1 <= i <= size(offsets, 1)
                    push!(bx, offsets[i, 1])
                    push!(by, offsets[i, 2])
                end
            end
            marker_color = _bmu_colors(som, idxs, bestmatchcolors)
            p(x = bx, y = by, type = "scatter", mode = "markers",
              marker = Config(color = marker_color, size = 10,
                              line = Config(color = "white", width = 1)),
              name = "BMUs",
              text = labels === nothing ? nothing : [_label_at(labels, i) for i in idxs],
              hoverinfo = labels === nothing ? "skip" : "text")
        end
        p.layout.xaxis.showgrid = false
        p.layout.yaxis.showgrid = false
        p.layout.xaxis.zeroline = false
        p.layout.yaxis.zeroline = false
        p.layout.xaxis.visible = false
        p.layout.yaxis.visible = false
        p.layout.yaxis.scaleanchor = "x"
        p.layout.yaxis.scaleratio = 1
    else
        # Match matplotlib imshow: row 0 at top
        p = plot(
            z = collect(sub),
            type = "heatmap",
            colorscale = colorscale,
            reversescale = reversescale,
            colorbar = Config(title = ""),
            showscale = colorbar,
            hovertemplate = "row=%{y}, col=%{x}, z=%{z:.4g}<extra></extra>",
        )
        if bestmatches && som.bmus !== nothing
            xs, ys, idxs = _filter_bmus(som, row_range, col_range)
            marker_color = _bmu_colors(som, idxs, bestmatchcolors)
            p(x = xs, y = ys, type = "scatter", mode = "markers+text",
              marker = Config(color = marker_color, size = 10,
                              line = Config(color = "white", width = 1)),
              text = labels === nothing ? fill("", length(idxs)) : [_label_at(labels, i) for i in idxs],
              textposition = "top right",
              name = "BMUs")
        end
        p.layout.yaxis.autorange = "reversed"
        p.layout.xaxis.title.text = "column"
        p.layout.yaxis.title.text = "row"
        p.layout.yaxis.scaleanchor = "x"
        p.layout.yaxis.scaleratio = 1
    end

    p.layout.title.text = title
    p.layout.width = width
    p.layout.height = height
    p.layout.margin = Config(l = 40, r = 20, t = 50, b = 40)

    if filename !== nothing
        PlotlyLight.save(p, filename)
    end
    return p
end

function _label_at(labels, i)
    i > length(labels) && return ""
    lab = labels[i]
    return lab === nothing ? "" : string(lab)
end

function _bmu_colors(som::Som, idxs::Vector{Int}, bestmatchcolors)
    if bestmatchcolors !== nothing
        return [bestmatchcolors[i] for i in idxs]
    elseif som.clusters !== nothing
        colors = Int[]
        for i in idxs
            c = Int(som.bmus[1, i])
            r = Int(som.bmus[2, i])
            push!(colors, som.clusters[r, c])
        end
        return colors
    else
        return fill("white", length(idxs))
    end
end

"""
    view_umatrix(som; kwargs...) -> PlotlyLight.Plot

Plot the U-matrix (Python `view_umatrix`).

# Keywords
- `colorscale="Spectral"`, `reversescale=true` (≈ matplotlib `Spectral_r`)
- `colorbar=true`, `bestmatches=false`, `bestmatchcolors=nothing`, `labels=nothing`
- `zoom=nothing` — `((row0,row1),(col0,col1))` half-open 0-based or 1-based ranges
- `figsize=nothing` — `(width_in, height_in)` approx inches
- `filename=nothing` — if set, save HTML via `PlotlyLight.save`
"""
function view_umatrix(som::Som; kwargs...)
    som.umatrix === nothing && error("The U-matrix is not available. Train a map first.")
    return _view_matrix(som, som.umatrix; title="U-matrix", kwargs...)
end

"""
    view_component_planes(som; dimensions=nothing, kwargs...) -> Vector{Plot}

Plot codebook component planes (Python `view_component_planes`).
Returns one plot per requested dimension.
"""
function view_component_planes(som::Som; dimensions=nothing, kwargs...)
    som.codebook === nothing && error("The codebook is not available. Train a map first.")
    dims = dimensions === nothing ? (1:size(som.codebook, 1)) : dimensions
    plots = Plot[]
    for d in dims
        p = _view_matrix(som, codebook_plane(som, d);
                         title="Component plane $d", kwargs...)
        push!(plots, p)
    end
    return plots
end

"""
    view_activation_map(som; data_vector=nothing, data_index=nothing, kwargs...) -> Plot

Plot activation of one sample or a new vector (Python `view_activation_map`).
"""
function view_activation_map(som::Som;
                             data_vector=nothing,
                             data_index=nothing,
                             activation_map=nothing,
                             kwargs...)
    if data_vector === nothing && data_index === nothing
        error("Specify data_vector or data_index")
    end
    if data_vector !== nothing && data_index !== nothing
        error("Cannot specify both data_vector and data_index")
    end
    matrix = if data_vector !== nothing
        activation_map_matrix(som, vec(data_vector))
    else
        if activation_map !== nothing
            reshape(activation_map[data_index, :], som.nrows, som.ncolumns)
        else
            activation_map_matrix(som, data_index)
        end
    end
    return _view_matrix(som, matrix; title="Activation map", kwargs...)
end

"""
    view_similarity_matrix(som; data=nothing, labels=nothing, kwargs...) -> Plot

Correlation similarity of activation vectors (Python `view_similarity_matrix`).
"""
function view_similarity_matrix(som::Som;
                                data::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                                labels=nothing,
                                figsize=nothing,
                                filename::Union{Nothing,AbstractString}=nothing,
                                colorscale::AbstractString="RdBu")
    X = if data !== nothing
        Matrix{Float64}(data)
    else
        get_surface_state(som)
    end
    # pairwise correlation distance → similarity = 1 - corr_dist
    # corr_dist(u,v) = 1 - corr(u,v)  ⇒  similarity = corr(u,v)
    n = size(X, 1)
    corrmat = Matrix{Float64}(undef, n, n)
    # center rows
    Xc = X .- mean(X; dims=2)
    norms = sqrt.(sum(abs2, Xc; dims=2))
    @inbounds for i in 1:n, j in 1:n
        ni = norms[i]
        nj = norms[j]
        corrmat[i, j] = (ni == 0 || nj == 0) ? 0.0 : dot(view(Xc, i, :), view(Xc, j, :)) / (ni * nj)
    end

    width, height = if figsize === nothing
        (720, 600)
    else
        round(Int, figsize[1] * 80), round(Int, figsize[2] * 80)
    end

    # Numeric axes avoid Plotly collapsing duplicate categorical labels.
    idx = collect(1:n)
    if labels !== nothing
        length(labels) == n || error("labels length must match number of samples ($n)")
    end
    p = plot(
        z = corrmat,
        type = "heatmap",
        colorscale = colorscale,
        zmid = 0,
        zmin = -1,
        zmax = 1,
        colorbar = Config(title = "corr"),
        x = idx,
        y = idx,
        hovertemplate = "i=%{y}, j=%{x}, corr=%{z:.3f}<extra></extra>",
    )
    p.layout.title.text = "Similarity matrix"
    p.layout.width = width
    p.layout.height = height
    p.layout.yaxis.autorange = "reversed"
    p.layout.xaxis.tickangle = -90
    if labels !== nothing && n <= 60
        # Unique tick text so categories stay one-to-one with samples
        ticktext = ["$(labels[i])#$i" for i in 1:n]
        p.layout.xaxis.tickmode = "array"
        p.layout.xaxis.tickvals = idx
        p.layout.xaxis.ticktext = ticktext
        p.layout.yaxis.tickmode = "array"
        p.layout.yaxis.tickvals = idx
        p.layout.yaxis.ticktext = ticktext
    end

    if filename !== nothing
        PlotlyLight.save(p, filename)
    end
    return p
end
