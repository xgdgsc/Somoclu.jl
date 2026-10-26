using Documenter
using Somoclu

DocMeta.setdocmeta!(Somoclu, :DocTestSetup, :(using Somoclu); recursive=true)

makedocs(
    modules = [Somoclu],
    authors = "xgdgsc and contributors",
    sitename = "Somoclu.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://xgdgsc.github.io/Somoclu.jl",
        edit_link = "main",
        assets = ["assets/fix-home-link.js"],
    ),
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Getting started" => "man/getting_started.md",
            "Visualization" => "man/visualization.md",
        ],
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/xgdgsc/Somoclu.jl.git",
    push_preview = true,
    devbranch = "main",
)
