# Registering Somoclu.jl in the General registry

Package is published at https://github.com/xgdgsc/Somoclu.jl (`v1.7.6`).

## One-time setup

1. Install the [JuliaRegistrator](https://github.com/apps/juliareigistrator) GitHub App on `xgdgsc/Somoclu.jl` (or the whole account).
2. Ensure [TagBot](https://github.com/JuliaRegistries/TagBot) workflow is present (already in `.github/workflows/TagBot.yml`).

## Register this release

On any commit / release / issue comment:

```text
@JuliaRegistrator register
```

JuliaRegistrator opens a PR to [JuliaRegistries/General](https://github.com/JuliaRegistries/General). New packages typically need ~3 days for AutoMerge (CI green, no name conflicts).

After merge, install with:

```julia
using Pkg
Pkg.add("Somoclu")
```

## Later releases

1. Bump `version` in `Project.toml`
2. Commit and push to `main`
3. Comment `@JuliaRegistrator register` (TagBot creates the `vX.Y.Z` tag after the General PR merges)
