# Registering Somoclu.jl in the General registry

Package is published at https://github.com/xgdgsc/Somoclu.jl (see `Project.toml` for version).


## One-time setup

### Option A — GitHub App (comment bot)

1. Install: https://github.com/apps/juliateam-registrator/installations/new  
   Choose **Only select repositories** → `Somoclu.jl`.
2. TagBot workflow is already in `.github/workflows/TagBot.yml`.
3. Comment on [issue #1](https://github.com/xgdgsc/Somoclu.jl/issues/1) or the release commit:

```text
@JuliaRegistrator register
```

### Option B — JuliaHub web UI (no App install)

1. Open https://juliahub.com and log in with GitHub.
2. Left menu → **Register packages**, submit `https://github.com/xgdgsc/Somoclu.jl`.

Either path opens a PR to [JuliaRegistries/General](https://github.com/JuliaRegistries/General). New packages typically need ~3 days for AutoMerge.

After merge:

```julia
using Pkg
Pkg.add("Somoclu")
```

## Later releases

1. Bump `version` in `Project.toml`
2. Commit and push to `main`
3. Comment `@JuliaRegistrator register` (TagBot creates the `vX.Y.Z` tag after the General PR merges)
