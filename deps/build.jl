using BinDeps

@BinDeps.setup

link = "https://github.com/peterwittek/somoclu/releases/download/1.7.0/somoclu-1.7.0-julia-test.tar.gz"

libsomoclu = library_dependency("libsomoclu")
provides(Sources, Dict(URI(link) => libsomoclu))
provides(BuildProcess, Autotools(libtarget = joinpath("src", "libsomoclu.so"), configure_options=[AbstractString("--without-mpi")]), libsomoclu)

@BinDeps.install Dict(:libsomoclu => :libsomoclu)
