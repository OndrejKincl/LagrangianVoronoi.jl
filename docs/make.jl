using Documenter, Literate, Pkg
include("../src/LagrangianVoronoi.jl")
#
# Replace SOURCE_URL marker with github url of source
#

function make_all()
    #
    # Generate Markdown pages from examples
    #
    examples = [
        "gresho.jl",
        "cavity.jl",
        "sedov.jl",
        "doubleshear.jl",
        "rayleightaylor.jl",
        "triplepoint.jl",
        "heat.jl",
        "taylorgreen.jl"
    ]
    example_md_dir  = joinpath(@__DIR__,"src","examples")
    generated_examples = []
    for example in examples
        base,ext=splitext(example)
        if ext==".jl"
            Literate.markdown(joinpath(@__DIR__,"..","examples",example),
                              example_md_dir,
                              documenter=false,
                              info=false
                             )
            push!(generated_examples, joinpath("examples", base*".md"))
        end
    end
    #generated_examples=joinpath.("examples",readdir(example_md_dir, sort = false))
    println(generated_examples)

    makedocs(
        sitename="LagrangianVoronoi.jl",
        authors = "O. Kincl",
        format = Documenter.HTML(prettyurls = false),
        doctest = true,
        clean = true,
        pages=[
            "Home"=>"index.md",
            "Examples" => generated_examples,
            "API Documentation"=>"api.md"
        ]
    )
end

make_all()

deploydocs(
    repo = "github.com/OndrejKincl/LagrangianVoronoi.jl.git",
)
