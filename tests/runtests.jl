using Test

@testset "taylorgreen" begin
    include("taylorgreen.jl")
    taylorgreen.main()
end