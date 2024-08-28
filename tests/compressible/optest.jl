module optest
include("../../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi, LinearAlgebra, Random

function ic!(p::VoronoiPolygon)
    p.rho = 1.0 #+ rand()
    p.mass = area(p)*p.rho
    p.c2 = 1.0 #+ randn()
end

function main()
    dr = 0.01
    dt = 0.01
    dom = Rectangle(VEC0, VECX + VECY)
    grid = GridNSc(dom, dr)
    #populate_lloyd!(grid; ic! = ic!, niterations = 1)
    populate_hex!(grid; ic! = ic!)
    op = CompressibleOperator(grid)
    refresh!(op, grid, dt)
    for i in 1:10
        @info "test no $i"
        test_symmetry(op)
        test_posdef(op)
    end
end

function test_symmetry(A)
    x = ThreadedVec(rand(A.n))
    y = ThreadedVec(rand(A.n))
    dot1 = dot(y, A*x) 
    dot2 = dot(x, A*y)
    err = dot1 - dot2
    @show dot1
    @show dot2
    @show err
    if abs(err) < 1e-6 
        @info "passed"
    else
        @warn "non-symmetric"
        @show A*x
    end
end

function test_posdef(A)
    x = ThreadedVec(randn(A.n))
    dp = dot(x, A*x)
    if dp < 0.0
        @warn "indefinite"
        @show dp
    else
        @info "passed"
    end
end

end