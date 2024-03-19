module circ

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using WriteVTK
using Random
using Parameters


const N = 4
const dr = 0.02
const crop_R = 0.6*dr
const seed = 123
const xlims = (0.0, 1.0)
const ylims = (0.0, 1.0)


const export_path = "results/cpatch"
include("../utils/populate.jl")
include("../utils/freecut.jl")

@with_kw mutable struct PhysFields
    I1::Float64
    I2::Float64
    crop_R::Float64
    PhysFields(x::RealVector) = new(0.0, 0.0, crop_R)
end

function integral(p::VoronoiPolygon, fun::Function)::Float64
    int = 0.0
    for e in p.edges
        int += integral(p.x, e.v1, e.v2, fun, 5)
    end
    return int
end

function integral(a::RealVector, b::RealVector, c::RealVector, fun::Function, n::Int = 0)::Float64
    if n == 0
        A = tri_area(a, b, c)
        return (A/3)*(fun(2/3*a + 1/6*b + 1/6*c) + fun(1/6*a + 2/3*b + 1/6*c) + fun(1/6*a + 1/6*b + 2/3*c))
    else
        M = (a + b + c)/3
        return integral(a, b, M, fun, n-1) + integral(b, c, M, fun, n-1) + integral(c, a, M, fun, n-1)
    end
end

function get_I1!(p::VoronoiPolygon)
    charfun = (x::RealVector -> (norm(x - p.x) < crop_R ? 1.0 : 0.0))
    p.var.I1 = integral(p, charfun)
end

function get_I2!(p::VoronoiPolygon)
    p.var.I2 = free_area(p)
end



function main()
    #Random.seed!(seed)
    grid = VoronoiGrid{PhysFields}(dr, Rectangle(xlims = xlims, ylims = ylims))
    grid.rr_max = Inf
    #populate_rect!(grid, dr)
    populate_rand!(grid, dr)
    remesh!(grid)
    @time apply_unary!(grid, get_I1!)
    @time apply_unary!(grid, get_I2!)
    vtp = export_grid(grid, "results/circ_integration.vtp", :I1, :I2)
    vtk_save(vtp)
end

end
