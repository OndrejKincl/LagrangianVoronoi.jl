module divergence

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using WriteVTK
using Random
using Match
include("../utils/populate.jl")

const test_no = 2

function v_init(x)
    @match test_no begin
        1 => -x[1]*VECX + x[2]*VECY
        2 => x[2]*cos(pi*x[1])*VECX + (x[1] + sin(pi*x[2]))*VECY
        _ => VEC0
    end
end

function div_exact(x)
    @match test_no begin
        1 => 0.0
        2 => -pi*x[2]*sin(pi*x[1]) + pi*cos(pi*x[2])
        _ => 0.0
    end
end

mutable struct PhysFields
    v::RealVector
    div::Float64
    div_exact::Float64
    PhysFields(x::RealVector) = new(v_init(x), 0.0, div_exact(x))
end

function lr_ratio(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)::Float64
    l2 = norm_squared(e.v1 - e.v2)
    r2 = norm_squared(p.x - q.x)
    return sqrt(l2/r2)
end

function get_div!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    p.var.div += lr_ratio(p,q,e)*dot(m - q.x, p.var.v - q.var.v)
end

function get_div2!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    p.var.div += lr_ratio(p,q,e)*dot(p.x - m, p.var.v - q.var.v)
end

function average_value(p::VoronoiPolygon, f::Function)::Float64
    int = 0.0
    for e in p.edges
        A = 0.5*abs(LagrangianVoronoi.cross2(e.v1 - p.x, e.v2 - p.x))
        int += (A/3)*f(2/3*p.x + 1/6*e.v1 + 1/6*e.v2)
        int += (A/3)*f(1/6*p.x + 2/3*e.v1 + 1/6*e.v2)
        int += (A/3)*f(1/6*p.x + 1/6*e.v1 + 2/3*e.v2)
    end
    return int/area(p)
end

function solve(N)
    dr = 1.0/N
    h = 2*dr
    dom = Rectangle(xlims = (0.0, 1.0), ylims = (0.0, 1.0))
    grid = VoronoiGrid{PhysFields}(h, dom)
    grid.rr_max = Inf
    populate_circ!(grid, dr)
    #Random.seed!(42)
    #populate_rand!(grid, dr)
    apply_binary!(grid, get_div2!)
    err = 0.0
    for p in grid.polygons
        p.var.div /= area(p)
        if !isboundary(p)
            div_avg = average_value(p, div_exact)
            #err += area(p)*abs(p.var.div - p.var.div_exact)
            err += area(p)*abs(p.var.div - div_avg)
        end
    end 
    @show err
    vtp = export_grid(grid, "results/divergence.vtp", :v, :div, :div_exact)
    vtk_save(vtp)
end

end