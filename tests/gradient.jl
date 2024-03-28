module gradient

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using WriteVTK
using Random
using Match
using LinearAlgebra
import SmoothedParticles:wendland2

const test_no = 4

function phi_init(x)
    @match test_no begin
        1 => -x[1] + x[2]
        2 => 10*x[1] + 5*x[2] + 3
        3 => cos(x[1])*x[2]
        4 => cos(pi*x[1])*cos(pi*x[2])/pi
    end
end

function grad_exact(x)
    @match test_no begin
        1 => -VECX + VECY
        2 => 10*VECX + 5*VECY
        3 => -sin(x[1])*x[2]*VECX + cos(x[1])*VECY
        4 => -sin(pi*x[1])*cos(pi*x[2])*VECX - cos(pi*x[1])*sin(pi*x[2])*VECY
    end
end

mutable struct PhysFields
    phi::Float64
    grad1::RealVector
    grad2::RealVector
    grad3::RealVector
    grad4::RealVector
    grad_exact::RealVector
    stencil_area::Float64
    gamma::Float64
    PhysFields(x::RealVector) = new(phi_init(x), VEC0, VEC0, VEC0, VEC0, grad_exact(x), 0.0, 0.0)
end

function lr_ratio(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)::Float64
    l2 = norm_squared(e.v1 - e.v2)
    r2 = norm_squared(p.x - q.x)
    return sqrt(l2/r2)
end

function get_grad1!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    p.var.grad1 += lr_ratio(p,q,e)/area(p)*(p.var.phi - q.var.phi)*(p.x - m)
end

function get_grad2!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    z = 0.5*(p.x + q.x)
    p.var.grad2 -= lr_ratio(p,q,e)/area(p)*(
        -(p.var.phi - q.var.phi)*(m - z) 
        +0.5*(-p.var.phi + q.var.phi)*(p.x - q.x)
    )
end

function get_grad3!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    p.var.grad3 += area(q)*q.var.grad2
    p.var.stencil_area += area(q)
    #p.var.grad3 += area(q)*wendland2(h,r)*q.var.grad2
end

function get_grad3!(p::VoronoiPolygon)
    p.var.grad3 += area(p)*p.var.grad2
    p.var.stencil_area += area(p)
    p.var.grad3 /= p.var.stencil_area
    #p.var.grad3 += area(p)*wendland2(h,0.)*p.var.grad2 #p.var.stencil_area
end

function get_grad4!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64, h::Float64)
    p.var.grad4 += area(q)*wendland2(h,r)*q.var.grad2
    p.var.gamma += area(q)*wendland2(h,r)
end

function get_grad4!(p::VoronoiPolygon)
    p.var.grad4 /= p.var.gamma
end

function solve(N)
    dr = 1.0/N
    h = 2*dr
    dom = Rectangle(xlims = (0.0, 1.0), ylims = (0.0, 1.0))
    grid = VoronoiGrid{PhysFields}(h, dom)
    #populate_circ!(grid, dr)
    #populate_rect!(grid, dr)
    Random.seed!(42)
    populate_rand!(grid, dr)
    #populate_vogel!(grid, dr)
    @show length(grid.polygons)
    apply_binary!(grid, get_grad1!)
    apply_binary!(grid, get_grad2!)
    apply_binary!(grid, get_grad3!)
    apply_unary!(grid, get_grad3!)
    h4 = 12*dr
    apply_local!(grid, (p,q,r) -> get_grad4!(p, q, r, h4), h4)
    err1 = 0.0
    err2 = 0.0
    err3 = 0.0
    err4 = 0.0
    for p in grid.polygons
        if !isboundary(p)
            err1 += area(p)*norm(p.var.grad1 - p.var.grad_exact)
            err2 += area(p)*norm(p.var.grad2 - p.var.grad_exact)
            err3 += area(p)*norm(p.var.grad3 - p.var.grad_exact)
            err4 += area(p)*norm(p.var.grad4 - p.var.grad_exact)
        end
    end 
    @show err1
    @show err2
    @show err3
    @show err4
    vtp = export_grid(grid, "results/gradient.vtp", :phi, :grad1, :grad2, :grad3, :grad4, :grad_exact)
    vtk_save(vtp)
end

if abspath(PROGRAM_FILE) == @__FILE__
    solve(50)
end

end