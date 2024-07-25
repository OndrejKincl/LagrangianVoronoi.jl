module projtest
include("../../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi, LinearAlgebra, WriteVTK

const P0 = 3.0
const dr = 0.01
const dt = 0.1
const c2 = 1.0
const L = 1.0
const V0 = 1.0
const rho0 = 2.0
const A = 4pi*V0*L*rho0*dt/(L^2/c2 + (2pi*dt)^2)

function ic!(p::VoronoiPolygon)
    p.rho = rho0
    p.area = area(p)
    p.mass = area(p)*p.rho
    p.c2 = c2
    p.P = P0
    p.v = v_init(p.x)
end

function v_init(x::RealVector)::RealVector
    return V0*(sin(2pi*x[1]/L)*VECX + sin(2pi*x[2]/L)*VECY)
end

function P_exact(x::RealVector)::Float64
    return P0 - 0.5*A*(cos(2pi*x[1]/L) + cos(2pi*x[2]/L))
end

function main()
    dom = Rectangle(VEC0, L*VECX + L*VECY)
    grid = GridNSc(dom, dr)
    #populate_lloyd!(grid; ic! = ic!, niterations = 10)
    #populate_hex!(grid; ic! = ic!)
    populate_rect!(grid; ic! = ic!)
    solver = CompressibleSolver(grid)
    find_pressure!(solver, dt)
    err = 0.0
    for p in grid.polygons
        p.e = (p.P - P_exact(p.x))/abs(A)
        err += area(p)*p.e^2
    end
    err = sqrt(err)
    @show err
    vtk_file = export_grid(grid, "results/projtest.vtp", :P, :v, :e)
    vtk_save(vtk_file)
    #measure time
    for p in grid.polygons
        p.P = P0
    end
    @time find_pressure!(solver,dt)
end

end