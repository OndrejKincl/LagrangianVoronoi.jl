module centralmesh


include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using Match, WriteVTK, LinearAlgebra

const xlims = (-0.5, 0.5)
const ylims = (-0.5, 0.5)
const N = 100 #resolution
const dr = 1.0/N
const dt = 0.01
const t_end = 1.0

@kwdef mutable struct MyPolygon <: VoronoiPolygon
    x::RealVector         # position
    v::RealVector = VEC0
    v_mesh::RealVector = VEC0
    # sides of the polygon
    edges::FastVector{Edge} = LagrangianVoronoi.emptypolygon()
end

MyPolygon(x::RealVector)::MyPolygon = MyPolygon(x=x)
const MyGrid = VoronoiGrid{MyPolygon}

function v_exact(x::RealVector)::RealVector
    omega = @match norm(x) begin
        r, if r < 0.2 end => 5.0
        r, if r < 0.4 end => 2.0/r - 5.0
        _ => 0.0
    end
    return omega*RealVector(-x[2], x[1])
end

function centralize!(grid::VoronoiGrid, nit = 20)
    for p in grid.polygons
        p.v_mesh = VEC0
    end
    for _ in 1:nit
        for p in grid.polygons
            c = centroid(p)
            p.v_mesh += (p.x - c)/dt
            p.x = c
        end
        remesh!(grid)
    end
end

function step!(grid::VoronoiGrid)
    for p in grid.polygons
        p.v = v_exact(p.x)
        p.x += dt*p.v
    end
    remesh!(grid)
end

function main()
    domain = Rectangle(xlims = xlims, ylims = ylims)
    grid = MyGrid(domain, dr)
    populate_hex!(grid)
    #centralize!(grid, 100)
    newtonlloyd!(grid)
    path = "results/centralmesh"
    if !ispath(path)
        mkpath(path)
        @info "created a new path: $(path)"
    end 
    pvd = paraview_collection(joinpath(path, "cells.pvd"))
    frameno = 0
    for t in range(0.0, t_end, step = dt)
        @show t
        #centralize!(grid)
        newtonlloyd!(grid)
        step!(grid)
        pvd[t] = export_grid(grid, joinpath(path, "cfile$frameno.vtp"), :v, :v_mesh)
        frameno += 1
    end
    vtk_save(pvd)
end

end