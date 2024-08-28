# Gresho Vortex Benchmark

module sedov

using WriteVTK, LinearAlgebra, Random, Match,  Parameters, Polyester
using SmoothedParticles:rDwendland2
using LaTeXStrings, DataFrames, CSV, Plots, Measures

include("../../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi


const rho0 = 1.0
const xlims = (-1.0, 1.0)
const ylims = (-1.0, 1.0)
const N = 100 #resolution
const dr = 1.0/N


const nframes = 100


const gamma = 1.4

const p0 = 1e-6
const c0 = sqrt(gamma*p0/rho0)  # sound speed
const r_bomb = 0.1
const E_bomb = 0.3
const t_bomb = sqrt(rho0/E_bomb*r_bomb^5)
const t_end = 1.0
const CFL = 0.1
const t_relax = 0.1


const export_path = "results/sedov/N$(N)"

# enforce inital condition on a VoronoiPolygon
function ic!(p::VoronoiPolygon)
    p.P = VEC0
    p.A = area(p)
    p.M = rho0*p.A
    p.E = p.A*p0/(gamma - 1.0)
end

function detonate_bomb!(grid::VoronoiGrid)
    A_bomb = 0.0
    for p in grid.polygons
        r = norm(p.x)
        if r < r_bomb
            A_bomb += area(p)
        end
    end
    p_bomb = (gamma-1.0)*E_bomb/A_bomb
    for p in grid.polygons
        r = norm(p.x)
        if r < r_bomb
            p.E = p.A*p_bomb/(gamma-1.0)
        end
    end
end

mutable struct Simulation <: SimulationWorkspace
    grid::GridMUSCL
    E::Float64
    t::Float64
    Simulation() = begin
        domain = Rectangle(xlims = xlims, ylims = ylims)
        grid = GridMUSCL(domain, dr)
        #populate_circ!(grid)
        populate_hex!(grid, ic! = ic!)
        detonate_bomb!(grid)
        return new(grid, 0.0, t_bomb)
    end
end

function step!(sim::Simulation)
    v_shock = 0.4*sim.t^(-0.6)*(E_bomb/rho0)^0.2
    dt = CFL*dr/(sqrt(6.0)*v_shock)
    RK2_step!(sim.grid, dt)
    sim.t += dt
    return
end

# find energy and l2 error
function postproc!(sim::Simulation)
    sim.E = 0.0
    for p in sim.grid.polygons
        sim.E += p.E
    end
    @show sim.E
end

function main()
    if !ispath(export_path)
        mkpath(export_path)
        @info "created a new path: $(export_path)"
    end 
    pvd_c = paraview_collection(joinpath(export_path, "cells.pvd"))
    pvd_p = paraview_collection(joinpath(export_path, "points.pvd"))
    nframe = 0
    sim = Simulation()
    milestones = collect(range(t_end, t_bomb, nframes))
    vtp_vars = (:rho, :u, :e)
    while sim.t < t_end
        step!(sim)
        if sim.t > milestones[end]
            @show sim.t
            postproc!(sim)
            println()
            filename= joinpath(export_path, "cframe$(nframe).vtp")
            pvd_c[sim.t] = export_grid(sim.grid, filename, vtp_vars...)
            filename= joinpath(export_path, "pframe$(nframe).vtp")
            pvd_p[sim.t] = export_points(sim.grid, filename, vtp_vars...)
            pop!(milestones)
            nframe += 1
        end
    end
    vtk_save(pvd_c)
    vtk_save(pvd_p)
    # store velocity and pressure along midline
    x = Float64[]
    rho = Float64[]
    for p in sim.grid.polygons
        push!(x, norm(p.x))
        push!(rho, p.rho)
    end
    csv_data = DataFrame(x=x, rho=rho)
	CSV.write(string(export_path, "/linedata.csv"), csv_data)
    plotdata()
end

function plotdata()
    csv_data = CSV.read(string(export_path, "/linedata.csv"), DataFrame)
    csv_ref = CSV.read("../compressible/reference/sedov.csv", DataFrame)
    plt = scatter(
        csv_data.x,
        csv_data.rho,
        xlabel = L"x",
        ylabel = L"rho",
        label = "density",
        color = :red,
        markeralpha = 0.5,
        bottom_margin = 5mm,
        markersize = 1,
        markerstrokewidth=0,
    )
    plot!(
        plt,
        csv_ref.r,
        csv_ref.rho,
        label = "analytic",
        color = :blue,
        linewidth = 2
    )
    savefig(plt, string(export_path, "/density.pdf"))
    return
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end