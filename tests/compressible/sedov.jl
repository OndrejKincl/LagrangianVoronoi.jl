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
const N = 50 #resolution
const dr = 1.0/N


const nframes = 50


const gamma = 1.4

const P0 = 1e-4
const c0 = sqrt(gamma*P0/rho0)  # sound speed
const r_bomb = 0.1
const E_bomb = 0.3
const t_bomb = sqrt(rho0/E_bomb*r_bomb^5)
const v_char = r_bomb/t_bomb
@show v_char
@show t_bomb

const dt = 0.5*dr/(sqrt(6.0)*v_char)

const t_end = 1.0 #1.0 - t_bomb


const export_path = "results/sedov/N$(N)"

# enforce inital condition on a VoronoiPolygon
function ic!(p::VoronoiPolygon)
    p.v = VEC0
    p.rho = rho0
    p.area = area(p)
    p.P = P0
    p.e = p.P/((gamma-1.0)*p.rho)
    p.M = p.rho*p.area
    p.U = p.M*p.v
    p.E = p.M*p.e
end

function detonate_bomb!(grid::VoronoiGrid)
    A_bomb = 0.0
    for p in grid.polygons
        r = norm(p.x)
        if r < r_bomb
            A_bomb += area(p)
        end
    end
    P_bomb = E_bomb/A_bomb*(rho0*(gamma-1.0))
    for p in grid.polygons
        r = norm(p.x)
        if r < r_bomb
            p.P = P_bomb
            p.e = p.P/((gamma-1.0)*p.rho)
            p.E = p.M*p.e
        end
    end
end

mutable struct Simulation <: SimulationWorkspace
    grid::GridNSc
    solver::CompressibleSolver
    E::Float64
    l2_err::Float64
    Simulation() = begin
        domain = Rectangle(xlims = xlims, ylims = ylims)
        grid = GridNSc(domain, dr)
        #populate_circ!(grid)
        populate_hex!(grid, ic! = ic!)
        detonate_bomb!(grid)
        return new(grid, CompressibleSolver(grid, dt, verbose=0), 0.0, 0.0)
    end
end




function step!(sim::Simulation, t::Float64)
    find_dv!(sim.grid, 0.01, dt)
    dv_step!(sim.grid, dt)
    viscous_step!(sim.grid, dt, dr)
    ideal_eos!(sim.grid, gamma, P0)
    find_pressure!(sim.solver)
    pressure_step!(sim.grid, dt)
    move!(sim.grid, dt)
    find_rho!(sim.grid)
    return
end

# find energy and l2 error
function postproc!(sim::Simulation, t::Float64) 
    sim.l2_err = 0.0
    sim.E = 0.0
    for p in sim.grid.polygons
        sim.E += p.E
    end
    @show sim.E
end

function main()
    sim = Simulation()
    @time run!(sim, dt, t_end, step!; path = export_path, 
        vtp_vars = (:v, :P, :rho, :M, :U, :E), csv_vars = (:E, :l2_err),
        postproc! = postproc!,
        nframes = nframes
    )
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
    csv_ref = CSV.read("reference/sedov.csv", DataFrame)
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