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


const nframes = 100


const gamma = 1.4

const P0 = 1e-4
const c0 = sqrt(gamma*P0/rho0)  # sound speed
const v_char = 4.0
const r_core = 0.2
const E_core = 0.3
const P_core = E_core/(pi*r_core^2)*(rho0*(gamma-1.0))
const dt = 0.1*dr/v_char

const t_end = 1.0
const tau = 0.01


const h_stab = 1.5*dr
const P_stab = 0.01*P_core

const export_path = "results/sedov/N$(N)"

# enforce inital condition on a VoronoiPolygon
function ic!(p::VoronoiPolygon)
    p.v = VEC0
    p.rho = rho0
    p.P = P0
    p.mass = p.rho*area(p)
    p.e = p.P/((gamma-1.0)*p.rho)
end

function detonate_bomb!(grid::VoronoiGrid)
    
    for p in grid.polygons
        r = norm(p.x)
        if r < r_core
            p.P = P_core
            p.e = p.P/((gamma-1.0)*p.rho)
        end
    end
    
    #p = argmin(p -> norm(p.x), grid.polygons)
    #p.P = ((gamma-1.0)*E_core)/area(p)
end

mutable struct Simulation <: SimulationWorkspace
    grid::GridNSc
    solver::CompressibleSolver
    E::Float64
    l2_err::Float64
    ls::LloydStabilizer
    Simulation() = begin
        domain = Rectangle(xlims = xlims, ylims = ylims)
        grid = GridNSc(domain, dr)
        #populate_circ!(grid)
        populate_hex!(grid, ic! = ic!)
        detonate_bomb!(grid)
        return new(grid, CompressibleSolver(grid, dt, verbose=0), 0.0, 0.0, LloydStabilizer(grid))
    end
end

function SPH_stabilizer!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64)
    (xlims[1] + h_stab < p.x[1] < xlims[2] - h_stab) || return
    (ylims[1] + h_stab < p.x[2] < ylims[2] - h_stab) || return
	p.v += -dt*q.mass*rDwendland2(h_stab,r)*(P_stab/p.rho + P_stab/q.rho)*(p.x - q.x)
    return
end

function get_rho!(grid::GridNSc)
    @batch for p in grid.polygons
        p.area = area(p)
        p.rho = p.mass/p.area
    end
end

function ideal_eos!(grid::GridNSc, gamma::Float64)
    @batch for p in grid.polygons
        p.P = (gamma-1.0)*p.rho*(p.e - 0.5*norm_squared(p.v))
        p.c2 = gamma*max(p.P,P0)/p.rho
    end
end

function step!(sim::Simulation, t::Float64)
    get_rho!(sim.grid)
    #apply_local!(sim.grid, SPH_stabilizer!, h_stab)
    viscous_step!(sim.grid, dt, dr, gamma)
    ideal_eos!(sim.grid, gamma)
    find_pressure!(sim.solver)
    pressure_force!(sim.grid, dt, stabilize=false)
    energy_balance!(sim.grid, dt)
    #apply_unary!(sim.grid, lloyd_stabilizer!)
    move!(sim.grid, dt)
    stabilize!(sim.ls)
    return
end


function lloyd_stabilizer!(p::VoronoiPolygon)
    if !isboundary(p)
        p.x = (tau*p.x + dt*centroid(p))/(tau + dt)
    end
    return
end

# find energy and l2 error
function postproc!(sim::Simulation, t::Float64) 
    sim.l2_err = 0.0
    sim.E = 0.0
    for p in sim.grid.polygons
        sim.E += p.mass*p.e #0.5*p.mass*norm_squared(p.v) + p.mass*p.P/(p.rho*(gamma - 1.0))
    end
    @show sim.E
    end

function main()
    sim = Simulation()
    @time run!(sim, dt, t_end, step!; path = export_path, 
        vtp_vars = (:v, :P, :rho), csv_vars = (:E, :l2_err),
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