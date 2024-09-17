# Rayleigh-Taylor instability

module rti

using WriteVTK, LinearAlgebra, Random, Match, DataFrames, CSV, Parameters
using SmoothedParticles:rDwendland2

include("../../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi

const rho_d = 1.0
const rho_u = 1.8
const Re = 420.0
const Fr = 1.0
const c = 20.0
const g = 1/(Fr^2)
const gamma = 1.4
#const atwood = (rho_u - rho_d)/(rho_u + rho_d)

const xlims = (0.0, 1.0)
const ylims = (0.0, 2.0)

const N = 100 #resolution
const dr = 1.0/N
const h = 2*dr

const v_char = 1.0
const l_char = 1.0
const dt = 0.1*dr/v_char
const t_end =  10*dt #5.0
const nframes = 400

const export_path = "results/rtiN$(N)"


function dividing_curve(x::Float64)::Float64
    return 1.0 - 0.15*sin(2*pi*x[1])
end

# inital condition
function ic!(p::VoronoiPolygon)
    dy = dividing_curve(p.x[1])
    p.phase = (p.x[2] > dy ? 0 : 1)
    p.rho = (p.x[2] > dy ? rho_u : rho_d)
    p.mass = p.rho*area(p)
    p.mu = p.rho/Re
    p.P = rho_d*c^2/gamma
    p.P -= max(p.x[2], dy)*rho_d*g
    p.P -= min(0.0, p.x[2]-dy)*rho_u*g
    p.e = p.P/(p.rho*(gamma - 1.0)) + g*p.x[2]
end


mutable struct Simulation <: SimulationWorkspace
    grid::GridNS
    psolver::PressureSolver{PolygonNS}
    msolver::MultiphaseSolver{PolygonNS}
    E::Float64
    Simulation() = begin
        domain = Rectangle(xlims = xlims, ylims = ylims)
        grid = GridNS(domain, dr)
        populate_lloyd!(grid, ic! = ic!)
        return new(grid, PressureSolver(grid), MultiphaseSolver(grid), 0.0)
    end
end

function step!(sim::Simulation, t::Float64)
    move!(sim.grid, dt)
    gravity_step!(sim.grid, -g*VECY, dt)
    ideal_eos!(sim.grid)
    find_pressure!(sim.psolver, dt)
    pressure_step!(sim.grid, dt)
    find_D!(sim.grid, noslip = true)
    viscous_step!(sim.grid, dt)
    find_dv!(sim.grid, dt)
    multiphase_projection!(sim.msolver)
    relaxation_step!(sim.grid, dt)
    return
end

function postproc!(sim::Simulation, t::Float64)
    sim.E = 0.0
    for p in sim.grid.polygons
        sim.E += p.mass*p.e
    end
    println("energy = $(sim.E)")
end

function main()
    sim = Simulation()
    run!(sim, dt, t_end, step!, 
        path = export_path,
        vtp_vars = (:rho, :P, :v, :phase), save_csv = false,
        postproc! = postproc!,
        nframes = nframes
    )
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end