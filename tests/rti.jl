# Rayleigh-Taylor instability

module rti

using WriteVTK, LinearAlgebra, Random, Match, DataFrames, CSV, Parameters
using SmoothedParticles:rDwendland2

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi

const rho_d = 1.0
const rho_u = 1.8
const Re = 420.0
const Fr = 1.0
#const atwood = (rho_u - rho_d)/(rho_u + rho_d)

const xlims = (0.0, 1.0)
const ylims = (0.0, 2.0)

const N = 60 #resolution
const dr = 1.0/N
const h = 2*dr

const v_char = 10.0
const l_char = 1.0
const dt = 0.1*dr/v_char
const tau = 0.2
const t_end =  3.0

const export_path = "results/rtiN$(N)"
const PROJECTION_STEPS = 10


function dividing_curve(x::Float64)::Float64
    return 1.0 - 0.15*sin(2*pi*x[1])
end

# inital condition
function ic!(p::VoronoiPolygon)
    p.rho = (p.x[2] > dividing_curve(p.x[1]) ? rho_u : rho_d)
    p.mass = p.rho*area(p)
end

function gravity!(p::VoronoiPolygon)
    p.v -= dt/(Fr^2)*VECY
    if isboundary(p)
        p.v = VEC0
    end
end

function lloyd_stabilizer!(p::VoronoiPolygon)
    p.x = (tau*p.x + dt*centroid(p))/(tau + dt)
    return
end

mutable struct Simulation <: SimulationWorkspace
    grid::GridNS
    solver::PressureSolver{PolygonNS}
    E::Float64
    Simulation() = begin
        domain = Rectangle(xlims = xlims, ylims = ylims)
        grid = GridNS(domain, dr)
        populate_lloyd!(grid, ic! = ic!)
        return new(grid, PressureSolver(grid), 0.0)
    end
end

function step!(sim::Simulation, t::Float64)
    apply_unary!(sim.grid, lloyd_stabilizer!)
    move!(sim.grid, dt)
    viscous_force!(sim.grid, 1.0/Re, dt)
    apply_unary!(sim.grid, gravity!)
    for _ in 1:PROJECTION_STEPS
        find_pressure!(sim.solver, dt)
    end
    pressure_force!(sim.grid, dt)
end

function postproc!(sim::Simulation, t::Float64)
    sim.E = 0.0
    for p in sim.grid.polygons
        sim.E += 0.5*p.mass*norm_squared(p.v) + p.mass*(1.0/Fr^2)*(p.x[2] - ylims[1])
    end
    println("energy = $(sim.E)")
end

function main()
    sim = Simulation()
    run!(sim, dt, t_end, step!, 
        path = export_path,
        vtp_vars = (:rho, :P, :v), save_csv = false,
        postproc! = postproc!
    )
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end