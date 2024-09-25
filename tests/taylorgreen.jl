# a simplified version of `examples/taylorgreen.jl` for unit testing
# not a very thorough test but better than nothing

module taylorgreen
include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi, Test, LinearAlgebra


const rho0 = 1.0
const xlims = (0.0, 1.0)
const ylims = (0.0, 1.0)
const t_end = 0.1
const Re = 400
const N = 80
const dr = 1.0/N
const dt = 0.1*dr
const c0 = 50.0
const gamma = 1.4
const P0 = rho0*c0^2/gamma
const export_path = "results"

function v_max(t::Float64)::Float64
    return exp(-8.0*pi^2*t/Re)
end

function ic!(p::VoronoiPolygon)
    p.v = v_exact(p.x, 0.0)
    p.rho = rho0
    p.mass = p.rho*area(p)
    p.P = P_exact(p.x, 0.0)
    p.e = 0.5*norm_squared(p.v) + p.P/(p.rho*(gamma - 1.0))
    p.mu = 1.0/Re
end

function v_exact(x::RealVector, t::Float64)::RealVector
    u0 =  cos(2pi*x[1])*sin(2pi*x[2])
    v0 = -sin(2pi*x[1])*cos(2pi*x[2])
    return v_max(t)*(u0*VECX + v0*VECY)
end

function P_exact(x::RealVector, t::Float64)::Float64
    return 0.5*(v_max(t)^2)*(sin(2pi*x[1])^2 + sin(2pi*x[2])^2 - 1.0)
end

mutable struct Simulation <: SimulationWorkspace
    grid::GridNS
    solver::PressureSolver{PolygonNS}
    v_err::Float64
    P_err::Float64
    E_err::Float64
    E0::Float64
    Simulation() = begin
        domain = Rectangle(xlims = xlims, ylims = ylims)
        grid = GridNS(domain, dr, xperiodic = true, yperiodic = true)
        populate_hex!(grid, ic! = ic!)
        solver = PressureSolver(grid)
        return new(grid, solver, 0.0, 0.0, 0.0, NaN)
    end
end

function step!(sim::Simulation, t::Float64)
    move!(sim.grid, dt)
    stiffened_eos!(sim.grid, gamma, P0)
    find_pressure!(sim.solver, dt)
    pressure_step!(sim.grid, dt)
    find_D!(sim.grid)
    viscous_step!(sim.grid, dt; artificial_viscosity = false)
    find_dv!(sim.grid, dt)
    relaxation_step!(sim.grid, dt)
    return
end

function postproc!(sim::Simulation, t::Float64)
    @show t
    grid = sim.grid
    sim.P_err = 0.0
    sim.v_err = 0.0
    sim.E_err = 0.0
    p_avg = 0.0
    for p in grid.polygons
        p_avg += area(p)*p.P
    end
    for p in grid.polygons
        sim.E_err += p.mass*p.e
        sim.v_err += area(p)*norm_squared(p.v - v_exact(p.x, t))
        sim.P_err += area(p)*(p.P - p_avg - P_exact(p.x, t))^2
    end
    if isnan(sim.E0)
        sim.E0 = sim.E_err
    end
    sim.E_err -= sim.E0
    sim.P_err = sqrt(sim.P_err)
    sim.v_err = sqrt(sim.v_err)
    @show sim.v_err
    @show sim.P_err
    @show sim.E_err
    println()
    return
end

function main()
    sim = Simulation()
    run!(sim, dt, t_end, step!; 
        postproc! = postproc!,
        nframes = 20, 
        path = export_path,
        save_csv = true,
        save_points = true,
        save_grid = true,
        vtp_vars = (:P, :v)
    )
    @test sim.E_err < 1e-8
    @test sim.v_err < 0.01
    @test sim.P_err < 0.01
    @test ispath("results/simdata.csv")
    @test ispath("results/cells.pvd")
    @test ispath("results/cframe19.vtp")
    @test ispath("results/points.pvd")
    @test ispath("results/pframe19.vtp")
    rm("results", recursive=true)
    @info("deleted folder: results")
    return
end

end
