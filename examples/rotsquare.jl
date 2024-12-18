#=
# Example 11: Rotating square
```@raw html
    <img src='../assets/rotsquare.png' alt='missing' width="75%"><br>
```

A square patch of fluid with initial velocity field corresponding to a solid body rotation
and surrounded by air deforms into a star-shaped patch. The result is to be compared 
qualitatively with a [reference solution](https://www.sciencedirect.com/science/article/pii/S0045782519300702).
=#


module rotsquare

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi

using StaticArrays
using Plots
using Parameters
using Base.Threads
using WriteVTK
using LinearAlgebra
using Polyester
using LaTeXStrings, CSV, DataFrames

const R = 0.5
const Rho = 1000.0
const rho = Rho/800
const dr = R/20

const v_char = 1.0
const dt = 0.1*dr/v_char
const t_end =  2.0
const nframes = 100
const smoothing_length = 3dr

const export_path = "results/rotsquare"
const xlims = (-1.5, 1.5)
const ylims = (-1.5, 1.5)

const WATER = 0
const AIR = 1

function ic!(p::VoronoiPolygon)
    p.c2 = Inf
    r = max(abs(p.x[1]), abs(p.x[2]))
    p.phase = (r < R) ? WATER : AIR
    p.rho = (p.phase == WATER ? Rho : rho)
    p.mass = p.rho*area(p)
    p.v = v_init(p.x)
end

mutable struct Simulation <: SimulationWorkspace
    grid::GridNS
    solver::PressureSolver{PolygonNS}
    Simulation() = begin
        domain = Rectangle(xlims = xlims, ylims = ylims)
        grid = GridNS(domain, dr)
        populate_rect!(grid, ic! = ic!)
        solver = PressureSolver(grid, verbose=false)
        return new(grid, solver, 0.0, true)
    end
end

function v_init(x::RealVector)::RealVector
    r = max(abs(x[1]), abs(x[2]))
    omega = (r < R ? 1.0 : 0.0)
    return omega*RealVector(x[2], -x[1])
end

function step!(sim::Simulation, t::Float64)
    move!(sim.grid, dt)
    find_rho!(sim.grid)
    find_pressure!(sim.solver, dt)
    pressure_step!(sim.grid, dt)
    phase_preserving_remapping!(sim.grid, dt, smoothing_length)
end

function postproc!(sim::Simulation, t::Float64)
    @show t
    println("Mesh quality = $(mesh_quality(sim.grid))")
    println()
end


function main()
    sim = Simulation()
    run!(sim, dt, t_end, step!; 
        path = export_path, 
        postproc! = postproc!,
        vtp_vars = (:v, :P, :rho, :phase),
        nframes = nframes,
        save_points = true
    )
end

end