module double_shear

using WriteVTK, LinearAlgebra, Random, Match,  Parameters, Polyester
using SmoothedParticles:rDwendland2
using LaTeXStrings, DataFrames, CSV, Plots, Measures


include("../../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi

const rho0 = 1.0
const xlims = (0.0, 1.0)
const ylims = (0.0, 1.0)
const mu = 2e-4
const dr = 5e-3
const gamma = 1.4
const P0 = 100.0/gamma

const delta = 0.05
const slope = 30.0
const v_char = 2.0
const dt = 0.1*dr/v_char

const t_end = 1.8

const export_path = "results/double_shear2"
const nframes = 200

function ic!(p::VoronoiPolygon)
    p.v = v_init(p.x)
    p.rho = rho0
    p.mass = p.rho*area(p)
    p.P = 100.0/gamma
    p.e = 0.5*norm_squared(p.v) + p.P/(p.rho*(gamma - 1.0))
    p.mu = mu
end

function v_init(x::RealVector)::RealVector
    u =  (x[2] <= 0.5) ? tanh(slope*(x[2] - 0.25)) : tanh(slope*(0.75 - x[2]))
    v = delta*sin(2pi*x[1])
    return RealVector(u,v)
end

@kwdef mutable struct MyPolygon <: VoronoiPolygon
    x::RealVector        # position

    rho::Float64  = 0.0  # density
    v::RealVector = VEC0 # velocity
    e::Float64    = 0.0  # specific energy
    
    P::Float64    = 0.0  # pressure
    c2::Float64   = 0.0  # speed of sound squared
    D::RealMatrix = MAT0 # velocity deformation tensor
    S::RealMatrix = MAT0 # viscous stress
    mu::Float64 = 0.0    # dynamic viscosity
    dv::RealVector = VEC0

    # extensive vars
    mass::Float64 = 0.0
    momentum::RealVector = VEC0
    energy::Float64 = 0.0
    phase::Int = 0

    vort::Float64 = 0.0 # vorticity

    # sides of the polygon (in no particular order)
    edges::FastVector{Edge} = emptypolygon()
end

MyPolygon(x::RealVector)::MyPolygon = MyPolygon(x=x)
const MyGrid = VoronoiGrid{MyPolygon}

mutable struct Simulation <: SimulationWorkspace
    grid::MyGrid
    solver::CompressibleSolver{MyPolygon}
    E::Float64
    S::Float64
    E0::Float64
    S0::Float64
    first_step::Bool
    Simulation() = begin
        domain = Rectangle(xlims = xlims, ylims = ylims)
        grid = MyGrid(domain, dr, xperiodic = true, yperiodic = true)
        populate_lloyd!(grid, ic! = ic!)
        solver = CompressibleSolver(grid)
        return new(grid, solver, 0.0, 0.0, 0.0, 0.0, true)
    end
end

function get_vort!(grid::MyGrid)
    @batch for p in grid.polygons
        p.vort = 0.0
        for (q,e) in neighbors(p, grid)
            pq = LagrangianVoronoi.get_arrow(p.x,q.x,grid)
            lrr = lr_ratio(pq, e)
            m = 0.5*(e.v1 + e.v2)
            tmp = (p.v[2] - q.v[2])*VECX - (p.v[1] - q.v[1])*VECY
            p.vort -= lrr*dot(tmp, m - p.x) 
        end
        p.vort /= area(p)
    end
end

function step!(sim::Simulation, t::Float64)
    move!(sim.grid, dt)
    ideal_eos!(sim.grid, gamma)
    find_pressure!(sim.solver, dt)
    pressure_step!(sim.grid, dt)
    find_D!(sim.grid)
    viscous_step!(sim.grid, dt)
    relaxation_step!(sim.grid, dt)
    return
end

function postproc!(sim::Simulation, t::Float64)
    get_vort!(sim.grid)
    @show t
    grid = sim.grid
    sim.E = 0.0
    sim.S = 0.0
    for p in grid.polygons
        sim.E += p.mass*p.e
        sim.S += p.mass*(log(p.P/P0) - gamma*log(p.rho/rho0))
    end
    if sim.first_step
        sim.E0 = sim.E
        sim.S0 = sim.S
        sim.first_step = false
    end
    sim.E -= sim.E0
    sim.S -= sim.S0
    @show sim.E
    @show sim.S
    println()
    return
end

function main()
    sim = Simulation()
    run!(sim, dt, t_end, step!; 
        postproc! = postproc!,
        nframes = nframes, 
        path = export_path,
        save_csv = false,
        save_points = false,
        save_grid = true,
        vtp_vars = (:P, :v, :rho, :vort)
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end
