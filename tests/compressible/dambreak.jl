module dambreak
include("../../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using Match, Polyester, LinearAlgebra, WriteVTK

const g = 9.8 #gravitation acceleration
const mu = 1e-4 #dynamic viscosity
const hbox = 3.0
const wbox = 4.0
const hcol = 2.0
const wcol = 1.0

# phases
const LIQUID = 1
const GAS    = 0

struct TaitModel
    rho0::Float64
    c2::Float64
    gamma::Float64
end

const liquid_model = TaitModel(1000.0, 2500.0, 1.0)
const gas_model    = TaitModel(   1.0, 2500.0, 1.0)

const dr = 0.02
const v_char = 1000.0
const dt = 0.1*dr/v_char
const nframes = 200
const t_end = 1.0

const export_path = "results/dambreak"

@kwdef mutable struct PolygonNSc <: VoronoiPolygon
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

    # sides of the polygon (in no particular order)
    edges::FastVector{Edge} = emptypolygon()
end

PolygonNSc(x::RealVector)::PolygonNSc = PolygonNSc(x=x)
const GridNScM = VoronoiGrid{PolygonNSc}

function get_P(m::TaitModel, rho::Float64)::Float64
    return (m.rho0*m.c2/m.gamma)*((rho/m.rho0)^m.gamma - 1.0)
end

function get_c2(m::TaitModel, rho::Float64)::Float64
    return m.c2*abs(rho/m.rho0)^(m.gamma - 1.0)
end

function get_rho(m::TaitModel, P::Float64)::Float64
    return m.rho0*abs((m.gamma*P)/(m.rho0*m.c2) + 1.0)^(1.0/m.gamma)
end

function get_model(p::VoronoiPolygon)::TaitModel
    return (p.phase == LIQUID ? liquid_model : gas_model)
end

function P_init(x::RealVector)::Float64
    if (x[1] < wcol) && (x[2] < hcol)
        return g*gas_model.rho0*(hbox - hcol) + g*liquid_model.rho0*(hcol - x[2])
    end
    return g*gas_model.rho0*(hbox - x[2])
end

function ic!(p::VoronoiPolygon)
    isliquid = (p.x[1] <= wcol && p.x[2] <= hcol)
    p.phase = (isliquid ? LIQUID : GAS)
    m = get_model(p)
    p.P = P_init(p.x)
    p.rho = get_rho(m, p.P)
    p.mass = p.rho*area(p)
end

function eos!(grid::VoronoiGrid)
    @batch for p in grid.polygons
        p.rho = p.mass/area(p)
        m = get_model(p)
        p.c2 = get_c2(m, p.rho)
        p.P = get_P(m, p.rho)
    end
end

mutable struct Simulation <: SimulationWorkspace
    grid::GridNScM
    solver::CompressibleSolver{PolygonNSc}
    E::Float64
    Simulation() = begin
        xlims = (0.0, wbox)
        ylims = (0.0, hbox)
        domain = Rectangle(xlims = xlims, ylims = ylims)
        grid = GridNScM(domain, dr)
        populate_rect!(grid, ic! = ic!)
        return new(grid, CompressibleSolver(grid), 0.0)
    end
end



function step!(sim::Simulation, t::Float64)
    move!(sim.grid, dt)
    gravity_step!(sim.grid, -g*VECY, dt)
    eos!(sim.grid)
    find_pressure!(sim.solver, dt)
    pressure_step!(sim.grid, dt)
    find_D!(sim.grid)
    viscous_step!(sim.grid, dt)
    relaxation_step!(sim.grid, dt)
    return
end

# find energy
function postproc!(sim::Simulation, t::Float64) 
    sim.E = 0.0
    for p in sim.grid.polygons
        sim.E += 0.5*p.mass*norm_squared(p.v)
    end
    @show sim.E
end


function main()
    sim = Simulation()
    run!(sim, dt, t_end, step!; path = export_path, 
        vtp_vars = (:v, :P, :phase, :rho),
        postproc! = postproc!,
        nframes = nframes
    )
end

end