module rtest
include("../../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi, WriteVTK, Random, LinearAlgebra

const dr = 1.0/40
const dt = 1e-3
const t_end = 0.1
const nframes = 100
const gamma = 1.4
const export_path = "results/rtest"

const WHITE = 1
const BLACK = 2
const NEUTRAL = 3

function get_phase(x::RealVector)::Int
    (norm(x) >= 1.0) && return NEUTRAL
    (norm(x - 0.5*VECY) < 1/6) && return WHITE
    (norm(x + 0.5*VECY) < 1/6) && return BLACK
    s = sign(x[2])*sqrt(0.25 - (0.5 - abs(x[2]))^2)
    return (x[1] > s ? WHITE : BLACK)
end

function rho_init(x::RealVector)
    1.0 #+ 0.1*cos(x[1]) + 0.3*cos(3*x[2])
end

function v_init(x::RealVector)
    return sin(x[2])*VECX + sin(x[1])*VECY
end

function e_init(x::RealVector)
    eps = 10.0 + 0.7*cos(2*x[1]) + 0.4*cos(x[2])
    v = v_init(x)
    return eps + 0.5*norm_squared(v)
end

function ic!(p::VoronoiPolygon)
    p.rho = rho_init(p.x)
    p.v = v_init(p.x)
    p.e = e_init(p.x)
    p.D = MAT1
    p.mass = area(p)*p.rho
    p.phase = get_phase(p.x)
end

function get_errors(grid::VoronoiGrid)
    E_rho = 0.0
    E_v = 0.0
    E_e = 0.0
    for p in grid.polygons
        A = area(p)
        E_rho += A*abs(p.rho - rho_init(p.x))
        E_v += A*norm(p.v - v_init(p.x))
        E_e += A*abs(p.e - e_init(p.x))
    end
    return (E_rho, E_v, E_e)
end

mutable struct Simulation <: SimulationWorkspace
    grid::GridNSc
    rx::Relaxator{PolygonNSc}
    
    E_rho::Float64
    E_v::Float64
    E_e::Float64
    mass::Float64
    momemtum::RealVector
    energy::Float64
    S::Float64 #entropy
    rho_min::Float64
    dv_max::Float64
    white_area::Float64
    black_area::Float64
    Simulation() = begin
        dom = Rectangle(-(VECX + VECY), (VECX + VECY))
        grid = GridNSc(dom, dr)
        Random.seed!(42)
        populate_rand!(grid; ic! = ic!)
        # add some more random particles
        N = 3*count(p -> (p.phase == BLACK), grid.polygons)
        while N > 0
            x1 = 2rand() - 1.0
            x2 = 2rand() - 1.0
            x = RealVector(x1, x2)
            if get_phase(x) == BLACK
                push!(grid.polygons, PolygonNSc(x))
                N -= 1
            end
        end
        remesh!(grid)
        apply_unary!(grid, ic!)
        rx = Relaxator(grid, multiprojection=false)
        return new(grid, rx, 0.0, 0.0, 0.0, 0.0, VEC0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    end
end

function step!(sim::Simulation, t::Float64)
    if (t > 0.0)
        relaxation_step!(sim.rx, dt)
        remesh!(sim.grid)
    end
    ideal_eos!(sim.grid, gamma)
    return
end

function postproc!(sim::Simulation, t::Float64)
    sim.E_rho = 0.0
    sim.E_v = 0.0
    sim.E_e = 0.0
    sim.mass = 0.0
    sim.momemtum = VEC0
    sim.energy = 0.0
    sim.S = 0.0
    sim.rho_min = Inf
    sim.dv_max = 0.0
    sim.black_area = 0.0
    sim.white_area = 0.0
    for p in sim.grid.polygons
        A = area(p)
        sim.E_rho += A*abs(p.rho - rho_init(p.x))
        sim.E_v += A*norm(p.v - v_init(p.x))
        sim.E_e += A*abs(p.e - e_init(p.x))
        sim.mass += p.mass
        sim.energy += p.mass*p.e
        sim.momemtum += p.mass*p.v
        sim.S += p.mass*log(abs(p.P)/(abs(p.rho)^gamma))
        sim.rho_min = min(sim.rho_min, p.rho)
        sim.dv_max = max(sim.dv_max, norm(p.dv))
        if p.phase == BLACK
            sim.black_area += A
        end
        if p.phase == WHITE
            sim.white_area += A
        end
    end
    println()
    @show t
    @show sim.E_rho
    @show sim.E_v
    @show sim.E_e
    @show sim.mass
    @show sim.momemtum
    @show sim.energy
    @show sim.S
    @show sim.rho_min
    @show sim.dv_max
    @show sim.black_area
    @show sim.white_area
end


function main()
    sim = Simulation()
    run!(sim, dt, t_end, step!; path = export_path, 
        vtp_vars = (:v, :rho, :e, :P, :mass, :dv, :phase),
        postproc! = postproc!,
        nframes = nframes
    )
end

end