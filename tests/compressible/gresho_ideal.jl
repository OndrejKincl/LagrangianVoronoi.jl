# Gresho Vortex Benchmark

module gresho

using WriteVTK, LinearAlgebra, Random, Match,  Parameters
using SmoothedParticles:rDwendland2
using LaTeXStrings, DataFrames, CSV, Plots, Measures

include("../../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi

const v_char = 1.0
const l_char = 0.4
const rho0 = 1.0
const xlims = (-0.5, 0.5)
const ylims = (-0.5, 0.5)
const N = 100 #resolution
const dr = 1.0/N

const dt = 0.1*dr/v_char
const t_end =  1.0
const nframes = 100
const c0 = 10.0  # sound speed
const tau = 0.1
const gamma = 1.4

const h_stab = 2.0*dr
const P_stab = 0.05*rho0*v_char^2
const artificial_visc = 1e-4

const export_path = "results/gresho/N$(N)"

# exact solution and initial velocity
function v_exact(x::RealVector)::RealVector
    omega = @match norm(x) begin
        r, if r < 0.2 end => 5.0
        r, if r < 0.4 end => 2.0/r - 5.0
        _ => 0.0
    end
    return omega*RealVector(-x[2], x[1])
end

function h_exact(x::RealVector)::Float64
    return @match norm(x) begin
        r, if r < 0.2 end => 12.5*r^2
        r, if r < 0.4 end => 4.0 + 4*log(5*r) - 20.0*r + 12.5*r^2
        _ => -2.0 + 4*log(2)
    end
end

# enforce inital condition on a VoronoiPolygon
function ic!(p::VoronoiPolygon)
    p.v = v_exact(p.x)
    P0 = rho0*c0^2/gamma
    h0 = c0^2/(gamma - 1.0)
    h = h_exact(p.x) + h0
    p.rho = rho0*abs(h/h0)^(1.0/(gamma - 1.0))
    p.P = P0*abs(p.rho/rho0)^gamma
    #p.P = rho0*h + P0
    #p.rho = rho0*(p.P/P0)^(1.0/gamma)
    p.e = 0.5*norm_squared(p.v) + p.P/((gamma - 1.0)*p.rho)
end

function assign_mass!(p::VoronoiPolygon)
    p.mass = p.rho*area(p)
end

mutable struct Simulation <: SimulationWorkspace
    grid::GridNSc
    solver::CompressibleSolver
    E::Float64
    l2_err::Float64
    Simulation() = begin
        domain = Rectangle(xlims = xlims, ylims = ylims)
        grid = GridNSc(domain, dr)
        populate_rect!(grid, ic! = ic!)
        remesh!(grid)
        apply_unary!(grid, assign_mass!)
        return new(grid, CompressibleSolver(grid, dt, verbose=0), 0.0, 0.0)
    end
end

function SPH_stabilizer!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64)
    #(xlims[1] + h_stab < p.x[1] < xlims[2] - h_stab) || return
    #(ylims[1] + h_stab < p.x[2] < ylims[2] - h_stab) || return
	p.v += -dt*q.mass*rDwendland2(h_stab,r)*(P_stab/p.rho + P_stab/q.rho)*(p.x - q.x)
    return
end



function lloyd_stabilizer!(p::VoronoiPolygon)
    p.x = (tau*p.x + dt*centroid(p))/(tau + dt)
    return
end

function step!(sim::Simulation, t::Float64)
    #apply_unary!(sim.grid, lloyd_stabilizer!)
    move!(sim.grid, dt)
    ideal_eos!(sim.grid, gamma)
    viscous_force!(sim.grid, artificial_visc, dt) 
    #viscous_step!(sim.grid, dt, dr, 0.001)
    ideal_pressurefix!(sim.grid, gamma, dt)
    apply_local!(sim.grid, SPH_stabilizer!, h_stab)
    find_pressure!(sim.solver)
    energy_balance!(sim.grid, dt)
    pressure_force!(sim.grid, dt, stabilize=false)
    return
end

# find energy and l2 error
function postproc!(sim::Simulation, t::Float64) 
    sim.l2_err = 0.0
    sim.E = 0.0
    for p in sim.grid.polygons
        sim.l2_err += area(p)*norm_squared(p.v - v_exact(p.x))
        sim.E += p.mass*p.e
    end
    sim.l2_err = sqrt(sim.l2_err)
    @show sim.E
    @show sim.l2_err
end


function main()
    sim = Simulation()
    @time run!(sim, dt, t_end, step!; path = export_path, 
        vtp_vars = (:v, :P, :rho, :e), csv_vars = (:E, :l2_err),
        postproc! = postproc!,
        nframes = nframes
    )
    # store velocity profile along midline
    vy = Float64[]
    vy_exact = Float64[]
    x_range = 0.0:(2*dr):xlims[2]
    for x1 in x_range
        x = RealVector(x1, 0.0)
        push!(vy, point_value(sim.grid, x, p -> p.v[2]))
        push!(vy_exact, v_exact(x)[2])
    end
    csv_data = DataFrame(x = x_range, vy = vy, vy_exact = vy_exact)
	CSV.write(string(export_path, "/midline_data.csv"), csv_data)
    plot_midline()
end

function test(sim)
    remesh!(sim.grid)
    LagrangianVoronoi.refresh!(sim.solver)
    for k in 1:100
        LagrangianVoronoi.symm_test(sim.solver)
        LagrangianVoronoi.posdef_test(sim.solver)
    end
end

function plot_midline()
    csv_data = CSV.read(string(export_path, "/midline_data.csv"), DataFrame)
    plt = plot(
        csv_data.x,
        csv_data.vy_exact,
        xlabel = L"x",
        ylabel = L"v_y",
        label = "EXACT",
        color = :black,
        axisratio = 0.5,
        bottom_margin = 5mm,
        linewidth = 2.0
    )
    AREPO_ref = CSV.read("../reference/AREPO.csv", DataFrame)
    plot!(
        plt,
        AREPO_ref.x,
        AREPO_ref.vy,
        label = "AREPO (Springel et al)",
        markershape = :hex,
        markersize = 3,
        linewidth = 2.0
    )
    plot!(
        plt,
        csv_data.x,
        csv_data.vy,
        label = "ILVA",
        markershape = :hex,
        markersize = 3,
        linewidth = 2.0
    )
    savefig(plt, string(export_path, "/midline_plot.pdf"))
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end