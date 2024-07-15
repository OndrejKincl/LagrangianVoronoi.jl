# Gresho Vortex Benchmark

module gresho

using WriteVTK, LinearAlgebra, Random, Match,  Parameters, Polyester
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

const dt = 0.01*dr/v_char
const t_end =  0.2
const nframes = 200
const c0 = 10.0  # sound speed

const gamma = 1.4
const h0 = c0^2/(gamma - 1.0)
const P0 = rho0*c0^2/gamma
const alpha = 1e-3

const export_path = "results/gresho/N$(N)"

const h_stab = 2.4*dr
const P_stab = 0.01*rho0*v_char^2

# exact solution and initial velocity
function v_exact(x::RealVector)::RealVector
    omega = @match norm(x) begin
        r, if r < 0.2 end => 5.0
        r, if r < 0.4 end => 2.0/r - 5.0
        _ => 0.0
    end
    return omega*RealVector(-x[2], x[1])
end

function P_exact(x::RealVector)::Float64
    return @match norm(x) begin
        r, if r < 0.2 end => P0 + 12.5*r^2
        r, if r < 0.4 end => P0 + 4.0 + 4*log(5*r) - 20.0*r + 12.5*r^2
        _ => P0 - 2.0 + 4*log(2)
    end
end

# enforce inital condition on a VoronoiPolygon
function ic!(p::VoronoiPolygon)
    p.rho = rho0
    p.v = v_exact(p.x)
    p.P = P_exact(p.x)
    p.area = area(p)
    p.mass = p.rho*p.area
    p.e = p.P/((gamma - 1.0)*p.rho) + 0.5*norm_squared(p.v)
end

mutable struct Simulation <: SimulationWorkspace
    grid::GridNSc
    solver::CompressibleSolver
    E::Float64
    l2_err::Float64
    Simulation() = begin
        domain = Rectangle(xlims = xlims, ylims = ylims)
        grid = GridNSc(domain, dr)
        populate_circ!(grid)
        remesh!(grid)
        apply_unary!(grid, ic!)
        return new(grid, CompressibleSolver(grid, dt, verbose=0), 0.0, 0.0)
    end
end

function enforce_bc!(grid::VoronoiGrid)
    @batch for p in grid.polygons
        if !(xlims[1] + h_stab < p.x[1] < xlims[2] - h_stab)
            p.v = VEC0
        end
        if !(ylims[1] + h_stab < p.x[2] < ylims[2] - h_stab)
            p.v = VEC0
        end
    end
end

function step!(sim::Simulation, t::Float64)
    find_D!(sim.grid)
    find_mu!(sim.grid, dr, 1e-4)
    viscous_step!(sim.grid, dt)
    enforce_bc!(sim.grid)
    SPH_stabilizer!(sim.grid, P_stab, h_stab, dt)
    ideal_eos!(sim.grid, gamma)
    find_pressure!(sim.solver)
    pressure_step!(sim.grid, dt)
    #lloyd_step!(sim.grid, dt, 20.0)
    move!(sim.grid, dt)
    #find_rho!(sim.grid)
    return
end

# find energy and l2 error
function postproc!(sim::Simulation, t::Float64) 
    sim.l2_err = 0.0
    sim.E = 0.0
    for p in sim.grid.polygons
        sim.l2_err += p.area*norm_squared(p.v - v_exact(p.x))
        sim.E += p.mass*p.e
    end
    sim.l2_err = sqrt(sim.l2_err)
    @show sim.E
    @show sim.l2_err
end

function main()
    sim = Simulation()
    @time run!(sim, dt, t_end, step!; path = export_path, 
        vtp_vars = (:v, :P, :rho, :e, :rho_SPH, :u), csv_vars = (:E, :l2_err),
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
        louterinewidth = 2.0
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