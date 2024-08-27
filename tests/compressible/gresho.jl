# Gresho Vortex Benchmark

module gresho

using WriteVTK, LinearAlgebra, Random, Match,  Parameters
using LaTeXStrings, DataFrames, CSV, Plots, Measures

include("../../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi

const v_char = 1.0
const l_char = 0.4
const rho0 = 1.0
const xlims = (-0.5, 0.5)
const ylims = (-0.5, 0.5)
const N = 200 #resolution
const dr = 1.0/N

const dt = 0.1*dr/v_char
const t_end =  3.0
const nframes = 100

const c0 = 1.0
const gamma = 1.4
const P0 = rho0*c0^2/gamma

const export_path = "results/gresho/norelax"

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
    p.v = v_exact(p.x)
    p.rho = rho0
    p.mass = p.rho*area(p)
    p.P = P_exact(p.x)
    p.e = 0.5*norm_squared(p.v) + p.P/(p.rho*(gamma - 1.0))
end

mutable struct Simulation <: SimulationWorkspace
    grid::GridNSc
    solver::CompressibleSolver{PolygonNSc}
    E::Float64
    l2_err::Float64
    Simulation() = begin
        domain = Rectangle(xlims = xlims, ylims = ylims)
        grid = GridNSc(domain, dr)
        populate_circ!(grid, ic! = ic!)
        return new(grid, CompressibleSolver(grid), 0.0, 0.0)
    end
end

function step!(sim::Simulation, t::Float64)
    move!(sim.grid, dt)
    ideal_eos!(sim.grid, gamma)
    find_pressure!(sim.solver, dt)
    pressure_step!(sim.grid, dt)
    find_D!(sim.grid)
    viscous_step!(sim.grid, dt)
    #relaxation_step!(sim.grid, dt)
    return
end

# find energy and l2 error
function postproc!(sim::Simulation, t::Float64) 
    sim.l2_err = 0.0
    sim.E = 0.0
    for p in sim.grid.polygons
        sim.l2_err += p.mass*norm_squared(p.v - v_exact(p.x))
        sim.E += p.mass*p.e
    end
    sim.l2_err = sqrt(sim.l2_err)
    @show sim.E
    @show sim.l2_err
end


function main()
    sim = Simulation()
    run!(sim, dt, t_end, step!; path = export_path, 
        vtp_vars = (:v, :P), csv_vars = (:E, :l2_err),
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
        linewidth = 2.0
    )
    plot!(
        plt,
        csv_data.x,
        csv_data.vy,
        label = "SILVA",
        markershape = :hex,
        markersize = 3,
        linewidth = 2.0
    )
    savefig(plt, string(export_path, "/midline_plot.pdf"))
end

function plot_midline_all_Mach()
    Ma0001 = CSV.read(string("results/gresho/Ma0.001/midline_data.csv"), DataFrame)
    Ma001 = CSV.read(string("results/gresho/Ma0.01/midline_data.csv"), DataFrame)
    Ma01 = CSV.read(string("results/gresho/Ma0.1/midline_data.csv"), DataFrame)
    Ma1 = CSV.read(string("results/gresho/Ma1.0/midline_data.csv"), DataFrame)
    #Ma10 = CSV.read(string("results/gresho/Ma10.0/midline_data.csv"), DataFrame)
    Ma100 = CSV.read(string("results/gresho/Ma100.0/midline_data.csv"), DataFrame)
    plt = plot(
        xlabel = L"x",
        ylabel = L"v_y",
        bottom_margin = 5mm,
    )
    plot!(plt,
        Ma1.x,
        Ma1.vy_exact,
        label = "exact solution",
        color = :black,
        linewidth = 1.0,
    )
    plot_y = [Ma0001.vy Ma001.vy Ma01.vy Ma1.vy Ma100.vy]
    #plot_y = (plot_y .- Ma1.vy_exact)
    plot!(plt,
        Ma1.x,
        plot_y,
        label = ["Ma = 0.001" "Ma = 0.01" "Ma = 0.1" "Ma = 1" "Ma = 100"],
        #markershape = :hex,
        #markersize = 2,
        linewidth = 1.0,
        color = [:blue :red :darkgreen :purple :orange],
        markershape = [:hex :circ :star7 :utriangle :dtriangle]
    )
    savefig(plt, string("results/gresho/midline_plot_all_Mach.pdf"))
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end