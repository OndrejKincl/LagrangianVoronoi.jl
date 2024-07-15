# Gresho Vortex Benchmark

module vortex

using WriteVTK, LinearAlgebra, Random, Match,  Parameters, Polyester
using SmoothedParticles:rDwendland2
using LaTeXStrings, DataFrames, CSV, Plots, Measures

include("../../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi

const v_char = 1.0
const l_char = 1.2
const rho0 = 1.0
const xlims = (-l_char, l_char)
const ylims = (-l_char, l_char)
const N = 243 #resolution
const dr = 2l_char/N


const nframes = 20
const c0 = 1.0  # sound speed
const dt = 0.1*dr/max(c0, v_char)
const t_end =  0.2


const gamma = 1.4
const P0 = rho0*c0^2/gamma
const export_path = "results/vortex/N$(N)"

# exact solution and initial velocity
function v_exact(x::RealVector)::RealVector
    omega = @match norm(x) begin
        r, if r < 1.0 end => (1.0 - r^2)^3
        _ => 0.0
    end
    return omega*RealVector(-x[2], x[1])
end

function P_exact(x::RealVector)::Float64
    Pinf = P0 + rho0/14
    return @match norm(x) begin
        r, if r < 1.0 end => Pinf - (rho0/14)*(1.0 - r^2)^7
        _ => Pinf
    end
end

# enforce inital condition on a VoronoiPolygon
function ic!(p::VoronoiPolygon)
    v = v_exact(p.x)
    P = P_exact(p.x)
    p.A = area(p)
    p.M = rho0*p.A
    p.P = p.M*v
    p.E = p.A*P/(gamma - 1.0) + 0.5*p.M*norm_squared(v)
end

mutable struct Simulation <: SimulationWorkspace
    grid::GridMUSCL
    E::Float64
    P::RealVector
    M::Float64
    l2_err::Float64
    Simulation() = begin
        domain = Rectangle(xlims = xlims, ylims = ylims)
        grid = GridMUSCL(domain, dr)
        populate_circ!(grid)
        remesh!(grid)
        apply_unary!(grid, ic!)
        return new(grid, 0.0, VEC0, 0.0, 0.0)
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
    RK2_step!(sim.grid, dt; gamma = gamma)
    return
end

# find energy and l2 error
function postproc!(sim::Simulation, t::Float64) 
    sim.l2_err = 0.0
    sim.E = 0.0
    sim.P = VEC0
    sim.M = 0.0
    M_min = Inf
    for p in sim.grid.polygons
        sim.l2_err += p.A*norm_squared(p.u/p.rho - v_exact(p.x))
        sim.E += p.E
        sim.P += p.P
        sim.M += p.M
        M_min = min(M_min, p.M)
    end
    sim.l2_err = sqrt(sim.l2_err)
    @show sim.E
    @show norm(sim.P)
    @show sim.M
    @show M_min 
    @show sim.l2_err
end

function main()
    sim = Simulation()
    @time run!(sim, dt, t_end, step!; path = export_path, 
        vtp_vars = (:u, :rho, :e), csv_vars = (:E, :l2_err),
        postproc! = postproc!,
        nframes = nframes
    )
    # store velocity profile along midline
    vy = Float64[]
    vy_exact = Float64[]
    x_range = 0.0:(2*dr):xlims[2]
    for x1 in x_range
        x = RealVector(x1, 0.0)
        push!(vy, point_value(sim.grid, x, p -> p.u[2]/p.rho))
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
        bottom_margin = 5mm,
        louterinewidth = 2.0
    )
    plot!(
        plt,
        csv_data.x,
        csv_data.vy,
        label = "LV_MUSCL",
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