module gresho

using WriteVTK, LinearAlgebra, Random, Match, DataFrames, CSV, Plots, Parameters
using SmoothedParticles:rDwendland2
using LaTeXStrings


include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi

const v_char = 1.0
const rho0 = 1.0
const xlims = (-0.5, 0.5)
const ylims = (-0.5, 0.5)
const N = 100 #resolution
const dr = 1.0/N

const dt = 0.2*dr/v_char
const t_end =  0.5
const nframes = 100

const P_stab = 0.01*rho0*v_char^2
const h = 2.0*dr
const h_stab = h

const export_path = "results/gresho"

include("../utils/isolver.jl")
include("../utils/populate.jl")

@with_kw mutable struct PhysFields
    mass::Float64 = 0.0
    v::RealVector = VEC0
    a::RealVector = VEC0
    P::Float64 = 0.0
    bc_type::Int = NOT_BC
    rho::Float64 = rho0
end

function PhysFields(x::RealVector)
    return PhysFields(v = v_exact(x))
end

function v_exact(x::RealVector)::RealVector
    omega = @match norm(x) begin
        r, if r < 0.2 end => 5.0
        r, if r < 0.4 end => 2.0/r - 5.0
        _ => 0.0
    end
    return omega*RealVector(-x[2], x[1])
end

export_vars = (:v, :P)

function find_l2_error(grid::VoronoiGrid)::Float64
    L2_error = 0.0
    for p in grid.polygons
        L2_error += area(p)*LagrangianVoronoi.norm_squared(p.var.v - v_exact(p.x))
    end
    return sqrt(L2_error)
end

function find_energy(grid::VoronoiGrid)::Float64
    E = 0.0
    for p in grid.polygons
        E += p.var.mass*LagrangianVoronoi.norm_squared(p.var.v)
    end
    return E
end

function main()
    domain = Rectangle(xlims = xlims, ylims = ylims)
    grid = VoronoiGrid{PhysFields}(h, domain)
    populate_circ!(grid, dr)
    apply_unary!(grid, get_mass!)
    if !ispath(export_path)
        mkpath(export_path)
        @info "created a new path \""*export_path*"\""
    end 
    pvd_p = paraview_collection(export_path*"/points.pvd")
    pvd_c = paraview_collection(export_path*"/cells.pvd")
    nframe = 0
    energy = Float64[]
    l2_error = Float64[]
    time = Float64[]

    k_end = round(Int, t_end/dt)
    k_frame = max(1, round(Int, t_end/(nframes*dt)))

    
    for k = 0 : k_end
        apply_unary!(grid, move!)
        remesh!(grid)
        apply_local!(grid, stabilizer!, grid.h)
        apply_unary!(grid, no_slip!)
        find_pressure!(grid; no_dirichlet = true)
        apply_binary!(grid, pressure_force!)
        apply_unary!(grid, accelerate!)
        apply_unary!(grid, no_slip!)
        if ((k_end - k) % k_frame == 0)
            t = k*dt
            @show t
            push!(time, t)
            push!(l2_error, find_l2_error(grid))
            push!(energy, find_energy(grid))
            println("energy error = ", energy[end]/energy[1] - 1.0)
            println("l2 error = ", l2_error[end])
            pvd_c[t] = export_grid(grid, string(export_path, "/cframe", nframe, ".vtp"), export_vars...)
            pvd_p[t] = export_points(grid, string(export_path, "/pframe", nframe, ".vtp"), export_vars...)
            nframe += 1
        end
    end
    vtk_save(pvd_p)
    vtk_save(pvd_c)

    csv_data = DataFrame(time = time, energy = energy, l2_error = l2_error)
	CSV.write(string(export_path, "/error_data.csv"), csv_data)

    # export velocity profile along midline
    x_range = 0.0:(2*dr):xlims[2]
    vy_sim = Float64[]
    vy_exact = Float64[]
    for x1 in x_range
        x = RealVector(x1, 0.0)
        push!(vy_sim, point_value(grid, x, p -> p.var.v[2]))
        push!(vy_exact, v_exact(x)[2])
    end
    csv_data = DataFrame(x = x_range, vy_sim = vy_sim, vy_exact = vy_exact)
	CSV.write(string(export_path, "/midline_data.csv"), csv_data)
    plot_midline()

end

function plot_midline()
    csv_data = CSV.read(string(export_path, "/midline_data.csv"), DataFrame)
    plt = plot(
        csv_data.x, 
        csv_data.vy_exact, 
        label = "exact solution",
        xlabel = L"x",
        ylabel = L"v_y",
        color = :blue,
        linestyle = :dash
    )
    plot!(
        plt,
        csv_data.x,
        csv_data.vy_sim,
        label = "simulation result",
        markershape = :hex,
        markersize = 2,
        color = :orange
    )

    savefig(plt, string(export_path, "/midline_plot.pdf"))
end


end