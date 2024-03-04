module gresho

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using WriteVTK, LinearAlgebra, Random, Match, DataFrames, CSV, Plots
using SmoothedParticles:rDwendland2
using LaTeXStrings, Parameters
using Base.Threads

const v_char = 1.0
const rho = 1.0
const xlims = (-0.5, 0.5)
const ylims = (-0.5, 0.5)
const N = 100 #resolution
const dr = 1.0/N

const dt = 0.2*dr/v_char
const t_end =  3.0
const nframes = 100
#const t_frame = max(dt, t_end/100)

const stabilize = true
const P_stab = 0.01*rho*v_char^2
const h_stab = 2.0*dr

const export_path = "results/gresho5"

include("implicit_solver.jl")

@with_kw mutable struct PhysFields
    mass::Float64 = 0.0
    v::RealVector = VEC0
    a::RealVector = VEC0
    E::Float64 = 0.0
    P::Float64 = 0.0
    invA::Float64 = 0.0
    isdirichlet::Bool = false
    L::RealMatrix = zero(RealMatrix)
    div::Float64 = 0.0
    err::RealVector = VEC0
end

function PhysFields(x::RealVector)::PhysFields
    return PhysFields(v = v_exact(x), P = P_exact(x))
end

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
        r, if r < 0.2 end => 5.0 + 12.5*r^2
        r, if r < 0.4 end => 9.0 + 12.5*r^2 - 20.0*r + 4.0*log(r/0.2)
        _ => 3.0 + 4.0*log(2.0)
    end
end

export_vars = (:v, :P, :isdirichlet, :div, :err)

function populate!(grid::VoronoiGrid, rect::Rectangle, dr::Float64)
    for r in (0.5*dr):dr:sqrt(2)
        k_max = round(Int, 2.0*pi*r*N)
        for k in 1:k_max
            theta = 2.0*pi*k/k_max
            x = RealVector(r*cos(theta), r*sin(theta))
            if isinside(rect, x)
                push!(grid.polygons, VoronoiPolygon{PhysFields}(x))
            end
        end
    end
    #grid.polygons[1].var.isdirichlet = true
    @show length(grid.polygons)
    remesh!(grid)
    apply_unary!(grid, get_mass!)
	return
end

function get_mass!(p::VoronoiPolygon)
    p.var.mass = rho*area(p)
    #=
    if isboundary(p)
        p.var.isdirichlet = true
        p.var.P = P_exact(p.x)
    end
    =#
end

function bc_condition!(p::VoronoiPolygon)
    if isboundary(p)
        p.var.v = VEC0
        p.var.a = VEC0
    end
end

function find_energy!(p::VoronoiPolygon)
    p.var.E = 0.5*p.var.mass*dot(p.var.v, p.var.v)
end


function find_l2_error(grid::VoronoiGrid)::Float64
    L2_error = 0.0
    for p in grid.polygons
        p.var.err = p.var.v - v_exact(p.x)
        L2_error += area(p)*LagrangianVoronoi.norm_squared(p.var.err)
    end
    return sqrt(L2_error)
end


function update!(grid::VoronoiGrid, dt::Float64)
    @threads for p in grid.polygons
        p.x += dt*p.var.v
    end
    remesh!(grid)
    apply_unary!(grid, bc_condition!)
    get_pressure_acc!(grid)
    @threads for p in grid.polygons
        p.var.v += dt*p.var.a
    end
    apply_unary!(grid, bc_condition!)
end

function main()
    domain = Rectangle(xlims = xlims, ylims = ylims)
    grid = VoronoiGrid{PhysFields}(2*dr, domain)
    populate!(grid, domain, dr)
    if !ispath(export_path)
        mkpath(export_path)
        @info "created a new path \""*export_path*"\""
    end 
    pvd_c = paraview_collection(export_path*"/cells.pvd")
    pvd_p = paraview_collection(export_path*"/points.pvd")
    nframe = 0
    energy = Float64[]
    l2_error = Float64[]
    time = Float64[]

    k_end = round(Int, t_end/dt)
    k_frame = max(1, round(Int, t_end/(nframes*dt)))
    # verlet time stepping
    @time for k = 0 : k_end
        try
            update!(grid, dt)
        catch e
            vtk_save(pvd_c)
            vtk_save(pvd_p) 
            throw(e)
        end
        if ((k_end - k) % k_frame == 0)
            t = k*dt
            @show t
            push!(time, t)
            apply_unary!(grid, find_energy!)
            push!(l2_error, find_l2_error(grid))
            push!(energy, sum(p -> p.var.E, grid.polygons, init = 0.0))
            println("energy error = ", energy[end]/energy[1] - 1.0)
            println("l2 error = ", l2_error[end])
            pvd_c[t] = export_grid(grid, string(export_path, "/cframe", nframe, ".vtp"), export_vars...)
            pvd_p[t] = export_points(grid, string(export_path, "/pframe", nframe, ".vtp"))
            nframe += 1
        end
    end
    vtk_save(pvd_c)
    vtk_save(pvd_p) 
    csv_data = DataFrame(time = time, energy = energy, l2_error = l2_error)
	CSV.write(string(export_path, "/error_data.csv"), csv_data)

    # export velocity profile along midline
    x_range = 0.0:(2*dr):xlims[2]
    vy_sim = Float64[]
    vy_exact = Float64[]
    for x1 in x_range
        x = RealVector(x1, 0.0)
        push!(vy_sim, LagrangianVoronoi.nearest_polygon(grid, x).var.v[2])
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