module gresho

using WriteVTK, LinearAlgebra, Random, Match,  Parameters
using SmoothedParticles:rDwendland2
using LaTeXStrings, DataFrames, CSV, Plots, Measures


include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi

const v_char = 1.0
const l_char = 0.4
const rho0 = 1.0
const xlims = (-0.5, 0.5)
const ylims = (-0.5, 0.5)
const N = 100 #resolution
const dr = 1.0/N

const dt = 0.05*dr/v_char
const tau_r = l_char/v_char
const t_end =  1.0
const nframes = 100

const h = 2.0*dr

const P_stab = 0.01*rho0*v_char^2
const h_stab = 2.0*dr

const export_path = "results/gresho_imex"

include("../utils/parallel_settings.jl")
include("../utils/lloyd.jl")


@with_kw mutable struct PhysFields
    mass::Float64 = 0.0
    v::RealVector = VEC0
    a::RealVector = VEC0
    P::Float64 = 0.0
    rho::Float64 = rho0
    x_old::RealVector = VEC0
    v_old::RealVector = VEC0
    a_old::RealVector = VEC0
    lloyd_dx::RealVector = VEC0
    lloyd_dv::RealVector = VEC0
end
include("../utils/cnab.jl")


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

    cnab = CNAB_scheme(grid, dt)
    @time for k = 1 : k_end
        try
            step!(cnab)
        catch e
            vtk_save(pvd_p)
            vtk_save(pvd_c)
            throw(e)
        end
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


function tri_area(a::RealVector, b::RealVector, c::RealVector)::Float64
    return 0.5*abs(LagrangianVoronoi.cross2(b - a, c - a))
end

function centroid(p::VoronoiPolygon)::RealVector
    A = 0.0
    c = VEC0
    for e in p.edges
        dA = tri_area(p.x, e.v1, e.v2)
        A += dA
        c += dA*(p.x + e.v1 + e.v2)/3
    end
    return c/A
end


function stabilize!(grid::VoronoiGrid)
    @threads for p in grid.polygons
        LapP = 0.0 #laplacian of pressure
        for e in p.edges
            if isboundary(e)
                continue
            end
            q = grid.polygons[e.label]
            LapP -= lr_ratio(p,q,e)*(p.var.P - q.var.P)
        end
        if LapP > 0.0
            c = centroid(p)
            p.var.a += 1.5*LapP/(p.var.mass)*(c - p.x)
        end
    end
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
        linestyle = :dash,
        bottom_margin = 5mm
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


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end