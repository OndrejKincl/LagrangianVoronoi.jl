# Gresho Vortex Benchmark

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

const dt = 0.2*dr/v_char
const t_end =  1.0
const nframes = 100

const h_stab = 2.0*dr
const P_stab = 0.01*rho0*v_char^2

const export_path = "results/gresho$(N)"
const export_vars = (:v, :P, :mass)

# exact solution
function v_exact(x::RealVector)::RealVector
    omega = @match norm(x) begin
        r, if r < 0.2 end => 5.0
        r, if r < 0.4 end => 2.0/r - 5.0
        _ => 0.0
    end
    return omega*RealVector(-x[2], x[1])
end

# inital condition
function ic!(p::VoronoiPolygon)
    p.v = v_exact(p.x)
    p.rho = rho0
    p.mass = p.rho*area(p)
end



function find_l2_error(grid::VoronoiGrid)::Float64
    L2_error = 0.0
    for p in grid.polygons
        L2_error += area(p)*norm_squared(p.v - v_exact(p.x))
    end
    return sqrt(L2_error)
end

function find_energy(grid::VoronoiGrid)::Float64
    E = 0.0
    for p in grid.polygons
        E += p.mass*norm_squared(p.v)
    end
    return E
end

function SPH_stabilizer!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64)
    if (xlims[1]+h_stab < p.x[1] < xlims[2]-h_stab) && (ylims[1]+h_stab < p.x[2] < ylims[2]-h_stab)
	    p.v += -dt*q.mass*rDwendland2(h_stab,r)*(2*P_stab/rho0^2)*(p.x - q.x)
    end
end

function main()
    domain = Rectangle(xlims = xlims, ylims = ylims)
    grid = VanillaGrid(domain, dr)
    populate_circ!(grid)
    apply_unary!(grid, ic!)
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
    solver = PressureSolver(grid)

    @time for k = 0 : k_end
        move!(grid, dt)
        apply_local!(grid, SPH_stabilizer!, h_stab)
        find_pressure!(solver, dt)
        pressure_force!(grid, dt)
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

    # store velocity profile along midline
    vy = Float64[]
    vy_exact = Float64[]
    x_range = 0.0:(2*dr):xlims[2]
    for x1 in x_range
        x = RealVector(x1, 0.0)
        push!(vy, point_value(grid, x, p -> p.v[2]))
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
        color = :blue,
        linestyle = :dash,
        bottom_margin = 5mm
    )
    plot!(
        plt,
        csv_data.x,
        csv_data.vy,
        label = "simulation",
        markershape = :hex,
        markersize = 2,
    )
    savefig(plt, string(export_path, "/midline_plot.pdf"))
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end