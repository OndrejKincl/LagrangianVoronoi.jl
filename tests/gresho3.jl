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
const N = 200 #resolution
const dr = 1.0/N

const dt = 0.2*dr/v_char
const tau_r = 0.2*l_char/v_char
const t_end =  3.0
const nframes = 100

const h = 2.0*dr
const export_path = "results/gresho$(N)"

const h_stab = h
const P_stab = 0.01*rho0*v_char^2

include("../utils/parallel_settings.jl")
include("../utils/lloyd.jl")


@with_kw mutable struct PhysFields
    mass::Float64 = 0.0
    v::RealVector = VEC0
    a::RealVector = VEC0
    P::Float64 = 0.0
    rho::Float64 = rho0
    lloyd_dx::RealVector = VEC0
    lloyd_dv::RealVector = VEC0
end
include("../utils/isolver.jl")


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


function stabilizer!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64)
	p.var.v += -dt*q.var.mass*rDwendland2(h_stab,r)*(2*P_stab/rho0^2)*(p.x - q.x)
end


function stabilization_test()
    domain = Rectangle(xlims = xlims, ylims = ylims)
    grid = VoronoiGrid{PhysFields}(h, domain)
    populate_circ!(grid, dr)
    remesh!(grid)
    apply_unary!(grid, get_mass!)
    p0 = grid.polygons[1]
    x0 = p0.x
    for p in grid.polygons
        p.var.P = 0.5*norm_squared(p.x - x0)
    end
    apply_binary!(grid, internal_force!)
    c0 = centroid(p0)
    acc = p0.var.rho*dot(p0.var.a, c0 - x0)/norm_squared(c0 - x0)
    @show acc
    stabilize!(grid)
    acc = p0.var.rho*dot(p0.var.a, c0 - x0)/norm_squared(c0 - x0)
    @show acc
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
    
    solver = PressureSolver(grid)
    vy_data = Vector{Vector{Float64}}()
    @time for k = 0 : k_end
        move!(grid, dt)
        #lloyd_stabilization!(grid, tau_r)
        remesh!(grid)
        apply_local!(grid, stabilizer!, h_stab)
        #apply_unary!(grid, get_mass!)
        #stabilize!(grid)
        #apply_unary!(grid, accelerate!)
        apply_unary!(grid, no_slip!)
        try
            find_pressure!(solver, dt)
        catch e
            vtk_save(pvd_p)
            vtk_save(pvd_c)
            throw(e)
        end
        apply_binary!(grid, internal_force!)
        stabilize!(grid)
        accelerate!(grid, dt)
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
            # store velocity profile along midline
            vy = Float64[]
            x_range = 0.0:(2*dr):xlims[2]
            for x1 in x_range
                x = RealVector(x1, 0.0)
                push!(vy, point_value(grid, x, p -> p.var.v[2]))
            end
            push!(vy_data, vy) 
        end
    end
    vtk_save(pvd_p)
    vtk_save(pvd_c)

    csv_data = DataFrame(time = time, energy = energy, l2_error = l2_error)
	CSV.write(string(export_path, "/error_data.csv"), csv_data)

    # plot midline
    dk = div(length(vy_data), 3)
    csv_data = DataFrame(x = 0.0:(2*dr):xlims[2], 
        vy_t0 = vy_data[1], 
        vy_t1 = vy_data[1 + dk], 
        vy_t2 = vy_data[1 + 2*dk],
        vy_t3 = vy_data[end]
    )
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
        xlabel = L"x",
        ylabel = L"v_y",
        color = :blue,
        linestyle = :dash,
        bottom_margin = 5mm
    )
    plot!(
        plt,
        csv_data.x,
        csv_data.vy_t3,
        label = "simulation",
        markershape = :hex,
        markersize = 2,
    )
    #=
    for (vy, label) in (
            (csv_data.vy_t0, L"t=0"), 
            (csv_data.vy_t1, L"t=1"), 
            (csv_data.vy_t2, L"t=2"), 
            (csv_data.vy_t3, L"t=3")
        )
        plot!(
            plt,
            csv_data.x,
            vy,
            label = label,
            markershape = :hex,
            markersize = 2,
        )
    end
    =#

    savefig(plt, string(export_path, "/midline_plot.pdf"))
end

function convergence_plot()
    font_size = 12
    plt = plot(
        xlabel = "x-coordinate",
        ylabel = "y-velocity",
        bottom_margin = 5mm,
        xtickfont=font(font_size), 
        ytickfont=font(font_size), 
        guidefont=font(font_size), 
        legendfont=font(font_size)
    )
    plt_energy = plot(
        range(0.0, 3.0, length = 3),
        ones(3),
        label = "analytic",
        xlabel = "time",
        ylabel = "energy",
        ylims = (0.6, 1.0),
        xtickfont=font(font_size), 
        ytickfont=font(font_size), 
        guidefont=font(font_size), 
        legendfont=font(font_size)
    )
    i = 1
    colors = cgrad(:darkrainbow, 3, categorical = true)
    shapes = [:circ, :hex, :square, :star4, :star5, :utriangle, :dtriangle, :pentagon, :rtriangle, :ltriangle]
    for res in (200, 100, 50)
        path = "results/gresho$(res)/midline_data.csv"
        energy_path = "results/gresho$(res)/error_data.csv"
        data = CSV.read(path, DataFrame)
        if i == 1
            plot!(
                plt,
                data.x,
                data.vy_t0,
                label = "analytic",
            )
        end
        plot!(
            plt,
            data.x,
            data.vy_t3,
            label = "N = $(res)",
            markershape = shapes[i],
            markersize = 2,
            color = colors[i]
        )
        energy_data = CSV.read(energy_path, DataFrame)
        plot!(
            plt_energy,
            energy_data.time[1:4:end],
            energy_data.energy[1:4:end]./energy_data.energy[1],
            label = "N = $(res)",
            markershape = shapes[i],
            markersize = 2,
            color = colors[i]
        )
        i += 1
    end
    savefig(plt, "gresho_convergence.pdf")
    savefig(plt_energy, "gresho_energy.pdf")
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end