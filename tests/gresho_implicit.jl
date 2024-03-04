module gresho

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using WriteVTK, LinearAlgebra, Random, Match, DataFrames, CSV, Plots
using SmoothedParticles:rDwendland2
using LaTeXStrings

const v_char = 1.0
const rho = 1.0
const xlims = (-0.5, 0.5)
const ylims = (-0.5, 0.5)
const N = 200 #resolution
const dr = 1.0/N
const mu = 1e-4 #artificial viscosity

const dt = 0.2*dr/v_char
const t_end =  3.0
const nframes = 100
#const t_frame = max(dt, t_end/100)

const stabilize = true
const P_stab = 0.01*rho*v_char^2
const h_stab = 2.0*dr

const export_path = "results/gresho_implicit"

mutable struct PhysFields
    mass::Float64
    v::RealVector
    a::RealVector
    P::Float64
    div::Float64
    fixpoint::Bool
    E::Float64
    invA::Float64
    PhysFields(x::RealVector) = begin
        return new(
            0.0,
            v_exact(x),
            VEC0,
            P_exact(x),
            0.0,
            false,
            0.0,
            0.0
        ) 
    end
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

export_vars = (:v, :a, :P, :div, :fixpoint)

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
    grid.polygons[1].var.fixpoint = true
    @show length(grid.polygons)
    remesh!(grid)
    apply_unary!(grid, get_mass!)
	return
end

function lr_ratio(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)::Float64
    l2 = norm_squared(e.v1 - e.v2)
    r2 = norm_squared(p.x - q.x)
    return sqrt(l2/r2)
end


# estimate divergence
function get_mass!(p::VoronoiPolygon)
    A = area(p)
    p.var.mass = rho*A
end

function get_invA!(p::VoronoiPolygon)
    A = area(p)
    p.var.invA = 1/A
end

function get_div!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    #z = 0.5*(p.x + q.x)
    p.var.div += (p.var.invA)*lr_ratio(p,q,e)*dot(m - q.x, p.var.v - q.var.v)
    #p.var.div += p.var.invA*lr_ratio(p,q,e)*dot(z - q.x, p.var.v - q.var.v)
    #p.var.div += p.var.invA*lr_ratio(p,q,e)*(dot(m - z, p.var.v - q.var.v) - 0.5*dot(p.x - q.x, p.var.v + q.var.v))
end

# tools to assemble linear system

function edge_element(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)::Float64
    return p.var.fixpoint ? 0.0 : -(p.var.invA)*lr_ratio(p,q,e)
end

function diagonal_element(grid::VoronoiGrid, p::VoronoiPolygon)::Float64
    return p.var.fixpoint ? 1.0 : sum(e -> e.label == 0 ? 0.0 : (p.var.invA)*lr_ratio(p, grid.polygons[e.label], e), p.edges)
end

function vector_element(p::VoronoiPolygon)::Float64
    return p.var.fixpoint ? 0.0 : -(rho/dt)*p.var.div
end

# pressure force
function internal_force!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    z = 0.5*(p.x + q.x)
    p.var.v += -(dt/p.var.mass)*lr_ratio(p,q,e)*(
        (q.var.P - p.var.P)*(m - z) 
        - 0.5*(p.var.P + q.var.P)*(p.x - q.x)
    )
end

function wall_force!(p::VoronoiPolygon)
    for e in p.edges
        if isboundary(e)
            n = normal_vector(e)
            p.var.v += -dot(p.var.v, n)*n
        end
    end
end

function stabilizer!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64)
	p.var.v += -dt*q.var.mass*rDwendland2(h_stab,r)*(2*P_stab/rho^2)*(p.x - q.x)
end

function find_l2_error(grid::VoronoiGrid)::Float64
    L2_error = 0.0
    for p in grid.polygons
        L2_error += area(p)*LagrangianVoronoi.norm_squared(p.var.v - v_exact(p.x))
    end
    return sqrt(L2_error)
end

function move!(p::VoronoiPolygon)
    p.x += dt*p.var.v
end

function find_energy!(p::VoronoiPolygon)
    p.var.E = 0.5*p.var.mass*dot(p.var.v, p.var.v)
end

function main()
    domain = Rectangle(xlims = xlims, ylims = ylims)
    grid = VoronoiGrid{PhysFields}(2*dr, domain)
    populate!(grid, domain, dr)
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
    # verlet time stepping
    @time for k = 0 : k_end
        apply_unary!(grid, move!)
        remesh!(grid)
        apply_unary!(grid, get_invA!)
        if stabilize
            apply_local!(grid, stabilizer!, h_stab)
        end
        apply_unary!(grid, wall_force!)
        apply_binary!(grid, get_div!)
        # assembling linear system
        A = assemble_matrix(grid, diagonal_element, edge_element)
        b = assemble_vector(grid, vector_element)
        # solving
        try
            pressure_vector = A\b
            for i in eachindex(grid.polygons)
                p = grid.polygons[i]
                p.var.P = pressure_vector[i]
                p.var.div = 0.0
            end
        catch
            @warn "Solver could not converge."
            break
        end
        apply_binary!(grid, internal_force!)
        apply_unary!(grid, wall_force!)
        if ((k_end - k) % k_frame == 0)
            t = k*dt
            @show t
            push!(time, t)
            apply_unary!(grid, find_energy!)
            push!(l2_error, find_l2_error(grid))
            push!(energy, sum(p -> p.var.E, grid.polygons))
            println("energy error = ", energy[end]/energy[1] - 1.0)
            println("l2 error = ", l2_error[end])
            pvd_c[t] = export_grid(grid, string(export_path, "/cframe", nframe, ".vtp"), export_vars...)
            pvd_p[t] = export_points(grid, string(export_path, "/pframe", nframe, ".vtp"), export_vars...)
            nframe += 1
        end
    end
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

    #p = plot(energy)
    #savefig(p, "results/gresho/energy.pdf")
    vtk_save(pvd_p)
    vtk_save(pvd_c)

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