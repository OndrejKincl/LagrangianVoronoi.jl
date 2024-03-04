module tagr
# Taylor-Green vortex

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using WriteVTK, LinearAlgebra, Random, Match, DataFrames, CSV, Plots
using SmoothedParticles:rDwendland2
using LaTeXStrings

const v_char = 1.0
const rho = 1.0
const xlims = (0.0, 2*pi)
const ylims = (0.0, 2*pi)
const N = 50 #resolution
const dr = 2*pi/N
const mu = 1e-4 #artificial viscosity

const dt = 0.2*dr/v_char
const t_end =  2.0
const nframes = 200
#const t_frame = max(dt, t_end/100)

const stabilize = true
const P_stab = 0.01*rho*v_char^2
const h_stab = 2.0*dr

const export_path = "results/tagr"

mutable struct PhysFields
    mass::Float64
    v::RealVector
    v_exact::RealVector
    a::RealVector
    P::Float64
    P_exact::Float64
    div::Float64
    fixpoint::Bool
    E::Float64
    invA::Float64
    PhysFields(x::RealVector) = begin
        return new(
            0.0,
            v_exact(x),
            v_exact(x),
            VEC0,
            P_exact(x),
            P_exact(x),
            0.0,
            false,
            0.0,
            0.0
        ) 
    end
end

function v_exact(x::RealVector)::RealVector
    u =  sin(x[1])*cos(x[2])
    v = -cos(x[1])*sin(x[2])
    return RealVector(u, v)
end

function P_exact(x::RealVector)::Float64
    return 0.25*(cos(2*x[1]) + cos(2*x[2]) - 2)
end

export_vars = (:v, :a, :P, :div, :v_exact, :P_exact)

function populate!(grid::VoronoiGrid)
    N = round(Int, (xlims[2] - xlims[1])/dr)
    M = round(Int, (ylims[2] - ylims[1])/dr)
    epsilon = dr/100
    xstep = (xlims[2] - xlims[1] - 2*epsilon)/N
    ystep = (ylims[2] - ylims[1] - 2*epsilon)/M
    for i in 0:N
        for j in 0:M
            x = RealVector(xlims[1] + epsilon + xstep*i, ylims[1] + epsilon + ystep*j)
            push!(grid.polygons, VoronoiPolygon{PhysFields}(x))
            #=
            if (i == div(N, 2) && j == div(M, 2))
                grid.polygons[end].var.fixpoint = true
            end
            =#
        end 
    end
    remesh!(grid)
    apply_unary!(grid, get_mass!)
    @show length(grid.polygons)
end

function populate_vogel!(grid::VoronoiGrid)
    rect = Rectangle(xlims = xlims, ylims = ylims)
    # add some points
    M = 0.5*pi*N*N
    for i in 1:M
        r = sqrt(2)*pi*sqrt(i/M)
        theta = 2.39996322972865332*i
        x = RealVector(pi + r*cos(theta), pi + r*sin(theta))
        if isinside(rect, x)
            push!(grid.polygons, VoronoiPolygon{PhysFields}(x))
        end    
    end
    remesh!(grid)
    apply_unary!(grid, get_mass!)
	return
end

function populate_hex!(grid::VoronoiGrid)
    rect = Rectangle(xlims = xlims, ylims = ylims)
    a = ((4/3)^0.25)*dr
    b = ((3/4)^0.25)*dr
    i_min = floor(Int, rect.xmin[1]/a) - 1
    j_min = floor(Int, rect.xmin[2]/b)
    i_max = ceil(Int, rect.xmax[1]/a)
    j_max = ceil(Int, rect.xmax[2]/b)
	for i in i_min:i_max, j in j_min:j_max
        x = RealVector((i + (j%2)/2)*a, j*b) + 0.5*dr*rand(RealVector)
        if isinside(rect, x)
            push!(grid.polygons, VoronoiPolygon{PhysFields}(x))
        end
        if i == div(i_max-i_min, 2) && j == div(j_max - j_min, 2)
            grid.polygons[end].var.fixpoint = true
        end
    end
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
    if isboundary(p)
        p.var.fixpoint = true
    end
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
    return p.var.fixpoint ? 1.0 : sum(e -> e.label == 0 ? 0.0 : (p.var.invA)*lr_ratio(p, grid.polygons[e.label], e), p.edges, init = 0.0)
end

function vector_element(p::VoronoiPolygon)::Float64
    return p.var.fixpoint ? P_exact(p.x) : -(rho/dt)*p.var.div
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
            p.var.v = v_exact(p.x)
            #=
            n = normal_vector(e)
            p.var.v += -dot(p.var.v, n)*n
            =#
        end
    end
end

function stabilizer!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64)
	p.var.v += -dt*q.var.mass*rDwendland2(h_stab,r)*(2*P_stab/rho^2)*(p.x - q.x)
end

function find_l2_error(grid::VoronoiGrid)::Float64
    L2_error = 0.0
    for p in grid.polygons
        p.var.v_exact = v_exact(p.x)
        p.var.P_exact = P_exact(p.x)
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
    populate_vogel!(grid)
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