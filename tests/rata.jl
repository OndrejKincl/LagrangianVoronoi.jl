module rata

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using WriteVTK, LinearAlgebra, Random, Match, DataFrames, CSV, Plots
using SmoothedParticles:rDwendland2
using LaTeXStrings

const v_char = 1.0
const rho = 1.0
const xlims = (-1.0/8, 1.0/8)
const ylims = (-0.3, 0.3)
const N = 300 #resolution
const dr = 1.0/N
const mu = 1e-4 #viscosity

const dt = 0.2*dr/v_char
const t_end =  3.0
const nframes = 200
#const t_frame = max(dt, t_end/100)

const stabilize = true
const P_stab = 0.01*rho*v_char^2
const h_stab = 2.0*dr

const g_force = 0.5*VECY

const export_path = "results/rata"

mutable struct PhysFields
    mass::Float64
    v::RealVector
    a::RealVector
    P::Float64
    div::Float64
    fixpoint::Bool
    isheavy::Bool
    E::Float64
    invA::Float64
    PhysFields(x::RealVector) = begin
        return new(
            0.0,
            VEC0,
            VEC0,
            0.0,
            0.0,
            false,
            false,
            0.0,
            0.0
        ) 
    end
end

export_vars = (:v, :P, :isheavy, :fixpoint)


function dividing_curve(x::Float64)::Float64
    return -0.05*cos(8*pi*x)
end

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
            if (i == div(N, 2) && j == div(M, 2))
                grid.polygons[end].var.fixpoint = true
            end
            if (x[2] > dividing_curve(x[1]))
                grid.polygons[end].var.isheavy = true
            end
        end 
    end
    remesh!(grid)
    apply_unary!(grid, get_mass!)
    @show length(grid.polygons)
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

function gravity!(p::VoronoiPolygon)
    p.var.a += (p.var.isheavy ? -1.0 : 1.0)*g_force
    p.var.v += dt*p.var.a
    p.var.a = VEC0
end

function viscosity!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    p.var.a += (mu/p.var.mass)*lr_ratio(p,q,e)*(q.var.v - p.var.v)
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
    #=
    for e in p.edges
        if isboundary(e)
            n = normal_vector(e)
            p.var.v += -dot(p.var.v, n)*n
        end
    end
    =#
    if isboundary(p)
        p.var.v = VEC0
    end
end

function stabilizer!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64)
	p.var.v += -dt*q.var.mass*rDwendland2(h_stab,r)*(2*P_stab/rho^2)*(p.x - q.x)
end

function move!(p::VoronoiPolygon)
    p.x += dt*p.var.v
end

function find_energy!(p::VoronoiPolygon)
    p.var.E = 0.5*p.var.mass*dot(p.var.v, p.var.v) 
    p.var.E += p.var.mass*(p.var.isheavy ? 1.0 : -1.0)*0.5*p.x[2]*norm(g_force)
end

function main()
    domain = Rectangle(xlims = xlims, ylims = ylims)
    grid = VoronoiGrid{PhysFields}(2*dr, domain)
    populate!(grid)
    if !ispath(export_path)
        mkpath(export_path)
        @info "created a new path \""*export_path*"\""
    end 
    pvd_p = paraview_collection(export_path*"/points.pvd")
    pvd_c = paraview_collection(export_path*"/cells.pvd")
    nframe = 0
    energy = Float64[]
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
        apply_binary!(grid, viscosity!)
        apply_unary!(grid, gravity!)
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
            push!(energy, sum(p -> p.var.E, grid.polygons))
            println("energy error = ", energy[end] - energy[1])
            pvd_c[t] = export_grid(grid, string(export_path, "/cframe", nframe, ".vtp"), export_vars...)
            pvd_p[t] = export_points(grid, string(export_path, "/pframe", nframe, ".vtp"), export_vars...)
            nframe += 1
        end
    end
    csv_data = DataFrame(time = time, energy = energy)
	CSV.write(string(export_path, "/error_data.csv"), csv_data)

    #p = plot(energy)
    #savefig(p, "results/gresho/energy.pdf")
    vtk_save(pvd_p)
    vtk_save(pvd_c)

end

end