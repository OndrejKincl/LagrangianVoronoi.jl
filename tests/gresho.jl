module gresho

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using Match
using WriteVTK
using LinearAlgebra
using CSV, DataFrames
#using Plots
using StaticArrays

const v_char = 1.0
const c_sound = 20.0
const tau0 = 1.0
const xlims = (-1.0, 1.0)
const ylims = (-1.0, 1.0)
const N = 50 #resolution

const mu = 1e-4 #artificial viscosity

const r_stab = 0.2/N
const e_stab = 0.5*v_char^2

const dt = 0.5*r_stab/c_sound
const t_end =  0.4*pi
const t_frame = max(dt, t_end/20)

const stabilize = false
const P0 = 0.0

const export_path = "results/gresho"

mutable struct PhysFields
    mass::Float64
    tau::Float64
    v::RealVector
    a::RealVector
    P::Float64
    E::Float64
    PhysFields(x::RealVector) = begin
        # the initial condition
        
        omega, P = @match norm(x) begin
            r, if r < 0.2 end => (5.0, 5.0 + 12.5*r^2)
            r, if r < 0.4 end => (2.0/r - 5.0, 9.0 + 12.5*r^2 - 20.0*r + 4.0*log(r/0.2))
            _ => (0.0, 3.0 + 4.0*log(2.0))
        end
        P += P0
        return new(
            0.0,
            tau0 - P*(tau0/c_sound)^2, 
            omega*RealVector(-x[2], x[1]),
            VEC0,
            P,
            0.0,
        ) 
    end
end

const export_vars = (:mass, :tau, :v, :P, :E)

function getmass!(p::VoronoiPolygon)
    p.var.mass = area(p)/p.var.tau
end

function getpressure!(p::VoronoiPolygon)
    p.var.tau = area(p)/p.var.mass
    p.var.P = (c_sound/tau0)^2*(tau0 - p.var.tau)
    #p.var.E = 0.5*p.var.mass*dot(p.var.v, p.var.v)
    #p.var.E += 0.5*p.var.mass*(c_sound/tau0)^2*(tau0 - p.var.tau)^2
    #p.var.a += -surface_element(p)*p.var.P/p.var.mass
end

function internal_force!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    l = len(e)
    r = norm(p.x - q.x)
    z = 0.5*(p.x + q.x)
    p.var.a += -(1.0/p.var.mass)*(l/r)*(
        (q.var.P - p.var.P)*(m - z) 
        - 0.5*(p.var.P + q.var.P)*(p.x - q.x)
    )
    if (r < r_stab) && stabilize
        #phi = (r/r_stab - 1.0)^2
        dphi = (2.0/r_stab)*(r/r_stab - 1.0)
        m_pq = 0.5*(p.var.mass + q.var.mass)
        f_pq = (2.0/r)*e_stab*m_pq*abs(dphi)*(p.x - q.x)
        p.var.a += (1.0/p.var.mass)*f_pq
    end
    #viscosity
    p.var.a += (mu/p.var.mass)*(l/r)*(q.var.v - p.var.v)
end

function move!(p::VoronoiPolygon)
    p.x += dt*p.var.v
    p.var.a = VEC0
end

function accelerate!(p::VoronoiPolygon)
    p.var.v += 0.5*dt*p.var.a
end

function wall_force!(p::VoronoiPolygon)
    for e in p.edges
        if isboundary(e)
            n = normal_vector(e)
            p.var.a += -len(e)*(p.var.P/p.var.mass)*n
            r = abs(dot(e.v1 - p.x, n))
            if r < r_stab && stabilize
                # elastic rebound
                p.var.a -= (2.0/dt)*pos(dot(p.var.v, n))*n
            end
        end
    end
end

@inline function pos(x::Float64)::Float64
    return (x > 0.0 ? x : 0.0)
end

function compute_force!(grid::VoronoiGrid)
    apply_unary!(grid, getpressure!)
    apply_binary!(grid, internal_force!)
    apply_unary!(grid, wall_force!)
end

function find_energy!(p::VoronoiPolygon)
    p.var.E = 0.5*p.var.mass*dot(p.var.v, p.var.v) + 0.5*p.var.mass*(c_sound/tau0)^2*(tau0 - p.var.tau)^2
end


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
    @show length(grid.polygons)
    remesh!(grid)
    apply_unary!(grid, getmass!)
	return
end

function findL2_error(grid::VoronoiGrid)::Float64
    L2_error = 0.0
    for p in grid.polygons
        omega = @match norm(p.x) begin
            r, if r < 0.2 end => 5.0
            r, if r < 0.4 end => 2.0/r - 5.0
            _ => 0.0
        end
        v0 = omega*RealVector(-p.x[2], p.x[1])
        L2_error += area(p)*LagrangianVoronoi.norm_squared(p.var.v - v0)
    end
    return sqrt(L2_error)
end

function main()
    rect = Rectangle(xlims = xlims, ylims = ylims)
    grid = VoronoiGrid{PhysFields}(2.0/N, rect)
    populate!(grid, rect, 1/N)
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
    compute_force!(grid)
    # verlet time stepping
    @time for k = 0 : round(Int, t_end/dt)
        apply_unary!(grid, accelerate!)
        apply_unary!(grid, move!)
        remesh!(grid)
        compute_force!(grid)
        if (k % round(Int, t_frame/dt) == 0)
            t = k*dt
            @show t
            push!(time, t)
            apply_unary!(grid, find_energy!)
            push!(l2_error, findL2_error(grid))
            push!(energy, sum(p -> p.var.E, grid.polygons))
            println("energy error = ", energy[end]/energy[1] - 1.0)
            println("l2 error = ", l2_error[end])
            pvd_c[t] = export_grid(grid, string(export_path, "/cframe", nframe, ".vtp"), export_vars...)
            pvd_p[t] = export_points(grid, string(export_path, "/pframe", nframe, ".vtp"), export_vars...)
            nframe += 1
        end
        apply_unary!(grid, accelerate!)
    end
    csv_data = DataFrame(time = time, energy = energy, l2_error = l2_error)
	CSV.write(string(export_path, "/data.csv"), csv_data)
    #p = plot(energy)
    #savefig(p, "results/gresho/energy.pdf")
    vtk_save(pvd_p)
    vtk_save(pvd_c)
end





end