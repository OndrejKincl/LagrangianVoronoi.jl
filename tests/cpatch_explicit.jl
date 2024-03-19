module cpatch_explicit

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi

using DifferentialEquations
using StaticArrays
using Plots
using Parameters
using Base.Threads
using WriteVTK
using SmoothedParticles:rDwendland2

const A0 = 0.5
const R = 1.0
const rho0 = 1.0
const dr = R/50

const v_char = A0*R
const c_sound = 10.0*v_char
const dt = 0.1*dr/c_sound
const t_end =  0.7
const nframes = 20

const P_stab = 0.01*rho0*v_char^2

const h_stab = 2*dr
const crop_R = 1.5*dr


const export_path = "results/cpatch_explicit"
include("../utils/populate.jl")
include("../utils/freecut_I.jl")

@with_kw mutable struct PhysFields
    mass::Float64 = 0.0
    v::RealVector = VEC0
    a::RealVector = VEC0
    P::Float64 = 0.0
    v_exact::RealVector = VEC0
    P_exact::Float64 = 0.0
    rho::Float64 = rho0
    x0::RealVector = VEC0
    isfree::Bool = false
    crop_R::Float64 = crop_R
end

export_vars = (:v, :P, :P_exact, :v_exact, :mass, :rho, :isfree) 

function PhysFields(x::RealVector)
    return PhysFields(x0 = x)
end

function ellipse_ode(e, p, t)
    return [
        -e[3]*e[1],
        +e[3]*e[2],
        (e[3]^2)*(e[1]^2 - e[2]^2)/(e[1]^2 + e[2]^2)
    ]
end

function v_exact(x::RealVector, e)::RealVector
    return RealVector(-e[3]*x[1], e[3]*x[2])
end

function P_exact(x::RealVector, e)::Float64
    return rho0*e[3]^2*(e[1]*e[2])/(e[1]^2 + e[2]^2)*(1.0 - (x[1]/e[1])^2 - (x[2]/e[2])^2)
end

function find_l2_error(grid::VoronoiGrid, e)::Float64
    L2_error = 0.0
    for p in grid.polygons
        L2_error += dr^2*LagrangianVoronoi.norm_squared(p.var.v - v_exact(p.x, e))
    end
    return sqrt(L2_error)
end

function find_energy(grid::VoronoiGrid)::Float64
    E = 0.0
    for p in grid.polygons
        E += 0.5*p.var.mass*dot(p.var.v, p.var.v)
        E += p.var.mass*c_sound^2*(log(abs(p.var.rho/rho0)) + rho0/p.var.rho - 1.0)
    end
    return E
end

function assign_mass!(p::VoronoiPolygon)
    p.var.mass = rho0*free_area(p)
end

function move!(p::VoronoiPolygon)
    p.x += dt*p.var.v
end

function find_pressure!(p::VoronoiPolygon)
   p.var.rho = p.var.mass/free_area(p) 
   p.var.P = c_sound^2*(p.var.rho - rho0)
end

function lr_ratio(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)::Float64
    l = free_length(p, e)
    r = norm(p.x - q.x)
    return l/r
end

function internal_force!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    z = 0.5*(p.x + q.x)
    p.var.v += (dt/p.var.mass)*lr_ratio(p,q,e)*((p.var.P - q.var.P)*(m - z) + 0.5*(p.var.P + q.var.P - 2*P_stab)*(p.x - q.x))
end

function main()
    ode = ODEProblem(ellipse_ode, [R, R, A0], (0.0, t_end))
    ellipse = solve(ode, Rodas4(), reltol = 1e-8, abstol = 1e-8)
    domain = Rectangle(xlims = (-3*R, 3*R), ylims = (-3*R, 3*R))
    grid = VoronoiGrid{PhysFields}(crop_R, domain)
    populate_circ!(grid, dr, charfun = (x -> norm_squared(x) < R^2))
    #populate_rect!(grid, dr, charfun = (x -> norm_squared(x) < R^2))
    limit_free_polygons!(grid)
    apply_unary!(grid, assign_mass!)
    @threads for p in grid.polygons
        p.var.v = v_exact(p.x, ellipse(0.0))
    end

    if !ispath(export_path)
        mkpath(export_path)
        @info "created a new path \""*export_path*"\""
    end 
    pvd_p = paraview_collection(export_path*"/points.pvd")
    pvd_c = paraview_collection(export_path*"/cells.pvd")

    k_end = round(Int, t_end/dt)
    k_frame = max(1, round(Int, t_end/(nframes*dt)))
    
    time = Float64[]
    energy = Float64[]
    l2_error = Float64[]
    nframe = 0
    
    for k = 0 : k_end
        e = ellipse(k*dt)
        apply_unary!(grid, move!)
        remesh!(grid)
        limit_free_polygons!(grid)
        apply_unary!(grid, find_pressure!)
        apply_binary!(grid, internal_force!)

        if ((k_end - k) % k_frame == 0)
            t = k*dt
            @show t
            push!(time, t)
            push!(energy, find_energy(grid))
            push!(l2_error, find_l2_error(grid, e))
            println("energy error = ", energy[end]/energy[1] - 1.0)
            println("l2 error = ", l2_error[end])
            pvd_c[t] = export_grid(grid, string(export_path, "/cframe", nframe, ".vtp"), export_vars...)
            pvd_p[t] = export_points(grid, string(export_path, "/pframe", nframe, ".vtp"), export_vars...)
            nframe += 1
        end
    end
    vtk_save(pvd_p)
    vtk_save(pvd_c)
end

end