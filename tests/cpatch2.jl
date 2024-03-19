module cpatch

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi

using DifferentialEquations
using StaticArrays
using Plots
using Parameters
using Base.Threads
using WriteVTK
using SmoothedParticles:rDwendland2

const A0 = -0.5
const R = 1.0
const rho0 = 1.0
const dr = R/50

const v_char = abs(A0)*R
const dt = 0.2*dr/v_char
const t_end =  1.0
const nframes = 20

const P_stab = 1e-2*rho0*v_char^2

const h = 2*dr
const h_stab = h
const crop_R = 0.9*dr

const FREE_N = 20


const export_path = "results/cpatch"
include("../utils/populate.jl")
include("../utils/freecut_isolver.jl")

@with_kw mutable struct PhysFields
    mass::Float64 = 0.0
    v::RealVector = VEC0
    a::RealVector = VEC0
    P::Float64 = 0.0
    v_exact::RealVector = VEC0
    P_exact::Float64 = 0.0
    bc_type::Int = NOT_BC
    rho::Float64 = rho0
    x0::RealVector = VEC0
    isfree::Bool = false
    crop_R::Float64 = crop_R
end

export_vars = (:v, :P, :P_exact, :v_exact, :mass)

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
        E += p.var.mass*LagrangianVoronoi.norm_squared(p.var.v)
    end
    return E
end

function assign_mass!(p::VoronoiPolygon)
    p.var.mass = rho0*free_area(p)
end

function main()
    ode = ODEProblem(ellipse_ode, [R, R, A0], (0.0, t_end))
    ellipse = solve(ode, Rodas4(), reltol = 1e-8, abstol = 1e-8)
    domain = Rectangle(xlims = (-3*R, 3*R), ylims = (-3*R, 3*R))
    grid = VoronoiGrid{PhysFields}(2*crop_R, domain)

    populate_vogel!(grid, dr, charfun = (x -> norm_squared(x) < R^2))
    @show length(grid.polygons)
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
        @threads for p in grid.polygons
            #p.x = RealVector(p.var.x0[1]*e[1]/R, p.var.x0[2]*e[2]/R)
            p.var.P_exact = P_exact(p.x, e)
            #p.var.v = p.var.v_exact
            p.var.v_exact = v_exact(p.x, e)
        end
        remesh!(grid)
        limit_free_polygons!(grid)
        apply_local!(grid, stabilizer!, grid.h)
        find_pressure!(grid)
        apply_binary!(grid, pressure_force!)
        apply_unary!(grid, accelerate!)

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