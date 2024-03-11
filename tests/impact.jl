module impact

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi

using DifferentialEquations
using StaticArrays
using Plots
using Parameters
using Base.Threads
using WriteVTK
using SmoothedParticles:rDwendland2

const sep = 1.0
const R1 = 0.3
const R2 = 0.3
const v1 = 0.5
const v2 = -0.5
const rho0 = 1.0
const dr = min(R1,R2)/70

const v_char = 2.0
const dt = 0.2*dr/v_char
const t_end =  1.0
const nframes = 100

const domain_size = 5.0

const P_stab = 0.0*rho0*v_char^2
const h_free = 1.5*dr
const h = h_free
const h_stab = h_free
const N_free = 10

const export_path = "results/impact"
include("../utils/populate.jl")
include("../utils/isolver.jl")

@with_kw mutable struct PhysFields
    mass::Float64 = 0.0
    v::RealVector = VEC0
    a::RealVector = VEC0
    P::Float64 = 0.0
    bc_type::Int = NOT_BC
    rho::Float64 = rho0
end

export_vars = (:v, :P, :mass)

function PhysFields(x::RealVector)
    return PhysFields(v = (x[1] < 0.0 ? v1*VECX : v2*VECX))
end

function find_energy(grid::VoronoiGrid)::Float64
    E = 0.0
    for p in grid.polygons
        E += p.var.mass*LagrangianVoronoi.norm_squared(p.var.v)
    end
    return E
end

function assign_mass!(p::VoronoiPolygon)
    p.var.mass = rho0*area(p)
end

function crop_free_polygons!(p::VoronoiPolygon)
    rad = maximum(e -> norm(e.v1 - p.x), p.edges) 
    if rad > h_free
        for k in 1:N_free
            theta = 2.0*pi*k/N_free
            y = p.x + 2.0*h_free*RealVector(cos(theta), sin(theta))
            LagrangianVoronoi.voronoicut!(p, y, 0)
        end
    end
    p.isbroken = false
end


function main()
    domain = Rectangle(xlims = (-domain_size, domain_size), ylims = (-domain_size, domain_size))
    grid = VoronoiGrid{PhysFields}(h, domain)
    grid.rr_max = (2.0*h_free/cos(pi/N_free))^2
    #grid.rr_max = Inf
    c1 = -0.5*sep*VECX
    c2 =  0.5*sep*VECX
    populate_circ!(grid, dr, charfun = (x -> norm_squared(x-c1) < R1^2), center = c1)
    populate_circ!(grid, dr, charfun = (x -> norm_squared(x-c2) < R2^2), center = c2)
    apply_unary!(grid, crop_free_polygons!)
    apply_unary!(grid, assign_mass!)
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
    nframe = 0
    
    for k = 0 : k_end
        apply_unary!(grid, move!)
        remesh!(grid)
        apply_unary!(grid, crop_free_polygons!)
        try
            find_pressure!(grid)
        catch e
            vtk_save(pvd_p)
            vtk_save(pvd_c)
            throw(e)
        end
        apply_binary!(grid, pressure_force!)
        apply_unary!(grid, accelerate!)

        if ((k_end - k) % k_frame == 0)
            t = k*dt
            @show t
            push!(time, t)
            push!(energy, find_energy(grid))
            println("energy error = ", energy[end]/energy[1] - 1.0)
            pvd_c[t] = export_grid(grid, string(export_path, "/cframe", nframe, ".vtp"), export_vars...)
            pvd_p[t] = export_points(grid, string(export_path, "/pframe", nframe, ".vtp"), export_vars...)
            nframe += 1
        end
    end
    vtk_save(pvd_p)
    vtk_save(pvd_c)
end

end