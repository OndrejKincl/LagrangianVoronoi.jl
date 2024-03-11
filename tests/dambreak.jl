module dambreak

using WriteVTK, LinearAlgebra, Match, Parameters
using SmoothedParticles:rDwendland2
using Base.Threads


include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi


const dr = 2.0e-2         # average particle distance (decrease to refine, increase to speed up)
const h = 3.0*dr          # size of kernel support
const rho_water = 10.0        # fluid density
const rho_air = 1.0
const rho_wall = rho_water
const g = 9.8             # gravitation acceleration

##geometry parameters
const water_column_width = 1.0
const water_column_height = 2.0
const box_height = 3.0
const box_width = 4.0
const nlayers = 3.5
const wall_width = nlayers*dr

##temporal parameters
const v_char = sqrt(2*g*water_column_height)
const dt = 0.1*h/v_char #*(rho_air/rho_water)
const t_end = 10*dt
const dt_frame = max(dt, t_end/200)

const nframes = 100

const P_stab = 0.01*rho_water*v_char^2
const h_stab = h

const export_path = "results/dambreak"

const AIR = 0
const FLUID = 1
const WALL = 2

include("../utils/isolver.jl")
include("../utils/populate.jl")

@with_kw mutable struct PhysFields
    mass::Float64 = 0.0
    v::RealVector = VEC0
    a::RealVector = VEC0
    P::Float64 = 0.0
    bc_type::Int = NOT_BC
    rho::Float64 = 0.0
    p_type::Int = FLUID
end

export_vars = (:v, :P, :p_type)

function iswall(x::RealVector)::Bool
    tol = dr/10
    return (x[1] < tol) || (x[1] > box_width - tol) || (x[2] < tol) || (x[2] > box_height - tol)
end

function isfluid(x::RealVector)::Bool
    return (x[1] > 0.0) && (x[1] < water_column_width) && (x[2] > 0.0) && (x[2] < water_column_height)
end

function PhysFields(x::RealVector)
    p_type = AIR
    rho = rho_air
    if iswall(x)
        p_type = WALL
        rho = rho_wall
    end
    if isfluid(x)
        p_type = FLUID
        rho = rho_water
    end
    return PhysFields(p_type = p_type, rho = rho, mass = rho*dr^2)
end

function gravity!(p::VoronoiPolygon)
    if p.var.p_type == FLUID
        p.var.v -= dt*g*VECY
    end
end

function kill_air_convection!(p::VoronoiPolygon)
    if p.var.p_type == AIR
        p.var.v = VEC0
    end
end

function wall_force!(p::VoronoiPolygon)
    if isboundary(p)
        p.var.v = VEC0
    end
end

function find_energy(grid::VoronoiGrid)::Float64
    E = 0.0
    for p in grid.polygons
        E += p.var.mass*(LagrangianVoronoi.norm_squared(p.var.v) + (p.var.p_type == FLUID)*p.x[2]*g)
    end
    return E
end

function main()
    domain = Rectangle(xlims = (0.0, box_width), ylims = (0.0, box_height))
    grid = VoronoiGrid{PhysFields}(h, domain)
    grid.rr_max = Inf
    populate_rect!(grid, dr)
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

    for k = 0 : k_end
        apply_unary!(grid, move!)
        remesh!(grid)
        apply_local!(grid, stabilizer!, grid.h)
        apply_unary!(grid, gravity!)
        apply_unary!(grid, wall_force!)
        find_pressure!(grid, no_dirichlet = true)
        apply_binary!(grid, pressure_force!)
        apply_unary!(grid, accelerate!)
        apply_unary!(grid, wall_force!)
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