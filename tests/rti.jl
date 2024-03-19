module rti

using WriteVTK, LinearAlgebra, Random, Match, DataFrames, CSV, Parameters
using SmoothedParticles:rDwendland2

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi

const rho0 = 1.0
const rho1 = 3.0
const atwood = (rho1 - rho0)/(rho1 + rho0)
const xlims = (-0.5, 0.5)
const ylims = (-1.0, 1.0)
const delta = 0.1
const N = 100 #resolution
const dr = 2.0/N
const h = 2*dr
const gravity = 1.0/atwood
const mu = 1e-3 #viscosity
const v_char = 2.0


const dt = 0.1*dr/v_char
const tau_r = 0.2
const t_end =  0.5
const nframes = 2

const export_path = "results/rti3"

include("../utils/isolver3.jl")
include("../utils/populate.jl")
include("../utils/lloyd.jl")

function dividing_curve(x::Float64)::Float64
    return -delta*cos(pi*x/xlims[2])
end

@with_kw mutable struct PhysFields
    mass::Float64 = 0.0
    v::RealVector = VEC0
    a::RealVector = VEC0
    P::Float64 = 0.0
    bc_type::Int = NOT_BC
    rho::Float64 = rho0
    lloyd_dx::RealVector = VEC0
    lloyd_dv::RealVector = VEC0
    rho_grad::RealVector = VEC0
    color::Float64 = 0.0
end

function PhysFields(x::RealVector)
    return PhysFields(rho = (x[2] < dividing_curve(x[1]) ? rho0 : rho1))
end

export_vars = (:v, :P, :rho, :mass, :color)

function find_energy(grid::VoronoiGrid)::Float64
    E = 0.0
    for p in grid.polygons
        E += 0.5*p.var.mass*LagrangianVoronoi.norm_squared(p.var.v) + p.var.mass*gravity*(p.x[2] - ylims[1])
    end
    return E
end

function gravity!(p::VoronoiPolygon)
    p.var.v -= dt*gravity*VECY
end

function viscosity!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    p.var.a -= mu/p.var.mass*lr_ratio(p,q,e)*(p.var.v - q.var.v)
end

function assign_color!(grid::VoronoiGrid)
    @threads for p in grid.polygons
        p.var.color = 0.0
        for e in p.edges
            if isboundary(e)
                continue
            end
            q = grid.polygons[e.label]
            if q.var.rho != p.var.rho
                p.var.color = 0.25
                break
            end
        end
        if p.var.rho == rho1
            p.var.color = 1.0 - p.var.color
        end
    end
end

function main()
    domain = Rectangle(xlims = xlims, ylims = ylims)
    grid = VoronoiGrid{PhysFields}(2*dr, domain)
    grid.rr_max = Inf
    populate_lloyd!(grid, dr)
    #apply_unary!(grid, get_mass!)
    for p in grid.polygons
        p.var.mass = dr^2*p.var.rho
    end
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

    
    @time for k = 0 : k_end
        lloyd_stabilization!(grid, tau_r)
        apply_unary!(grid, move!)
        remesh!(grid)
        apply_binary!(grid, viscosity!)
        apply_unary!(grid, accelerate!)
        apply_unary!(grid, gravity!)
        apply_unary!(grid, no_slip!)
        try
            find_pressure!(grid; no_dirichlet = true)
        catch e
            vtk_save(pvd_p)
            vtk_save(pvd_c)
            throw(e)
        end
        apply_binary!(grid, pressure_force!)
        apply_unary!(grid, accelerate!)
        apply_unary!(grid, no_slip!)
        if ((k_end - k) % k_frame == 0)
            t = k*dt
            @show t
            push!(time, t)
            push!(energy, find_energy(grid))
            println("energy error = ", energy[end]/energy[1] - 1.0)
            assign_color!(grid)
            pvd_c[t] = export_grid(grid, string(export_path, "/cframe", nframe, ".vtp"), export_vars...)
            pvd_p[t] = export_points(grid, string(export_path, "/pframe", nframe, ".vtp"), export_vars...)
            nframe += 1
        end
    end
    vtk_save(pvd_p)
    vtk_save(pvd_c)

    csv_data = DataFrame(time = time, energy = energy)
	CSV.write(string(export_path, "/error_data.csv"), csv_data)

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end