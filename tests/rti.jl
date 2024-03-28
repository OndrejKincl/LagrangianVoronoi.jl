module rti

using WriteVTK, LinearAlgebra, Random, Match, DataFrames, CSV, Parameters
using SmoothedParticles:rDwendland2

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi

const rho_d = 1.0
const rho_u = 2.0
const Re = 400.0
const Fr = 1.0
#const atwood = (rho_u - rho_d)/(rho_u + rho_d)

const xlims = (0.0, 1.0)
const ylims = (0.0, 1.0)

const N = 100 #resolution
const dr = 1.0/N
const h = 2*dr

const v_char = 3.0
const l_char = 1.0
const dt = 0.1*dr/v_char
const tau_r = 0.1*l_char/v_char
const t_end =  0.5
const nframes = 100 #100

const export_path = "results/rti_chaos"
const PROJECTION_STEPS = 1


include("../utils/lloyd.jl")

function dividing_curve(x::Float64)::Float64
    return 0.5 - 0.15*sin(2*pi*x[1])
    #return 0.5 + 0.125*(0.7*cos(2*pi*x[1])+ 0.5*sin(2*pi*x[1]) + 0.3*cos(4*pi*x[1]) + 0.5*sin(4*pi*x[1]) + 0.1*cos(6*pi*x[1]) + 0.3*sin(6*pi*x[1]))
end

@with_kw mutable struct PhysFields
    mass::Float64 = 0.0
    v::RealVector = VEC0
    a::RealVector = VEC0
    P::Float64 = 0.0
    rho::Float64 = rho_d
    lloyd_dx::RealVector = VEC0
    lloyd_dv::RealVector = VEC0
end

include("../utils/isolver2.jl")

function PhysFields(x::RealVector)
    return PhysFields(rho = (x[2] < dividing_curve(x[1]) ? rho_d : rho_u))
end

export_vars = (:v, :P, :rho, :mass)

function find_energy(grid::VoronoiGrid)::Float64
    E = 0.0
    for p in grid.polygons
        E += 0.5*p.var.mass*norm_squared(p.var.v) + p.var.mass*(1.0/Fr^2)*(p.x[2] - ylims[1])
    end
    return E
end

function gravity!(p::VoronoiPolygon)
    p.var.v -= dt/(Fr^2)*VECY
end

function viscosity!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    p.var.a -= 1.0/(Re*p.var.mass)*lr_ratio(p,q,e)*(p.var.v - q.var.v)
end

function step!(grid::VoronoiGrid, solver::PressureSolver)
    lloyd_stabilization!(grid, tau_r)
    apply_unary!(grid, move!)
    remesh!(grid)
    apply_binary!(grid, viscosity!)
    apply_unary!(grid, accelerate!)
    apply_unary!(grid, gravity!)
    apply_unary!(grid, no_slip!)
    for _ in 1:PROJECTION_STEPS
        find_pressure!(solver, dt)
    end
    apply_binary!(grid, internal_force!)
    apply_unary!(grid, accelerate!)
    apply_unary!(grid, no_slip!)
end


function main()
    domain = Rectangle(xlims = xlims, ylims = ylims)
    grid = VoronoiGrid{PhysFields}(2*dr, domain)
    #grid.rr_max = Inf
    populate_lloyd!(grid, dr)
    for p in grid.polygons
        #p.var.mass = dr^2*p.var.rho
        p.var.mass = area(p)*p.var.rho
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
    solver = PressureSolver(grid)
    @time for k = 0 : k_end
        try
            step!(grid, solver)
        catch e
            vtk_save(pvd_p)
            vtk_save(pvd_c)
            throw(e)
        end
        if ((k_end - k) % k_frame == 0)
            t = k*dt
            @show t
            push!(time, t)
            push!(energy, find_energy(grid))
            println("relative energy = ", energy[end]/energy[1])
            pvd_c[t] = export_grid(grid, string(export_path, "/cframe", nframe, ".vtp"), export_vars...)
            pvd_p[t] = export_points(grid, string(export_path, "/pframe", nframe, ".vtp"), export_vars...)
            nframe += 1
        end
    end
    vtk_save(pvd_p)
    vtk_save(pvd_c)

    csv_data = DataFrame(time = time, energy = energy)
	CSV.write(string(export_path, "/energy_data.csv"), csv_data)

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end