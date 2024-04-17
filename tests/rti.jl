# Rayleigh-Taylor instability

module rti

using WriteVTK, LinearAlgebra, Random, Match, DataFrames, CSV, Parameters
using SmoothedParticles:rDwendland2

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi

const rho_d = 1.0
const rho_u = 1.8
const Re = 420.0
const Fr = 1.0
#const atwood = (rho_u - rho_d)/(rho_u + rho_d)

const xlims = (0.0, 1.0)
const ylims = (0.0, 2.0)

const N = 60 #resolution
const dr = 1.0/N
const h = 2*dr

const v_char = 10.0
const l_char = 1.0
const dt = 0.1*dr/v_char
const tau_r = 0.1*l_char/v_char
const t_end =  3.0
const nframes = 100

const export_path = "results/rtiA"
const export_vars = (:v, :P, :rho)
const PROJECTION_STEPS = 10


function dividing_curve(x::Float64)::Float64
    return 1.0 - 0.15*sin(2*pi*x[1])
    #return 0.5 + 0.125*(0.7*cos(2*pi*x[1])+ 0.5*sin(2*pi*x[1]) + 0.3*cos(4*pi*x[1]) + 0.5*sin(4*pi*x[1]) + 0.1*cos(6*pi*x[1]) + 0.3*sin(6*pi*x[1]))
end


# inital condition
function ic!(p::VoronoiPolygon)
    p.rho = (p.x[2] > dividing_curve(p.x[1]) ? rho_u : rho_d)
    p.mass = p.rho*area(p)
end

function find_energy(grid::VoronoiGrid)::Float64
    E = 0.0
    for p in grid.polygons
        E += 0.5*p.mass*norm_squared(p.v) + p.mass*(1.0/Fr^2)*(p.x[2] - ylims[1])
    end
    return E
end

function gravity!(p::VoronoiPolygon)
    p.v -= dt/(Fr^2)*VECY
    if isboundary(p)
        p.v = VEC0
    end
end

function lloyd_relaxation!(p::VoronoiPolygon)
    if !isboundary(p)
        c = centroid(p)
        p.x += dt/(tau_r + dt)*(c - p.x)
    end
end

function step!(grid::VoronoiGrid, solver::PressureSolver)
    move!(grid, dt)
    #apply_unary!(grid, lloyd_relaxation!)
    #remesh!(grid)
    viscous_force!(grid, 1.0/Re, dt)
    apply_unary!(grid, gravity!)
    for _ in 1:PROJECTION_STEPS
        find_pressure!(solver, dt)
    end
    pressure_force!(grid, dt)
end

function main()
    domain = Rectangle(xlims = xlims, ylims = ylims)
    grid = VanillaGrid(domain, dr)
    populate_lloyd!(grid)
    apply_unary!(grid, ic!)
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