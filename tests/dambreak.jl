module dambreak

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using Match
using WriteVTK
using LinearAlgebra, Parameters
#using CSV, DataFrames
#using Plots
using StaticArrays

const v_char = 1.0
const c_sound = 50.0
const rho0 = 1000.0
const tau0 = 1.0/rho0
const dr = 1.5e-2 #resolution
const h = 2.0*dr

# geometrical constants
const water_column_width = 1.0
const water_column_height = 2.0
const box_height = 3.0
const box_width = 4.0
const g = -9.8*VECY        # gravitational acceleration

const r_stab = 0.2*dr
const e_stab = 0.5*v_char^2

const dt = 0.1*dr/c_sound
const t_end =  10*dt #4.0
const t_frame = max(dt, t_end/50)

const free_surface_rmax = 2.0*dr
const free_surface_rmin = 1.0*dr

const P0 = 0.0
const P1 = 0.005*c_sound^2*rho0

const stabilize = true

const export_path = "results/dambreak"

@with_kw mutable struct PhysFields
    mass::Float64 = rho0*dr*dr
    rho::Float64 = rho0
    v::RealVector = VEC0
    a::RealVector = VEC0
    P::Float64 = 0.0
    E::Float64 = 0.0
    iswall::Bool = false
    isfree::Bool = false
end

function PhysFields(x::RealVector)::PhysFields
    P = 0.0 #rho0*norm(g)*x[2]
    rho = rho0 + P/(c_sound^2)
    return PhysFields(P = P, rho = rho)
end

function free_surface_cutoff(d::Float64)
    if (d < free_surface_rmin) 
        return 1.0
    end
    if (d > free_surface_rmax)
        return 0.0
    end
    s = (d-free_surface_rmin)/(free_surface_rmax-free_surface_rmin)
    return (1.0 - s)^2*(1.0 + 2*s)
end

function normalize_pressure(P::Float64)::Float64
    if (P > P1) 
        return P
    elseif (P < P0)
        return 0.0
    end
    s = (P - P0)/(P1 - P0)
    return (1.0 - s)^2*(1.0 + 2*s)
end

const export_vars = (:mass, :rho, :v, :P, :E, :iswall, :isfree)

function getpressure!(p::VoronoiPolygon)
    p.var.rho = p.var.mass/area(p)
    p.var.P = c_sound^2*(p.var.rho - rho0)

    # identify free surface
    p.var.isfree = false
    begin
        diameter = 0.0
        for e in p.edges
            diameter = max(diameter, norm(e.v1 - p.x))
        end
        if diameter > free_surface_rmax
            p.var.isfree = true
        end
        #p.var.P = normalize_pressure(p.var.P)
        p.var.P = p.var.P*free_surface_cutoff(diameter)
    end

    p.var.E = 0.5*p.var.mass*dot(p.var.v, p.var.v)
    p.var.E += 0.5*p.var.mass*c_sound^2*(log(abs(p.var.rho/rho0)) + rho0/p.var.rho - 1.0)
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
end

function stabilizer!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64)
    phi = (r/r_stab - 1.0)^2
    dphi = (2.0/r_stab)*(r/r_stab - 1.0)
    m_pq = 0.5*(p.var.mass + q.var.mass)
    f_pq = (2.0/r)*e_stab*m_pq*abs(dphi)*(p.x - q.x)
    p.var.a += (1.0/p.var.mass)*f_pq
    p.var.E += m_pq*e_stab*phi
end

function move!(p::VoronoiPolygon)
    if p.var.iswall
        p.var.v = VEC0
    else
        p.x += dt*p.var.v
    end
    p.var.a = VEC0
end

function accelerate!(p::VoronoiPolygon)
    p.var.v += 0.5*dt*(p.var.a + g)
end

function compute_force!(grid::VoronoiGrid)
    apply_unary!(grid, getpressure!)
    apply_binary!(grid, internal_force!)
    if stabilize
        apply_local!(grid, stabilizer!, r_stab)
    end
end

function populate!(grid::VoronoiGrid)
    fluid_domain = Rectangle(
        xlims = (0.0, water_column_width), 
        ylims = (0.0, water_column_height)
    )
    box_interior = Rectangle(
        xlims = (0.0, box_width), 
        ylims = (0.0, box_height)
    )
    box_exterior = Rectangle(
        xlims = (-2*h, box_width + 2*h),
        ylims = (-2*h, box_height)
    )
    a = ((4/3)^0.25)*dr
    b = ((3/4)^0.25)*dr
    i_min = floor(Int, box_exterior.xmin[1]/a) - 1
    j_min = floor(Int, box_exterior.xmin[2]/b)
    i_max = ceil(Int,  box_exterior.xmax[1]/a)
    j_max = ceil(Int, box_exterior.xmax[2]/b)
	for i in i_min:i_max, j in j_min:j_max
        x = RealVector((i + (j%2)/2)*a, j*b)
        if isinside(fluid_domain, x)
            poly = VoronoiPolygon{PhysFields}(x)
            poly.var.iswall = false
            push!(grid.polygons, poly)
        elseif isinside(box_exterior, x) && !isinside(box_interior, x)
            poly = VoronoiPolygon{PhysFields}(x)
            poly.var.iswall = true
            push!(grid.polygons, poly)
        end 
	end
    @show length(grid.polygons)
    remesh!(grid)
	return
end

function main()
    domain = Rectangle(
        xlims = (-2*h, box_width + 2*h),
        ylims = (-2*h, box_height)
    )
    grid = VoronoiGrid{PhysFields}(h, domain)
    limit_cell_diameter!(grid, 10*dr)
    populate!(grid)
    if !ispath(export_path)
        mkpath(export_path)
        @info "created a new path \""*export_path*"\""
    end
    pvd_p = paraview_collection(export_path*"/points.pvd")
    pvd_c = paraview_collection(export_path*"/cells.pvd")
    nframe = 0
    energy = Float64[]
    compute_force!(grid)
    # verlet time stepping
    for k = 0 : round(Int, t_end/dt)
        apply_unary!(grid, accelerate!)
        apply_unary!(grid, move!)
        remesh!(grid)
        compute_force!(grid)
        if (k %  round(Int, t_frame/dt) == 0)
            t = k*dt
            @show t
            E = sum(p -> p.var.E, grid.polygons)
            push!(energy, E)
            @show E
            println("E_err = ", E/energy[1] - 1.0)
            pvd_c[t] = export_grid(grid, string(export_path, "/cframe", nframe, ".vtp"), export_vars...)
            pvd_p[t] = export_points(grid, string(export_path, "/pframe", nframe, ".vtp"), export_vars...)
            nframe += 1
        end
        apply_unary!(grid, accelerate!)
    end
    #p = plot(energy)
    #savefig(p, "results/gresho/energy.pdf")
    vtk_save(pvd_p)
    vtk_save(pvd_c)
end





end