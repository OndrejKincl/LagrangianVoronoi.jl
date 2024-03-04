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
const rho0 = 1.0
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
const t_end =  0.5 #4.0
const t_frame = max(dt, t_end/200)

const l_max = dr
const r_max = 2*dr

const mu = 1e-4

const export_path = "results/dambreak"

const domain = Rectangle(
        xlims = (0.0, box_width),
        ylims = (0.0, box_height)
    )

@with_kw mutable struct PhysFields
    mass::Float64 = rho0*dr*dr
    tau::Float64 = tau0
    v::RealVector = VEC0
    a::RealVector = VEC0
    P::Float64 = 0.0
    E::Float64 = 0.0
end

function PhysFields(x::RealVector)::PhysFields
    P = 0.0 #rho0*norm(g)*x[2]
    rho = rho0 + P/(c_sound^2)
    return PhysFields(P = P, tau = 1.0/rho)
end

const export_vars = (:mass, :tau, :v, :P)

function density_update!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    l = len(e)
    r = norm(p.x - q.x)
    if (r > r_max) || (l > l_max)
        return
    end
    m = 0.5*(e.v1 + e.v2)
    p.var.tau += dt*(l/r)*dot(m - q.x, p.var.v - q.var.v)/p.var.mass
end

function getpressure!(p::VoronoiPolygon)
    p.var.P = (c_sound/tau0)^2*(tau0 - p.var.tau)
    #p.var.a += -surface_element(p)*p.var.P/p.var.mass
    return
end

function internal_force!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    l = len(e)
    r = norm(p.x - q.x)
    if (r > r_max) || (l > l_max)
        return
    end
    m = 0.5*(e.v1 + e.v2)
    z = 0.5*(p.x + q.x)
    p.var.a += -(1.0/p.var.mass)*(l/r)*(
        (q.var.P - p.var.P)*(m - z) 
        - 0.5*(p.var.P + q.var.P)*(p.x - q.x)
    )
    if (r < r_stab)
        #phi = (r/r_stab - 1.0)^2
        dphi = (2.0/r_stab)*(r/r_stab - 1.0)
        m_pq = 0.5*(p.var.mass + q.var.mass)
        f_pq = (2.0/r)*e_stab*m_pq*abs(dphi)*(p.x - q.x)
        p.var.a += (1.0/p.var.mass)*f_pq
    end
    #viscosity
    p.var.a += (mu/p.var.mass)*(l/r)*(q.var.v - p.var.v)
    return
end

function move!(p::VoronoiPolygon)
    p.x += dt*p.var.v
    p.var.a = VEC0
end

@inline function pos(x::Float64)::Float64
    return (x > 0.0 ? x : 0.0)
end

function accelerate!(p::VoronoiPolygon)
    p.var.v += 0.5*dt*(p.var.a + g)
    # no-slip boundary 
    for e in p.edges
        if isboundary(e)
            n = normal_vector(e)
            r = abs(dot(e.v1 - p.x, n))
            if r < r_max
                p.var.v = VEC0
            end
        end
    end
end

function compute_force!(grid::VoronoiGrid)
    apply_unary!(grid, getpressure!)
    apply_binary!(grid, internal_force!)
    #apply_unary!(grid, wall_force!)
end

function find_energy!(p::VoronoiPolygon)
    p.var.E = 0.5*p.var.mass*dot(p.var.v, p.var.v) + 0.5*p.var.mass*(c_sound/tau0)^2*(tau0 - p.var.tau)^2 + p.var.mass*norm(g)*p.x[2]
end


function populate!(grid::VoronoiGrid)
    fluid_domain = Rectangle(
        xlims = (0.0, water_column_width), 
        ylims = (0.0, water_column_height)
    )
    a = ((4/3)^0.25)*dr
    b = ((3/4)^0.25)*dr
    i_min = floor(Int, domain.xmin[1]/a) - 1
    j_min = floor(Int, domain.xmin[2]/b)
    i_max = ceil(Int,  domain.xmax[1]/a)
    j_max = ceil(Int, domain.xmax[2]/b)
	for i in i_min:i_max, j in j_min:j_max
        x = RealVector((i + (j%2)/2)*a, j*b)
        if isinside(fluid_domain, x)
            poly = VoronoiPolygon{PhysFields}(x)
            push!(grid.polygons, poly)
        end
	end
    @show length(grid.polygons)
    remesh!(grid)
	return
end

function main()
    grid = VoronoiGrid{PhysFields}(h, domain)
    limit_cell_diameter!(grid, 2*r_max)
    populate!(grid)
    if !ispath(export_path)
        mkpath(export_path)
        @info "created a new path \""*export_path*"\""
    end
    pvd_p = paraview_collection(export_path*"/points.pvd")
    #pvd_c = paraview_collection(export_path*"/cells.pvd")
    nframe = 0
    energy = Float64[]
    compute_force!(grid)
    # verlet time stepping
    for k = 0 : round(Int, t_end/dt)
        apply_unary!(grid, accelerate!)
        apply_unary!(grid, move!)
        remesh!(grid)
        apply_binary!(grid, density_update!)
        compute_force!(grid)
        if (k %  round(Int, t_frame/dt) == 0)
            t = k*dt
            @show t
            apply_unary!(grid, find_energy!)
            push!(energy, sum(p -> p.var.E, grid.polygons))
            println("energy error = ", energy[end]/energy[1] - 1.0)
            #pvd_c[t] = export_grid(grid, string(export_path, "/cframe", nframe, ".vtp"), export_vars...)
            pvd_p[t] = export_points(grid, string(export_path, "/pframe", nframe, ".vtp"), export_vars...)
            nframe += 1
        end
        apply_unary!(grid, accelerate!)
    end
    #p = plot(energy)
    #savefig(p, "results/gresho/energy.pdf")
    vtk_save(pvd_p)
    #vtk_save(pvd_c)
end





end