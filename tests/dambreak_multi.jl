module dambreak

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using Match
using WriteVTK
using LinearAlgebra, Parameters, Base.Threads
#using CSV, DataFrames
#using Plots
using StaticArrays
using SmoothedParticles: rDwendland2

const v_char = 1.0
const c_water = 50.0
const c_air = 50.0
const rho0_water = 1000.0
const rho0_air = 100.0
const dr = 2.0e-2 #resolution
const h = 2.0*dr
const mu = 1e-3

# geometrical constants
const water_column_width = 1.0
const water_column_height = 2.0
const box_height = 3.0
const box_width = 4.0
const g = -9.8*VECY        # gravitational acceleration

const r_stab = 0.2*dr
const e_stab = 0.5*v_char^2

const dt = 0.05*dr/c_water
const t_end =  1.0
const t_frame = max(dt, t_end/50)

const stabilize = true

const export_path = "results/dambreak_multi"

const box = Rectangle(
        xlims = (0.0, box_width), 
        ylims = (0.0, box_height)
    )

@with_kw mutable struct PhysFields
    mass::Float64 = 0.0
    rho::Float64 = 0.0
    v::RealVector = VEC0
    a::RealVector = VEC0
    P::Float64 = 0.0
    E::Float64 = 0.0
    isair::Bool = false
end

function PhysFields(x::RealVector)::PhysFields
    P = 0.0 #rho0*norm(g)*x[2]
    rho = 0.0 #rho0 + P/(c_water^2)
    return PhysFields(P = P, rho = rho)
end

const export_vars = (:mass, :rho, :v, :P, :E, :isair)

function getpressure!(p::VoronoiPolygon)
    p.var.rho = p.var.mass/area(p)
    if isnan(p.var.rho)
        @show length(p.edges)
        @show p.x[1]
        @show p.x[2]
        for e in p.edges
            println(e.v1[1], " ; ", e.v1[2])
            println(e.v2[1], " ; ", e.v2[2])
            println()
        end
        throw("Ooops. An invalid polygon.")
    end
    c_sound =  (p.var.isair ? c_air : c_water)
    rho0 = (p.var.isair ? rho0_air : rho0_water)
    p.var.P = c_sound^2*(p.var.rho - rho0)
    p.var.E = 0.5*p.var.mass*dot(p.var.v, p.var.v)
    p.var.E += 0.5*p.var.mass*c_sound^2*(log(abs(p.var.rho/rho0)) + rho0/p.var.rho - 1.0)
    p.var.E += p.var.mass*norm(g)*p.x[2]
    p.var.a += -surface_element(p)*p.var.P/p.var.mass
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

function viscosity!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64)
    ker = q.var.mass*rDwendland2(h,r)
    p.var.a += 2*dt*ker*mu/(q.var.rho*p.var.rho)*(p.var.v - q.var.v)
end

function move!(p::VoronoiPolygon)
    p.x += dt*p.var.v
    p.var.a = VEC0
end

function accelerate!(p::VoronoiPolygon)
    p.var.v += 0.5*dt*(p.var.a + g)
    for e in p.edges
        if e.label <= 0
            # project v into the tangent space
            n = normal_vector(e)
            dotprod = dot(p.var.v, n)
            if dotprod > 0.0
                p.var.v -= dotprod*n
            end
        end
    end
    #=
    if isboundary(p)
        p.var.v = VEC0
    end
    =#
end

function compute_force!(grid::VoronoiGrid)
    apply_unary!(grid, getpressure!)
    apply_binary!(grid, internal_force!)
    if stabilize
        apply_local!(grid, stabilizer!, r_stab)
        apply_local!(grid, viscosity!, h)
    end
end

function populate!(grid::VoronoiGrid)
    fluid = Rectangle(
        xlims = (0.0, water_column_width), 
        ylims = (0.0, water_column_height)
    )
    # hexagrid
    a = dr #((4/3)^0.25)*dr
    b = dr #((3/4)^0.25)*dr
    i_min = floor(Int, box.xmin[1]/a) - 1
    j_min = floor(Int, box.xmin[2]/b)
    i_max = ceil(Int,  box.xmax[1]/a)
    j_max = ceil(Int, box.xmax[2]/b)
	for i in i_min:i_max, j in j_min:j_max
        #x = RealVector((i + (j%2)/2)*a, j*b)
        x = RealVector((i + 0.5)*a, (j + 0.5)*b)
        if isinside(fluid, x)
            poly = VoronoiPolygon{PhysFields}(x)
            P = rho0_air*norm(g)*(box_height - water_column_height) + rho0_water*(water_column_height - x[2])
            poly.var.rho = rho0_water + P/c_water^2
            push!(grid.polygons, poly)
        elseif isinside(box, x)
            poly = VoronoiPolygon{PhysFields}(x)
            poly.var.isair = true
            P = rho0_air*norm(g)*(box_height - x[2])
            poly.var.rho = rho0_air + P/c_air^2
            push!(grid.polygons, poly)
        end 
	end
    @show length(grid.polygons)
    remesh!(grid)
    apply_unary!(grid, get_mass!)
	return
end

function get_mass!(p::VoronoiPolygon)
    p.var.mass = p.var.rho*area(p)
end

function remove_out_of_bounds!(grid::VoronoiGrid)
    to_delete = Set{Int}()
    delete_lock = ReentrantLock()
    @threads for i in eachindex(grid.polygons)
        p = grid.polygons[i]
        if !isinside(box, p.x) || isnan(p.x[1]) || isnan(p.x[2])
            lock(delete_lock)
            try
                push!(to_delete, i)
            finally
                unlock(delete_lock)
            end
        end
    end
    if !isempty(to_delete)
        @warn "I had to delete some out of bound polygons!"
        throw("error")
        @show length(to_delete)
    end
    for i in to_delete
        deleteat!(grid.polygons, i)
    end
end

function main()
    domain = Rectangle(
        xlims = (0.0, box_width),
        ylims = (0.0, box_height)
    )
    grid = VoronoiGrid{PhysFields}(h, domain)
    #limit_cell_diameter!(grid, 10*dr)
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
        remove_out_of_bounds!(grid)
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