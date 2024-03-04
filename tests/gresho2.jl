module gresho

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using Match
using WriteVTK
using LinearAlgebra
#using CSV, DataFrames
#using Plots
using StaticArrays

const v_char = 1.0
const c_sound = 20.0
const rho0 = 1.0
const xlims = (-1.0, 1.0)
const ylims = (-1.0, 1.0)
const N = 50 #resolution


const r_min = 0.2/N # elastic rebound below this separating distance

const dt = 0.5*r_min/c_sound
const t_end =  0.4*pi
const t_frame = max(dt, t_end/10)

const export_path = "results/gresho2"

mutable struct PhysFields
    mass::Float64
    rho::Float64
    v::RealVector
    a::RealVector
    P::Float64
    E::Float64
    PhysFields(x::RealVector) = begin
        # the initial condition
        
        omega, P = @match norm(x) begin
            r, if r < 0.2 end => (5.0, 5.0 + 12.5*r^2)
            r, if r < 0.4 end => (2.0/r - 5.0, 9.0 + 12.5*r^2 - 20.0*r + 4.0*log(r/0.2))
            _ => (0.0, 3.0 + 4.0*log(2.0))
        end
        return new(
            0.0,
            rho0 + P/(c_sound^2), 
            omega*RealVector(-x[2], x[1]),
            VEC0,
            P,
            0.0,
        ) 
    end
end

const export_vars = (:mass, :rho, :v, :P, :E)

function getmass!(p::VoronoiPolygon)
    p.var.mass = p.var.rho*area(p)
end

function getpressure!(p::VoronoiPolygon)
    p.var.rho = p.var.mass/area(p)
    p.var.P = (c_sound)^2*(p.var.rho - rho0)
    p.var.a += -surface_element(p)*p.var.P/p.var.mass
end

function internal_force!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    x_pq = p.x - q.x
    z = 0.5*(p.x + q.x)
    l = norm(e.v1 - e.v2)
    r = norm(x_pq)
    p.var.a += -(1.0/p.var.mass)*(l/r)*(
        (q.var.P - p.var.P)*(m - z) 
        - 0.5*(p.var.P + q.var.P)*x_pq
    )
    if r < r_min
        # elastic rebound
        n = (1.0/r)*x_pq
        p.var.a += (2.0/dt)*pos(-dot(p.var.v - q.var.v, n))*n
    end
end

@inline function pos(x::Float64)::Float64
    return (x > 0.0 ? x : 0.0)
end

function move!(p::VoronoiPolygon)
    if isboundary(p)
        p.var.v = VEC0
    else
        p.x += dt*p.var.v
    end
    p.var.a = VEC0
end

function accelerate!(p::VoronoiPolygon)
    p.var.v += 0.5*dt*p.var.a
end

function compute_force!(grid::VoronoiGrid)
    apply_unary!(grid, getpressure!)
    apply_binary!(grid, internal_force!)
end

function find_energy!(p::VoronoiPolygon)
    p.var.E = 0.5*p.var.mass*dot(p.var.v, p.var.v) + p.var.mass*c_sound^2*(log(abs(p.var.rho/rho0)) + rho0/p.var.rho - 1.0)
end

function populate!(grid::VoronoiGrid, rect::Rectangle, dr::Float64)
    #=
    a = ((4/3)^0.25)*dr
    b = ((3/4)^0.25)*dr
    i_min = floor(Int, rect.xmin[1]/a) - 1
    j_min = floor(Int, rect.xmin[2]/b)
    i_max = ceil(Int, rect.xmax[1]/a)
    j_max = ceil(Int, rect.xmax[2]/b)
	for i in i_min:i_max, j in j_min:j_max
        x = RealVector((i + (j%2)/2)*a, j*b)
        if isinside(rect, x)
            push!(grid.polygons, VoronoiPolygon{PhysFields}(x))
        end
	end
    for i in eachindex(grid.polygons)
        grid.polygons[i].var.n = i
    end
    =#
    for r in (0.5*dr):dr:sqrt(2)
        k_max = round(Int, 2.0*pi*r*N)
        for k in 1:k_max
            theta = 2.0*pi*k/k_max
            x = RealVector(r*cos(theta), r*sin(theta))
            if isinside(rect, x)
                push!(grid.polygons, VoronoiPolygon{PhysFields}(x))
            end
        end
    end
    @show length(grid.polygons)
    remesh!(grid)
    apply_unary!(grid, getmass!)
	return
end

function main()
    rect = Rectangle(xlims = xlims, ylims = ylims)
    grid = VoronoiGrid{PhysFields}(2.0/N, rect)
    populate!(grid, rect, 1/N)
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
    @time for k = 0 : round(Int, t_end/dt)
        apply_unary!(grid, accelerate!)
        apply_unary!(grid, move!)
        remesh!(grid)
        compute_force!(grid)
        if (k %  round(Int, t_frame/dt) == 0)
            t = k*dt
            @show t
            apply_unary!(grid, find_energy!)
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