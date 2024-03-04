module gresho_RK4

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using Match
using WriteVTK
using LinearAlgebra
using CSV, DataFrames
using Plots
using StaticArrays

const v_char = 1.0
const c_sound = 20.0
const tau0 = 1.0
const xlims = (-0.8, 0.8)
const ylims = (-0.8, 0.8)
const N = 100 #resolution


const r_stab = 0.2/N
const e_stab = 0.5*v_char^2

const dt = 0.1/(c_sound*N)
const t_end = 0.5 #0.4*pi
const t_frame = max(dt, t_end/100)

const stabilize = true
const P0 = 0.0

mutable struct PhysFields
    mass::Float64
    tau::Float64
    v::RealVector
    a1::RealVector
    a2::RealVector
    a3::RealVector
    a4::RealVector
    P::Float64
    isboundary::Bool
    PhysFields(x::RealVector) = begin
        # the initial condition
        
        omega, P = @match norm(x) begin
            r, if r < 0.2 end => (5.0, 5.0 + 12.5*r^2)
            r, if r < 0.4 end => (2.0/r - 5.0, 9.0 + 12.5*r^2 - 20.0*r + 4.0*log(5.0*r))
            _ => (0.0, 3.0 + 4.0*log(2.0))
        end
        P += P0
        return new(
            0.0,
            tau0 - P*(tau0/c_sound)^2, 
            omega*RealVector(-x[2], x[1]),
            VEC0,
            VEC0,
            VEC0,
            VEC0,
            P,
            false
        ) 
    end
end

function getmass!(p::VoronoiPolygon)
    p.var.mass = area(p)/p.var.tau
    p.var.isboundary = isboundary(p)
end

function getpressure!(p::VoronoiPolygon)
    p.var.tau = area(p)/p.var.mass
    p.var.P = (c_sound/tau0)^2*(tau0 - p.var.tau)
end

function internal_force!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    l = len(e)
    r = norm(p.x - q.x)
    #dVpq = -l/r*(m - q.x)
    #dVqp = -l/r*(m - p.x)
    #p.var.v += dt/p.var.mass*(q.var.P*dVqp - p.var.P*dVpq)
    z = 0.5*(p.x + q.x)
    p.var.a4 += -(1.0/p.var.mass)*(l/r)*(
        (q.var.P - p.var.P)*(m - z) 
        - 0.5*(p.var.P + q.var.P)*(p.x - q.x)
    )
end

function stabilizer!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64)
    dphi = (2.0/r_stab)*(r/r_stab - 1.0)
    m_pq = 0.5*(p.var.mass + q.var.mass)
    f_pq = (2.0/r)*e_stab*m_pq*abs(dphi)*(p.x - q.x)
    p.var.a4 += (1.0/p.var.mass)*f_pq
end

function move1!(p::VoronoiPolygon)
    if !(p.var.isboundary)
        p.x += 0.5*dt*p.var.v
    end
    p.var.a1 = p.var.a4
    p.var.a4 = VEC0
end

function move2!(p::VoronoiPolygon)
    if !(p.var.isboundary)
        p.x += 0.25*dt*dt*p.var.a1
    end
    p.var.a2 = p.var.a4
    p.var.a4 = VEC0
end

function move3!(p::VoronoiPolygon)
    if !(p.var.isboundary)
        p.x += 0.5*dt*p.var.v + dt*dt*(0.5*p.var.a2 - 0.25*p.var.a1)
    end
    p.var.a3 = p.var.a4
    p.var.a4 = VEC0
end

function move4!(p::VoronoiPolygon)
    if !(p.var.isboundary)
        p.x += (dt*dt/6)*(p.var.a1 + p.var.a3 - 2*p.var.a2) 
        p.var.v += (dt/6)*(p.var.a1 + 2*p.var.a2 + 2*p.var.a3 + p.var.a4)
    end
    p.var.a1 = VEC0
    p.var.a2 = VEC0
    p.var.a3 = VEC0
    p.var.a4 = VEC0
end

function accelerate!(p::VoronoiPolygon)
    p.var.v += 0.5*dt*p.var.a
end

function compute_force!(grid::VoronoiGrid)
    apply_unary!(grid, getpressure!)
    apply_binary!(grid, internal_force!)
    if stabilize
        apply_local!(grid, stabilizer!, r_stab)
    end
end

function populate!(grid::VoronoiGrid, rect::Rectangle, dr::Float64)
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
    remesh!(grid)
    apply_unary!(grid, getmass!)
	return
end

function main()
    rect = Rectangle(xlims = xlims, ylims = ylims)
    grid = VoronoiGrid{PhysFields}(2.0/N, rect)
    populate!(grid, rect, 1/N)
    if !ispath("results/gresho_4K")
        mkpath("results/gresho_4K")
        @info "created a new path \"results/gresho_4K\""
    end 
    pvd_p = paraview_collection("results/gresho_4K/points.pvd")
    pvd_c = paraview_collection("results/gresho_4K/cells.pvd")
    nframe = 0
    # RK4 time stepping
    for k = 0 : round(Int, t_end/dt)
        compute_force!(grid)
        apply_unary!(grid, move1!)
        remesh!(grid)
        compute_force!(grid)
        apply_unary!(grid, move2!)
        remesh!(grid)
        compute_force!(grid)
        apply_unary!(grid, move3!)
        remesh!(grid)
        compute_force!(grid)
        apply_unary!(grid, move4!)
        remesh!(grid)
        if (k %  round(Int, t_frame/dt) == 0)
            t = k*dt
            @show t
            pvd_c[t] = export_grid(grid, string("results/gresho_4K/cframe", nframe, ".vtp"))
            pvd_p[t] = export_points(grid, string("results/gresho_4K/pframe", nframe, ".vtp"))
            nframe += 1
        end
    end
    vtk_save(pvd_p)
    vtk_save(pvd_c)
end





end