#=

OBSOLETE (DOES NOT WORK)

=#

module isolver_test

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using LinearAlgebra
using WriteVTK, StaticArrays, Random
using Plots, LaTeXStrings
using SmoothedParticles:rDwendland2
using Parameters
using Base.Threads


const DOMAIN = Rectangle(xlims = (0,1), ylims = (0,1))
const dt = 1.0
const rho = 1.0
const v0 = 1.0
const export_path = "results/isolver_test"
const tau_r = 0.2

@with_kw mutable struct PhysFields
    v::RealVector = VEC0
    a::RealVector = VEC0
    P::Float64 = 0.0
    v_exact::RealVector = VEC0
    P_exact::Float64 = 0.0
    mass::Float64 = 0.0
    rho::Float64 = rho
end

include("../utils/isolver2.jl")

function PhysFields(x::RealVector)::PhysFields
    pf = PhysFields()
    pf.v = v_init(x)
    pf.v_exact = v_exact(x)
    pf.P_exact = P_exact(x)
    return pf
end

function v_init(x::RealVector)::RealVector
    vx = -x[1]*(x[1] - 1.0)*(x[2] - 0.5) + pi*v0*cos(pi*x[1])*sin(pi*x[2]) #- pi*cos(pi*x[1])*sin(pi*x[2])
    vy = (x[1] - 0.5)*x[2]*(x[2] - 1.0) + pi*v0*sin(pi*x[1])*cos(pi*x[2]) #- pi*sin(pi*x[1])*cos(pi*x[2])
    return RealVector(vx, vy)
end

function v_exact(x::RealVector)::RealVector
    vx = -x[1]*(x[1] - 1.0)*(x[2] - 0.5)
    vy = (x[1] - 0.5)*x[2]*(x[2] - 1.0)
    return RealVector(vx, vy)
end

function div_exact(x::RealVector)::Float64
    vx_x = -(2*x[1] - 1.0)*(x[2] - 0.5) - pi*v0*sin(pi*x[1])*sin(pi*x[2])
    vy_y =  (x[1] - 0.5)*(2*x[2] - 1.0) - pi*v0*sin(pi*x[1])*sin(pi*x[2])
    return vx_x + vy_y
end

function P_exact(x::RealVector)::Float64
    return v0*rho/dt*(sin(pi*x[1])*sin(pi*x[2]))
end

function filter_fun(p::VoronoiPolygon)::Bool
    return norm(p.x-RealVector(0.5, 0.5)) < 0.4
end

function force!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    p.var.v += (dt/p.var.mass)*lr_ratio(p,q,e)*(p.var.P - q.var.P)*(m - p.x)
end

function solve(dr::Float64)
    grid = VoronoiGrid{PhysFields}(2*dr, DOMAIN)
    #populate_rect!(grid, dr)
    #populate_rand!(grid, dr)
    populate_vogel!(grid, dr)
    grid.rr_max = Inf
    remesh!(grid)
    apply_unary!(grid, get_mass!)
    if !ispath(export_path)
        mkpath(export_path)
        @info "created a new path \""*export_path*"\""
    end 
    solver = PressureSolver(grid, verbose = true)
    #apply_unary!(grid, no_slip!)
    @time find_pressure!(solver, dt)
    apply_binary!(grid, force!)
    #apply_unary!(grid, accelerate!)
    #apply_unary!(grid, no_slip!)

    p_err = 0.0
    v_err = 0.0
    for p in grid.polygons
        p_err += Float64(filter_fun(p))*area(p)*abs(p.var.P - p.var.P_exact)^2
        v_err += Float64(filter_fun(p))*area(p)*norm(p.var.v - p.var.v_exact)^2
    end
    p_err = sqrt(p_err)
    v_err = sqrt(v_err)
    @show p_err
    @show v_err
    vtk_file = export_grid(grid, string(export_path, "/wft.vtp"), :v, :P, :v_exact, :P_exact, :mass)
    vtk_save(vtk_file)
    #return (grid, p_err, v_err)
end

function main()
    if !ispath(export_path)
        mkpath(export_path)
        @info "created a new path \""*export_path*"\""
    end 
    pvd = paraview_collection(export_path*"/cells.pvd")
    Ns = [32, 48, 72, 108, 162, 243]
    P_errs = []
    v_errs = []
    for N in Ns
        grid, P_err, v_err = solve(1.0/N)
        push!(P_errs, P_err)
        push!(v_errs, v_err)
        pvd[N] = export_grid(grid, string(export_path, "/frame", N, ".vtp"), :v, :P, :v_exact, :P_exact)
    end
    vtk_save(pvd)
    logNs = log10.(Ns)
    logP_errs = log10.(P_errs)
    logv_errs = log10.(v_errs)
    plt = plot(
        logNs, [logP_errs logv_errs], 
        axis_ratio = 1, 
        xlabel = L"\log \, N", ylabel = L"\log \, \epsilon", 
        markershape = :hex, 
        label = ["pressure" "velocity"]
    )
    # linear regression
    A = [logNs ones(length(Ns))]
    b_P = A\logP_errs
    logP_errs_reg = [b_P[1]*logNs[i] + b_P[2] for i in 1:length(Ns)]
    b_v = A\logv_errs
    logv_errs_reg = [b_v[1]*logNs[i] + b_v[2] for i in 1:length(Ns)]
    plot!(plt, logNs, logP_errs_reg, linestyle = :dot, label = string("p slope = ", round(b_P[1], sigdigits=3)))
    plot!(plt, logNs, logv_errs_reg, linestyle = :dot, label = string("v slope = ", round(b_v[1], sigdigits=3)))
    savefig(plt, "helmholtz_convergence.pdf")
    println("P slope = ", b_P[1])
    println("v slope = ", b_v[1])
    
end

end