module tagr_imex

using WriteVTK, LinearAlgebra, Random, Match,  Parameters
using SmoothedParticles:rDwendland2
using LaTeXStrings, DataFrames, CSV, Plots, Measures


include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi

const rho0 = 1.0
const xlims = (-0.5, 0.5)
const ylims = (-0.5, 0.5)

const t_end = 0.2
const nframes = 5

const l_char = 1.0
const v_char = 1.0

const lambda = 1
const P_stab = 1e-2

export_path = "results/tagr_imex"

include("../utils/parallel_settings.jl")
#include("../utils/lloyd.jl")


@with_kw mutable struct PhysFields
    mass::Float64 = 0.0
    v::RealVector = VEC0
    a::RealVector = VEC0
    P::Float64 = 0.0
    rho::Float64 = rho0
    x_old::RealVector = VEC0
    v_old::RealVector = VEC0
    a_old::RealVector = VEC0
end
include("../utils/isolver.jl")


function PhysFields(x::RealVector)
    v = v_exact(x)
    return PhysFields(v = v)
end

function v_exact(x::RealVector)::RealVector
    u =  cos(lambda*pi*x[1])*sin(lambda*pi*x[2])
    v = -sin(lambda*pi*x[1])*cos(lambda*pi*x[2])
    return u*VECX + v*VECY
end

function a_exact(x::RealVector)::RealVector
    ax = -lambda*pi*sin(lambda*pi*x[1])*cos(lambda*pi*x[1])
    ay = -lambda*pi*sin(lambda*pi*x[2])*cos(lambda*pi*x[2])
    return ax*VECX + ay*VECY
end

function P_exact(x::RealVector)::Float64
    return 0.5*(sin(lambda*pi*x[1])^2 + sin(lambda*pi*x[2])^2 - 1.0)
end

function boundary_filter(x::RealVector)::Bool
    return max(abs(x[1]), abs(x[2])) > 0.4
end

export_vars = (:v, :P)

function get_errors(grid::VoronoiGrid)
    P_error = 0.0
    v_error = 0.0
    E_error = -0.5
    for p in grid.polygons
        E_error += p.var.mass*LagrangianVoronoi.norm_squared(p.var.v)
        if boundary_filter(p.x)
            continue
        end
        v_error += area(p)*LagrangianVoronoi.norm_squared(p.var.v - v_exact(p.x))
        P_error += area(p)*(p.var.P - P_exact(p.x))^2
    end
    return (E_error, sqrt(v_error), sqrt(P_error))
end

# implicit-explit midpoint
function init!(grid::VoronoiGrid, dt::Float64, solver::PressureSolver)
    @batch for p in grid.polygons
        p.var.x_old = p.x
        p.var.v_old = p.var.v
        p.var.a_old = p.var.a
        p.x += dt*p.var.v
        p.var.a = VEC0
    end
    remesh!(grid)
    apply_unary!(grid, get_mass!)
    find_pressure!(solver, dt)
    apply_binary!(grid, internal_force!)
    stabilize!(grid)
    @batch for p in grid.polygons
        p.var.v += dt*p.var.a
    end
end

function midpoint_step!(grid::VoronoiGrid, dt::Float64, solver::PressureSolver)
    # new positions
    @batch for p in grid.polygons
        x = p.var.x_old + 2*dt*p.var.v
        p.var.x_old = p.x
        if isinside(grid.boundary_rect, x)
            p.x = x
        end
    end
    remesh!(grid)
    # intermediate velocity
    @batch for p in grid.polygons
        tmp = p.var.v
        p.var.v = p.var.v_old + dt*p.var.a_old
        p.var.v_old = tmp
        p.var.a_old = p.var.a
        p.var.a = VEC0
    end
    # the implicit step
    apply_unary!(grid, get_mass!)
    find_pressure!(solver, dt)
    apply_binary!(grid, internal_force!)
    stabilize!(grid)
    @batch for p in grid.polygons
        p.var.v += dt*p.var.a
        #p.var.v = v_exact(p.x)
    end
end

function cnab_step!(grid::VoronoiGrid, dt::Float64, solver::PressureSolver)
    # new positions
    @batch for p in grid.polygons
        x = p.x + 1.5*dt*p.var.v - 0.5*dt*p.var.v_old
        if !isinside(grid.boundary_rect, x)
            p.x = x
        end
    end
    remesh!(grid)
    # intermediate velocity
    @batch for p in grid.polygons
        p.var.v_old = p.var.v
        p.var.v += 0.5*dt*p.var.a
        p.var.a = VEC0
    end
    # the implicit step
    h_stab = grid.h
    #apply_local!(grid, (p,q,r) -> stabilizer!(p,q,r,h_stab,dt), h_stab)
    #apply_unary!(grid, no_slip!)
    find_pressure!(solver, 0.5*dt)
    apply_binary!(grid, internal_force!)
    stabilize!(grid)
    @batch for p in grid.polygons
        p.var.v += 0.5*dt*p.var.a
        #p.var.v = v_exact(p.x)
    end
    #apply_unary!(grid, no_slip!)
end

function stabilizer!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64, h_stab::Float64, dt::Float64)
	p.var.v += -dt*q.var.mass*rDwendland2(h_stab,r)*(2*P_stab/rho0^2)*(p.x - q.x)
end


function solve(N::Int)
    dr = 1.0/N
    dt = 0.1*dr
    h = 2.0*dr

    @show N

    domain = Rectangle(xlims = xlims, ylims = ylims)
    grid = VoronoiGrid{PhysFields}(h, domain)
    #populate_vogel!(grid, dr)
    Random.seed!(123)
    #populate_lloyd!(grid, dr, niterations = 100)
    #populate_circ!(grid, dr)
    populate_rect!(grid, dr)
    #apply_unary!(grid, get_mass!)

    nframe = 0
    E_errors = Float64[]
    v_errors = Float64[]
    P_errors = Float64[]
    time = Float64[]

    k_end = round(Int, t_end/dt)
    k_frame = max(1, round(Int, t_end/(nframes*dt)))

    solver = PressureSolver(grid)
    @time for k = 1 : k_end
        if k < 3
            init!(grid, dt, solver)
        else
            cnab_step!(grid, dt, solver)
        end
        if ((k_end - k) % k_frame == 0)
            t = k*dt
            @show t
            push!(time, t)
            E_error, v_error, P_error = get_errors(grid)
            push!(v_errors, v_error)
            push!(P_errors, P_error)
            push!(E_errors, E_error)
            @show v_error
            @show P_error
            @show E_error
            nframe += 1
        end
    end

    #compute_fluxes(grid)
    #make_plot()

    # export velocity profile along midline
    #=
    x_range = 0.0:(2*dr):xlims[2]
    vy_sim = Float64[]
    vy_exact = Float64[]
    for x1 in x_range
        x = RealVector(x1, 0.0)
        push!(vy_sim, point_value(grid, x, p -> p.var.v[2]))
        push!(vy_exact, v_exact(x)[2])
    end
    csv_data = DataFrame(x = x_range, vy_sim = vy_sim, vy_exact = vy_exact)
	CSV.write(string(export_path, "/midline_data.csv"), csv_data)
    plot_midline()
    =#

    return (grid, E_errors[end], v_errors[end], P_errors[end])
end


function main()
    if !ispath(export_path)
        mkpath(export_path)
        @info "created a new path \""*export_path*"\""
    end 
    pvd = paraview_collection(export_path*"/cells.pvd")
    Ns = [32, 48, 72, 108]#,  162, 243]
    E_errs = []
    v_errs = []
    P_errs = []
    for N in Ns
        grid, E_err, v_err, P_err = solve(N)
        push!(E_errs, abs(E_err))
        push!(v_errs, v_err)
        push!(P_errs, P_err)
        pvd[N] = export_grid(grid, string(export_path, "/frame", N, ".vtp"), :v, :P)
    end
    vtk_save(pvd)
    logNs = log10.(Ns)
    logP_errs = log10.(P_errs)
    logv_errs = log10.(v_errs)
    logE_errs = log10.(E_errs)
    plt = plot(
        logNs, [logE_errs logv_errs logP_errs], 
        #axis_ratio = 1, 
        xlabel = L"\log \, N", ylabel = L"\log \, \epsilon", 
        markershape = :hex, 
        label = ["energy" "velocity" "pressure"]
    )
    # linear regression
    A = [logNs ones(length(Ns))]
    b_P = A\logP_errs
    logP_errs_reg = [b_P[1]*logNs[i] + b_P[2] for i in 1:length(Ns)]
    b_v = A\logv_errs
    logv_errs_reg = [b_v[1]*logNs[i] + b_v[2] for i in 1:length(Ns)]
    b_E = A\logE_errs
    logE_errs_reg = [b_E[1]*logNs[i] + b_E[2] for i in 1:length(Ns)]

    plot!(plt, logNs, logE_errs_reg, linestyle = :dot, label = string("E slope = ", round(b_E[1], sigdigits=3)))
    plot!(plt, logNs, logv_errs_reg, linestyle = :dot, label = string("v slope = ", round(b_v[1], sigdigits=3)))
    plot!(plt, logNs, logP_errs_reg, linestyle = :dot, label = string("p slope = ", round(b_P[1], sigdigits=3)))
    savefig(plt, export_path*"/convergence.pdf")
    
    println("E slope = ", b_E[1])
    println("v slope = ", b_v[1])
    println("P slope = ", b_P[1])
    
end

function stabilize!(grid::VoronoiGrid)
    @threads for p in grid.polygons
        LapP = 0.0 #laplacian of pressure
        for e in p.edges
            if isboundary(e)
                continue
            end
            q = grid.polygons[e.label]
            LapP -= lr_ratio(p,q,e)*(p.var.P - q.var.P)
        end
        if LapP > 0.0
            c = centroid(p)
            p.var.a += 1.5*LapP/(p.var.mass)*(c - p.x)
        end
    end
end


function lloyd_step!(grid::VoronoiGrid, tau::Float64 = 0.1)
    @threads for p in grid.polygons
        c = centroid(p)
        p.x = tau/(tau + dt)*p.x + dt/(tau + dt)*c 
    end
end


#=
### Functions to extract results and create plots.
=#

#=
function compute_fluxes(grid::VoronoiGrid, res = 100)
    s = range(0.,1.,length=res)
    v1 = zeros(res)
    v2 = zeros(res)
    for i in 1:res
		#x-velocity along y-centerline
		x = RealVector(0.5, s[i])
        v1[i] = point_value(grid, x, p -> p.var.v[1])
		#y-velocity along x-centerline
		x = RealVector(s[i], 0.5)
        v2[i] = point_value(grid, x, p -> p.var.v[2])
    end
    #save results into csv
    data = DataFrame(s=s, v1=v1, v2=v2)
	CSV.write(export_path*"/data.csv", data)
	make_plot()
end

function make_plot(Re=Re)
	ref_x2vy = CSV.read("reference/ldc-x2vy.csv", DataFrame)
	ref_y2vx = CSV.read("reference/ldc-y2vx.csv", DataFrame)
	propertyname = Symbol("Re", Re)
	ref_vy = getproperty(ref_x2vy, propertyname)
	ref_vx = getproperty(ref_y2vx, propertyname)
	ref_x = ref_x2vy.x
	ref_y = ref_y2vx.y
	data = CSV.read(export_path*"/data.csv", DataFrame)
	p1 = plot(
		data.s, data.v2,
		xlabel = L"x, y",
		ylabel = L"u, v",
		label = L"\mathrm{SILVA} \; u",
		linewidth = 2,
		legend = :topleft,
		color = :orange,
	)
	scatter!(p1, ref_x, ref_vy, label = L"\mathrm{Ghia} \; u", color = :orange, markersize = 4, markerstroke = stroke(1, :black), markershape = :circ)
	#savefig(p1, export_path*"/ldc-x2vy.pdf")    
	plot!(p1,
		data.s, data.v1,
        label = L"\mathrm{SILVA} \; v",
		linewidth = 2,
		color = :lightblue,
	)
	scatter!(p1, ref_y, ref_vx, label = L"\mathrm{Ghia} \; v", color = :lightblue, markersize = 4, markerstroke = stroke(1, :black), markershape = :square)
	savefig(p1, export_path*"/ldc.pdf")
end
=#

function plot_midline()
    csv_data = CSV.read(string(export_path, "/midline_data.csv"), DataFrame)
    plt = plot(
        csv_data.x, 
        csv_data.vy_exact, 
        label = "exact solution",
        xlabel = L"x",
        ylabel = L"v_y",
        color = :blue,
        linestyle = :dash,
        bottom_margin = 5mm
    )
    plot!(
        plt,
        csv_data.x,
        csv_data.vy_sim,
        label = "simulation result",
        markershape = :hex,
        markersize = 2,
        color = :orange
    )

    savefig(plt, string(export_path, "/midline_plot.pdf"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end