module tagr

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



export_path = "results/tagr"

include("../utils/parallel_settings.jl")
#include("../utils/lloyd.jl")


@with_kw mutable struct PhysFields
    mass::Float64 = 0.0
    v::RealVector = VEC0
    a::RealVector = VEC0
    P::Float64 = 0.0
    rho::Float64 = rho0
end
include("../utils/isolver.jl")


function PhysFields(x::RealVector)
    return PhysFields(v = v_exact(x))
end

function v_exact(x::RealVector)::RealVector
    u =  cos(pi*x[1])*sin(pi*x[2])
    v = -sin(pi*x[1])*cos(pi*x[2])
    return u*VECX + v*VECY
end

function P_exact(x::RealVector)::Float64
    return 0.5*(sin(pi*x[1])^2 + sin(pi*x[2])^2 - 1.0)
end

function SPH_stabilizer!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64, h_stab::Float64, dt::Float64, P0::Float64)
	p.var.v += -dt*q.var.mass*rDwendland2(h_stab,r)*(2*P0/rho0^2)*(p.x - q.x)
end

export_vars = (:v, :P)

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

function solve(N::Int, stabilize::Bool)
    dr = 1.0/N
    dt = 0.2*dr
    h = 2.0*dr
    P0 = dr
    @show N

    domain = Rectangle(xlims = xlims, ylims = ylims)
    grid = VoronoiGrid{PhysFields}(h, domain)
    #populate_vogel!(grid, dr)
    populate_rect!(grid, dr)
    #Random.seed!(123)
    #populate_lloyd!(grid, dr, niterations = 20)
    #populate_circ!(grid, dr)
    apply_unary!(grid, get_mass!)

    nframe = 0
    E_errors = Float64[]
    v_errors = Float64[]
    P_errors = Float64[]
    time = Float64[]

    k_end = round(Int, t_end/dt)
    k_frame = max(1, round(Int, t_end/(nframes*dt)))

    solver = PressureSolver(grid)
    @time for k = 0 : k_end
        move!(grid, dt)
        #lloyd_step!(grid, tau)
        remesh!(grid)
        #apply_local!(grid, stabilizer!, h_stab)
        #apply_binary!(grid, viscous_force!)
        #apply_unary!(grid, wall!)
        accelerate!(grid, dt)
        find_pressure!(solver, dt)
        apply_binary!(grid, internal_force!)
        stabilize!(grid)
        accelerate!(grid, dt)
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


function linear_reg!(plt, x, y, label, color)
    N = length(x)
    logx = log.(x)
    logy = log.(y)
    A = [logx ones(N)]
    b = A\logy
    logY = [b[1]*logx[i] + b[2] for i in 1:N]
    @show label
    @show b[1]
    plot!(plt,
        logx, logy,
        markershape = :hex, 
        label = label*" (slope = $(round(b[1], sigdigits=3)))",
        color = color
    )
    plot!(plt, 
        logx, logY, linestyle = :dot, 
        label = :none,
        color = color
    )
    return
end

function main()
    if !ispath(export_path)
        mkpath(export_path)
        @info "created a new path \""*export_path*"\""
    end 
    pvd = paraview_collection(export_path*"/cells.pvd")
    Ns = [16, 32, 48, 72, 108, 162, 243]
    Ncells = []
    E_errs = []
    v_errs = []
    P_errs = []
    for N in Ns
        grid, E_err, v_err, P_err = solve(N, true)
        push!(E_errs, abs(E_err))
        push!(v_errs, v_err)
        push!(P_errs, P_err)
        push!(Ncells, length(grid.polygons))
        pvd[N] = export_grid(grid, string(export_path, "/frame", N, ".vtp"), :v, :P)
    end
    vtk_save(pvd)
    plt = plot(
        axis_ratio = 1, 
        xlabel = L"\log \, N", ylabel = L"\log \, \epsilon"
    )
    linear_reg!(plt, Ns, E_errs, "energy error", :orange)
    linear_reg!(plt, Ns, P_errs, "pressure error", :royalblue)
    linear_reg!(plt, Ns, v_errs, "velocity error", :purple)
    savefig(plt, export_path*"/convergence.pdf")
    
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