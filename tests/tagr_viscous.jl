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

const Re = Inf

const structured = false

export_path = (isinf(Re) ? "results/tagrReInf" : "results/tagrRe$(round(Int, Re))") * (structured ? "" : "u")

include("../utils/parallel_settings.jl")
#include("../utils/lloyd.jl")


@with_kw mutable struct PhysFields
    mass::Float64 = 0.0
    v::RealVector = VEC0
    a::RealVector = VEC0
    P::Float64 = 0.0
    rho::Float64 = rho0
    div::Float64 = 0.0
end
include("../utils/isolver.jl")


function PhysFields(x::RealVector)
    return PhysFields(v = v_exact(x))
end

function v_max(t::Float64)::Float64
    return exp(-2.0*pi^2*t/Re)
end

function v_exact(x::RealVector, t::Float64=0.0)::RealVector
    u0 =  cos(pi*x[1])*sin(pi*x[2])
    v0 = -sin(pi*x[1])*cos(pi*x[2])
    return v_max(t)*(u0*VECX + v0*VECY)
end

function P_exact(x::RealVector, t::Float64=0.0)::Float64
    return 0.5*v_max(t)*(sin(pi*x[1])^2 + sin(pi*x[2])^2 - 1.0)
end

function SPH_stabilizer!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64, h_stab::Float64, dt::Float64, P0::Float64)
	p.var.v += -dt*q.var.mass*rDwendland2(h_stab,r)*(2*P0/rho0^2)*(p.x - q.x)
end

export_vars = (:v, :P)

function boundary_filter(x::RealVector)::Bool
    return max(abs(x[1]), abs(x[2])) > 0.4
end

export_vars = (:v, :P)

function get_errors(grid::VoronoiGrid, t::Float64)
    P_error = 0.0
    v_error = 0.0
    E_error = -0.5*v_max(t)^2
    div = 0.0
    for p in grid.polygons
        E_error += p.var.mass*LagrangianVoronoi.norm_squared(p.var.v)
        if boundary_filter(p.x)
            continue
        end
        v_error += area(p)*LagrangianVoronoi.norm_squared(p.var.v - v_exact(p.x, t))
        P_error += area(p)*(p.var.P - P_exact(p.x, t))^2
        # find divergence of p
        p.var.div = 0.0
        for e in p.edges
            if isboundary(e)
                continue
            end
            q = grid.polygons[e.label]
            lrr = lr_ratio(p, q, e)
            m = 0.5*(e.v1 + e.v2)
            z = 0.5*(p.x + q.x)
            p.var.div += lrr*(dot(p.var.v - q.var.v, m - z) - 0.5*dot(p.var.v + q.var.v, p.x - q.x))
        end
        div += area(p)*(p.var.div)^2
    end
    return (E_error, sqrt(v_error), sqrt(P_error), sqrt(div))
end

function viscous_force!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    lrr = lr_ratio(p, q, e)
    p.var.a += (1.0/Re)*lrr*(q.var.v - p.var.v)/p.var.mass
end

function solve(N::Int, stabilize::Bool)
    dr = 1.0/N
    dt = 0.03*min(dr, dr^2*Re)
    h = 2.0*dr
    P0 = dr
    @show N

    domain = Rectangle(xlims = xlims, ylims = ylims)
    grid = VoronoiGrid{PhysFields}(h, domain)
    if structured
        populate_rect!(grid, dr)
    else
        populate_vogel!(grid, dr)
        #Random.seed!(123)
        #populate_lloyd!(grid, dr, niterations = 100)
        #populate_circ!(grid, dr)
    end
    apply_unary!(grid, get_mass!)

    nframe = 0
    E_errors = Float64[]
    v_errors = Float64[]
    P_errors = Float64[]
    divs = Float64[]
    time = Float64[]

    k_end = round(Int, t_end/dt)
    k_frame = max(1, round(Int, t_end/(nframes*dt)))

    solver = PressureSolver(grid)
    
    @time for k = 0 : k_end
        for p in grid.polygons
            if boundary_filter(p.x)
                p.var.v = v_exact(p.x, k*dt)
            end
         end
        move!(grid, dt)
        #lloyd_step!(grid, tau)
        remesh!(grid)
        #apply_local!(grid, stabilizer!, h_stab)
        apply_binary!(grid, viscous_force!)
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
                E_error, v_error, P_error, div = get_errors(grid, t)
                push!(v_errors, v_error)
                push!(P_errors, P_error)
                push!(E_errors, E_error)
                push!(divs, div)
                @show v_error
                @show P_error
                @show E_error
                @show div
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

    return (grid, E_errors[end], v_errors[end], P_errors[end], divs[end])
end


function linear_reg!(plt, x, y, label, color, light=false, shape= :hex)
    N = length(x)
    logx = log10.(x)
    logy = log10.(y)
    A = [logx ones(N)]
    b = A\logy
    logY = [b[1]*logx[i] + b[2] for i in 1:N]
    @show label
    @show b[1]
    if !light
        plot!(plt,
            logx, logy,
            markershape = shape, 
            label = label*" ($(round(b[1], sigdigits=3)))",
            color = color,
        )
        plot!(plt, 
            logx, logY, linestyle = :dot, 
            label = :none,
            color = color
        )
    else
        plot!(plt,
            logx, logy, linestyle = :dot, 
            markershape = shape, 
            label = label,#*" ($(round(b[1], sigdigits=3)))",
            color = color,
        )
    end
    return
end


function add_slope_triangle!(plt, x, y; order = 1, text_margin = 0.1, sidelen = 1.0)
    a = sidelen
    shape = [(x-a,y), (x,y), (x, y-a*order), (x-a,y)]
    plot!(plt, shape, linecolor = :black, linestyle = :dash, label = :none)
    #annotate!(plt, x-0.5*a, y+text_margin,  text("1", :black, :right, 8))
    #annotate!(plt, x+text_margin, y-0.5*a*order, text(string(order), :black, :right, 8))
end

function main()
    if !ispath(export_path)
        mkpath(export_path)
        @info "created a new path \""*export_path*"\""
    end 
    pvd = paraview_collection(export_path*"/cells.pvd")
    Ns = [16, 32, 48, 72, 108]#, 162, 243]
    Ncells = []
    E_errs = []
    v_errs = []
    P_errs = []
    divs = []
    for N in Ns
        grid, E_err, v_err, P_err, div = solve(N, true)
        push!(E_errs, abs(E_err))
        push!(v_errs, v_err)
        push!(P_errs, P_err)
        push!(divs, div)
        push!(Ncells, length(grid.polygons))
        pvd[N] = export_grid(grid, string(export_path, "/frame", N, ".vtp"), :v, :P)
    end
    csv = DataFrame(Ns = Ns, E_errs = E_errs, v_errs = v_errs, P_errs = P_errs, divs = divs)
	CSV.write(string(export_path, "/convergence.csv"), csv)
    vtk_save(pvd)
    plt = plot(
        axis_ratio = 1, 
        xlabel = "log resolution", ylabel = "log error"
    )
    linear_reg!(plt, Ns, E_errs, "energy error", :orange)
    linear_reg!(plt, Ns, P_errs, "pressure error", :royalblue)
    linear_reg!(plt, Ns, v_errs, "velocity error", :purple)
    linear_reg!(plt, Ns, divs, "volume error", :darkgreen)
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

function joint_figure(Re_values = (400, 1000, Inf))
    myfont = font(12)
    for var in (:v_errs, :P_errs, :E_errs, :divs)
        plt = plot(
            axis_ratio = 1, 
            xlabel = "log resolution", ylabel = "log error",
            legend = :bottomleft, #:outerright,
            xtickfont=myfont, 
            ytickfont=myfont, 
            guidefont=myfont, 
            legendfont=myfont,
        )
        i = 1
        ymin = +Inf
        ymax = -Inf
        xmax = -Inf
        rainbow = cgrad(:darkrainbow, 2*length(Re_values), categorical = true)
        shapes = [:circ, :hex, :square, :star4, :star5, :utriangle, :dtriangle, :pentagon, :rtriangle, :ltriangle]
        for _structured in (true, false), Re in Re_values
            path = "results/tagrRe$(Re)"*(_structured ? "" : "u")*"/convergence.csv"
            csv = CSV.read(path, DataFrame)
            y = getproperty(csv, var)
            x = csv.Ns
            Re_label = @match Re begin
                400 => " 400"
                1000 => " 1000"
                Inf => " Inf"
            end
            linear_reg!(plt, x, y, (_structured ? "ST" : "UN") *Re_label, rainbow[i], true, shapes[i])

            xmax = max(xmax, log10(maximum(x)))
            ymax = max(ymax, log10(maximum(y)))
            ymin = min(ymin, log10(minimum(y)))
            i += 1
        end
        if var == :v_errs || var == :P_errs
            add_slope_triangle!(plt, xmax+0.5, ymax-0.1, order=1, sidelen=0.2*(ymax-ymin)) 
            add_slope_triangle!(plt, xmax+0.5, ymax-0.1, order=2, sidelen=0.2*(ymax-ymin))
        else
            add_slope_triangle!(plt, xmax+1.5, ymax-0.1, order=1, sidelen=0.2*(ymax-ymin)) 
            add_slope_triangle!(plt, xmax+1.5, ymax-0.1, order=2, sidelen=0.2*(ymax-ymin))        
        end
        savefig(plt, "results/tagr_$(var).pdf")
    end
end
    

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end