module ldc

using WriteVTK, LinearAlgebra, Random, Match,  Parameters
using SmoothedParticles:rDwendland2
using LaTeXStrings, DataFrames, CSV, Plots, Measures


include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi



const Re = 3200

const rho0 = 1.0
const xlims = (0.0, 1.0)
const ylims = (0.0, 1.0)
const N = 100 #resolution
const dr = 1.0/N

const dt = min(0.1*dr, 0.1*Re*dr^2)
const tau = 0.1
const t_end = 40.0
const nframes = 100

const h = 2.0*dr

#const l_char = 1.0
#const v_char = 1.0
#const P_stab = 0.0*rho0*v_char^2
#const h_stab = 2.0*dr
#const dirichlet_eps = 0.1*dr

const export_path = "results/ldc$(Re)"

include("../utils/parallel_settings.jl")
#include("../utils/lloyd.jl")


@with_kw mutable struct PhysFields
    mass::Float64 = 0.0
    v::RealVector = VEC0
    a::RealVector = VEC0
    P::Float64 = 0.0
    rho::Float64 = rho0
    iswall::Bool = false
    islid::Bool = false
end
include("../utils/isolver.jl")


function PhysFields(x::RealVector)
    return PhysFields()
end

export_vars = (:v, :P, :iswall, :islid)

function find_energy(grid::VoronoiGrid)::Float64
    E = 0.0
    for p in grid.polygons
        E += p.var.mass*LagrangianVoronoi.norm_squared(p.var.v)
    end
    return E
end

function stabilizer!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64)
	p.var.v += -dt*q.var.mass*rDwendland2(h_stab,r)*(2*P_stab/rho0^2)*(p.x - q.x)
end

function viscous_force!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    lrr = lr_ratio(p, q, e)
    p.var.a += (1.0/Re)*lrr*(q.var.v - p.var.v)/p.var.mass
end

function wall!(p::VoronoiPolygon)
    gamma = 1.0
    if isnan(p.var.P)
        throw("Pressure is NaN.")
    end
    for e in p.edges
        if !isboundary(e)
            continue
        end
        m = 0.5*(e.v1 + e.v2)
        n = normal_vector(e)
        l = len(e)
        r = abs(dot(m - p.x, n))
        v_D = (m[2] > ylims[2] - 0.1*dr ? VECX : VEC0)
        lambda = l/(Re*r*p.var.mass)
        p.var.v += dt*lambda*v_D
        gamma += dt*lambda
    end
    p.var.v = p.var.v/gamma
end

function main()
    domain = Rectangle(xlims = xlims, ylims = ylims)
    grid = VoronoiGrid{PhysFields}(h, domain)
    populate_vogel!(grid, dr, center = 0.5*VECX + 0.5*VECY)
    #Random.seed!(123)
    #populate_lloyd!(grid, dr, niterations = 20)
    apply_unary!(grid, get_mass!)
    for p in grid.polygons
        if isboundary(p)
            p.var.iswall = true
            if p.x[2] > ylims[2] - 0.5*dr
                p.var.islid = true
            end
        end
    end
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
            move!(grid, dt)
            #lloyd_step!(grid, tau)
            remesh!(grid)
            #apply_local!(grid, stabilizer!, h_stab)
            apply_binary!(grid, viscous_force!)
            apply_unary!(grid, wall!)
            accelerate!(grid, dt)
            find_pressure!(solver, dt)
            apply_binary!(grid, internal_force!)
            stabilize!(grid)
            accelerate!(grid, dt)
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
            println("energy = ", energy[end])
            pvd_c[t] = export_grid(grid, string(export_path, "/cframe", nframe, ".vtp"), export_vars...)
            pvd_p[t] = export_points(grid, string(export_path, "/pframe", nframe, ".vtp"), export_vars...)
            nframe += 1
        end
    end
    vtk_save(pvd_p)
    vtk_save(pvd_c)

    csv_data = DataFrame(time = time, energy = energy)
	CSV.write(string(export_path, "/error_data.csv"), csv_data)
    compute_fluxes(grid)
    make_plot()
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


function lloyd_step!(grid::VoronoiGrid, tau::Float64)
    @threads for p in grid.polygons
        c = centroid(p)
        p.x = tau/(tau + dt)*p.x + dt/(tau + dt)*c 
    end
end


#=
### Functions to extract results and create plots.
=#

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
	data = CSV.read("results/ldc$(Re)/data.csv", DataFrame)
    myfont = font(12)
	p1 = plot(
		data.s, data.v2,
		xlabel = "x, y",
		ylabel = "u, v",
		label = "u",
		linewidth = 2,
		legend = :topleft,
		color = :orange,
        xtickfont=myfont, 
        ytickfont=myfont, 
        guidefont=myfont, 
        legendfont=myfont,
	)
	scatter!(p1, ref_x, ref_vy, label = false, color = :orange, markersize = 4, markerstroke = stroke(1, :black), markershape = :circ)
	plot!(p1,
		data.s, data.v1,
        label = "v",
		linewidth = 2,
		color = :royalblue,
        xtickfont=myfont, 
        ytickfont=myfont, 
        guidefont=myfont, 
        legendfont=myfont,
	)
	scatter!(p1, ref_y, ref_vx, label = false, color = :royalblue, markersize = 4, markerstroke = stroke(1, :black), markershape = :square)
	savefig(p1, "results/ldc$(Re).pdf")
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end