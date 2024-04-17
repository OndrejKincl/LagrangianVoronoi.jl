# Lid-driven cavity

module ldc

using WriteVTK, LinearAlgebra, Random, Match,  Parameters
using SmoothedParticles:rDwendland2
using LaTeXStrings, DataFrames, CSV, Plots, Measures


include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi



const Re = 100

const rho0 = 1.0
const xlims = (0.0, 1.0)
const ylims = (0.0, 1.0)
const N = 100 #resolution
const dr = 1.0/N

const dt = min(0.1*dr, 0.1*Re*dr^2)
const t_end = 1.0
const nframes = 100

const export_path = "results/ldc$(Re)"

export_vars = (:v, :P)

# inital condition
function ic!(p::VoronoiPolygon)
    p.rho = rho0
    p.mass = p.rho*area(p)
end

function find_energy(grid::VoronoiGrid)::Float64
    E = 0.0
    for p in grid.polygons
        E += p.mass*norm_squared(p.v)
    end
    return E
end

function vDirichlet(x::RealVector)::RealVector
    return (x[2] > ylims[2] - 0.1*dr ? VECX : VEC0)
end


function main()
    domain = Rectangle(xlims = xlims, ylims = ylims)
    grid = VanillaGrid(domain, dr)
    populate_vogel!(grid, center = 0.5*VECX + 0.5*VECY)
    apply_unary!(grid, ic!)
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
        move!(grid, dt)
        viscous_force!(grid, 1.0/Re, dt, noslip = true, vDirichlet = vDirichlet) 
        find_pressure!(solver, dt)
        pressure_force!(grid, dt)
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
        v1[i] = point_value(grid, x, p -> p.v[1])
		#y-velocity along x-centerline
		x = RealVector(s[i], 0.5)
        v2[i] = point_value(grid, x, p -> p.v[2])
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