# Lid-driven cavity

module ldc

using LaTeXStrings, DataFrames, CSV, Plots, Measures


include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using SmoothedParticles:rDwendland2

const Re = 1000 # Reynolds number
const N = 100 # resolution
const dr = 1.0/N
const dt = min(0.1*dr, 0.1*Re*dr^2)
const t_end = 100.0
const export_path = "results/ldc/Re$(Re)N$(N)"
const tau = 0.1
const h_stab = 2.0*dr
const P_stab = 0.01

# inital condition
function ic!(p::VoronoiPolygon)
    p.rho = 1.0
    p.mass = area(p)
end

function vDirichlet(x::RealVector)::RealVector
    return (x[2] > 1.0 - 0.1*dr ? VECX : VEC0)
end

mutable struct Simulation <: SimulationWorkspace
    grid::GridNS
    E::Float64
    solver::PressureSolver{PolygonNS}
    Simulation() = begin
        domain = UnitRectangle()
        grid = GridNS(domain, dr)
        populate_lloyd!(grid, ic! = ic!)
        return new(grid, 0.0, PressureSolver(grid))
    end
end

function SPH_stabilizer!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64)
    (h_stab < p.x[1] < 1.0 - h_stab) || return
    (h_stab < p.x[2] < 1.0 - h_stab) || return
	p.v += -dt*q.mass*rDwendland2(h_stab,r)*(2*P_stab)*(p.x - q.x)
    return
end

function step!(sim::Simulation, t::Float64) 
    move!(sim.grid, dt)
    #apply_local!(sim.grid, SPH_stabilizer!, h_stab)
    apply_unary!(sim.grid, lloyd_stabilizer!)
    remesh!(sim.grid)
    viscous_force!(sim.grid, 1.0/Re, dt, noslip = true, vDirichlet = vDirichlet) 
    find_pressure!(sim.solver, dt)
    pressure_force!(sim.grid,  dt)
    return
end

function lloyd_stabilizer!(p::VoronoiPolygon)
    p.x = (tau*p.x + dt*centroid(p))/(tau + dt)
    return
end

function postproc!(sim::Simulation, t::Float64) 
    sim.E = 0.5*sum(p -> p.mass*norm_squared(p.v), sim.grid.polygons)
    println("energy = $(sim.E)")
    return
end

function main()
    sim = Simulation()
    @time run!(sim, dt, t_end, step!; 
        postproc! = postproc!,
        path = export_path,
        nframes = 500,
        vtp_vars = (:v, :P),
        csv_vars = (:E, )
    )
    compute_fluxes(sim.grid)
    return
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
	CSV.write(joinpath(export_path, "vprofile.csv"), data)
	make_plot()
end

function make_plot()
	ref_x2vy = CSV.read("reference/ldc-x2vy.csv", DataFrame)
	ref_y2vx = CSV.read("reference/ldc-y2vx.csv", DataFrame)
	propertyname = Symbol("Re", Re)
	ref_vy = getproperty(ref_x2vy, propertyname)
	ref_vx = getproperty(ref_y2vx, propertyname)
	ref_x = ref_x2vy.x
	ref_y = ref_y2vx.y
	data = CSV.read(joinpath(export_path, "vprofile.csv"), DataFrame)
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
	scatter!(p1, 
        ref_x, ref_vy, 
        label = false, 
        color = :orange, 
        markersize = 4, 
        markerstroke = stroke(1, :black), 
        markershape = :circ
    )
	plot!(p1,
		data.s, data.v1,
        label = "v",
	)
	scatter!(p1, 
        ref_y, 
        ref_vx, 
        label = false, 
        color = :royalblue, 
        markersize = 4, 
        markerstroke = stroke(1, :black), 
        markershape = :square
    )
	savefig(p1, joinpath(export_path, "vprofile.pdf"))
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end