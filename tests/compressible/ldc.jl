# Lid-driven cavity

module ldc

using LaTeXStrings, DataFrames, CSV, Plots, Measures, Polyester, LinearAlgebra


include("../../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi

const Re = 1000 # Reynolds number
const N = 100 # resolution
const dr = 1.0/N
const dt = min(0.1*dr, 0.1*Re*dr^2)
const t_end = 10*dt #0.1*Re
const export_path = "results/ldc/test"

const rho0 = 1.0
const c0 = 1000.0
const gamma = 1.4
const P0 = rho0*c0^2/gamma


# inital condition
function ic!(p::VoronoiPolygon)
    p.v = VEC0
    p.rho = rho0
    p.mass = p.rho*area(p)
    p.P = 0.0
    p.e = 0.5*norm_squared(p.v) + p.P/(p.rho*(gamma - 1.0))
    p.mu = 1.0/Re
end

function vDirichlet(x::RealVector)::RealVector
    return (x[2] > 1.0 - 0.1*dr ? VECX : VEC0)
end

function bdary_friction!(grid::VoronoiGrid, dt::Float64)
    @batch for p in grid.polygons
        tmp = 1.0
        for e in boundaries(p)
            m = 0.5*(e.v1 + e.v2)
            n = normal_vector(e)
            lrr = len(e)/abs(dot(m - p.x, n))
            f = p.mu*lrr*vDirichlet(m)/p.mass
            tmp += dt*p.mu*lrr/p.mass
            p.e += dt*dot(f, p.v)
            p.v += dt*f
        end
        p.v = p.v/tmp
    end
end

mutable struct Simulation <: SimulationWorkspace
    grid::GridNS
    mesh_quality::Float64
    energy::Float64
    solver::PressureSolver{PolygonNS}
    Simulation() = begin
        domain = UnitRectangle()
        grid = GridNS(domain, dr)
        populate_lloyd!(grid, ic! = ic!)
        return new(grid, 0.0, 0.0, PressureSolver(grid))
    end
end

function step!(sim::Simulation, t::Float64)
    move!(sim.grid, dt)
    stiffened_eos!(sim.grid, gamma, P0)
    find_pressure!(sim.solver, dt)
    pressure_step!(sim.grid, dt)
    find_D!(sim.grid, noslip = false)
    viscous_step!(sim.grid, dt; artificial_viscosity = false)
    bdary_friction!(sim.grid, dt)
    find_dv!(sim.grid, dt)
    relaxation_step!(sim.grid, dt)
    return
end


function postproc!(sim::Simulation, t::Float64) 
    sim.mesh_quality = Inf
    for p in sim.grid.polygons
        sim.energy += p.mass*p.e
        #sim.S += p.mass*(log(abs(p.P)) - fp.gamma*log(abs(p.rho)))
        sim.mesh_quality = min(sim.mesh_quality, p.quality)
    end
    println("mesh quality = $(sim.mesh_quality)")
    println("energy = $(sim.energy)")
    return
end

function main()
    sim = Simulation()
    @time run!(sim, dt, t_end, step!; 
        postproc! = postproc!,
        path = export_path,
        nframes = 500,
        vtp_vars = (:v, :P, :quality),
        csv_vars = (:energy, )
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
	CSV.write(joinpath(export_path, "vprofile_legacy.csv"), data)
	make_plot()
end

function make_plot()
	ref_x2vy = CSV.read("../reference/ldc_x2vy_abdelmigid.csv", DataFrame)
	ref_y2vx = CSV.read("../reference/ldc_y2vx_abdelmigid.csv", DataFrame)
	propertyname = Symbol("Re", Re)
	ref_vy = getproperty(ref_x2vy, propertyname)
	ref_vx = getproperty(ref_y2vx, propertyname)
	ref_x = ref_x2vy.x
	ref_y = ref_y2vx.y
	data = CSV.read(joinpath(export_path, "vprofile_legacy.csv"), DataFrame)
    myfont = font(12)
    moving_rad = 1
    v1 = movingavg(data.v1, moving_rad)
    v2 = movingavg(data.v2, moving_rad)
	p1 = plot(
		data.s, v2,
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
		data.s, v1,
        label = "v",
        color = :royalblue,
        linewidth = 2,
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
	savefig(p1, joinpath(export_path, "vprofile_legacy.pdf"))
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end