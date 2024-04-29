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
const Res = (400, 1000, Inf)
const Ns = [16, 24, 36, 54, 81, 121]

const l_char = 1.0
const v_char = 1.0

const export_path = "results/tagr"

function v_max(Re::Float64, t::Float64)::Float64
    return exp(-2.0*pi^2*t/Re)
end

function ic!(p::VoronoiPolygon, Re::Float64)
    p.rho = rho0
    p.v = v_exact(p.x, Re, 0.0)
    p.mass = p.rho*area(p)
end

function v_exact(x::RealVector, Re::Float64, t::Float64)::RealVector
    u0 =  cos(pi*x[1])*sin(pi*x[2])
    v0 = -sin(pi*x[1])*cos(pi*x[2])
    return v_max(Re, t)*(u0*VECX + v0*VECY)
end

function P_exact(x::RealVector, Re::Float64, t::Float64)::Float64
    return 0.5*v_max(Re, t)*(sin(pi*x[1])^2 + sin(pi*x[2])^2 - 1.0)
end

function boundary_filter(x::RealVector)::Bool
    return max(abs(x[1]), abs(x[2])) > 0.4
end

mutable struct Simulation <: SimulationWorkspace
    dr::Float64
    dt::Float64
    Re::Float64
    grid::GridNS
    solver::PressureSolver{PolygonNS}
    v_err::Float64
    P_err::Float64
    E_err::Float64
    div::Float64
    Simulation(N::Int, Re::Number, structured::Bool) = begin
        Re = Float64(Re)
        dr = 1.0/N
        dt = 0.05*dr
        domain = Rectangle(xlims = xlims, ylims = ylims)
        grid = GridNS(domain, dr)
        structured ? populate_rect!(grid) : populate_vogel!(grid)
        _ic! = (p -> ic!(p, Re))
        apply_unary!(grid, _ic!)
        solver = PressureSolver(grid)
        return new(dr, dt, Re, grid, solver, 0.0, 0.0, 0.0, 0.0)
    end
end

function step!(sim::Simulation, t::Float64)
    move!(sim.grid, sim.dt)
    viscous_force!(sim.grid, 1.0/sim.Re, sim.dt)
    find_pressure!(sim.solver, sim.dt)
    pressure_force!(sim.grid, sim.dt)
end

function postproc!(sim::Simulation, t::Float64)
    grid = sim.grid
    sim.P_err = 0.0
    sim.v_err = 0.0
    sim.E_err = -0.5*v_max(sim.Re, t)^2
    sim.div = 0.0
    for p in grid.polygons
        sim.E_err += p.mass*norm_squared(p.v)
        boundary_filter(p.x) && continue
        sim.v_err += area(p)*norm_squared(p.v - v_exact(p.x, sim.Re, t))
        sim.P_err += area(p)*(p.P - P_exact(p.x, sim.Re, t))^2
        # find the divergence of p
        div = 0.0
        for (q,e) in neighbors(p, grid)
            lrr = lr_ratio(p, q, e)
            m = 0.5*(e.v1 + e.v2)
            z = 0.5*(p.x + q.x)
            div += lrr*(dot(p.v - q.v, m - z) - 0.5*dot(p.v + q.v, p.x - q.x))
        end
        sim.div += area(p)*div^2
    end
    sim.P_err = sqrt(sim.P_err)
    sim.v_err = sqrt(sim.v_err)
    sim.div = sqrt(sim.div)
    return
end

function get_convergence(Re::Number, structured::Bool)
    @show structured
    @show Re
    println("***")
    path = joinpath(export_path, structured ? "ST$(Re)" : "UN$(Re)")
    if !ispath(path)
        mkpath(path)
    end 
    pvd = paraview_collection(joinpath(path, "cells.pvd"))
    E_errs = Float64[]
    v_errs = Float64[]
    P_errs = Float64[]
    divs = Float64[]
    for N in Ns
        @show N
        sim = Simulation(N, Re, structured)
        run!(sim, sim.dt, t_end, step!; 
            postproc! = postproc!,
            nframes = 5, 
            path = path,
            save_csv = false,
            save_points = false,
            save_grid = false
        )
        push!(E_errs, abs(sim.E_err))
        push!(v_errs, sim.v_err)
        push!(P_errs, sim.P_err)
        push!(divs, sim.div)
        pvd[N] = export_grid(sim.grid, joinpath(path, "frame$(N).vtp"), :v, :P)
    end
    vtk_save(pvd)
    csv = DataFrame(Ns = Ns, E_errs = E_errs, v_errs = v_errs, P_errs = P_errs, divs = divs)
	CSV.write(joinpath(path, "convergence.csv"), csv)
    for var in (:E_errs, :v_errs, :P_errs, :divs)
        println(var)
        b = linear_regression(Ns, getproperty(csv, var))
        println("noc = $(b[1])")
    end
    println() 
    return
end

function linear_regression(x, y)
    N = length(x)
    logx = log10.(x)
    logy = log10.(y)
    A = [logx ones(N)]
    b = A\logy
    return b
end

function main()
    for structured in (true, false), Re in Res
        get_convergence(Re, structured)
    end
    makeplots()
    return
end

function makeplots()
    rainbow = cgrad(:darkrainbow, 2*length(Res), categorical = true)
    shapes = [:circ, :hex, :square, :star4, :star5, :utriangle, :dtriangle, :pentagon, :rtriangle, :ltriangle]
    for var in (:v_errs, :P_errs, :E_errs, :divs)
        myfont = font(12)
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
        for structured in (true, false), Re in Res
            label = structured ? "ST$(Re)" : "UN$(Re)"
            path = joinpath(export_path, label, "convergence.csv")
            csv = CSV.read(path, DataFrame)
            y = getproperty(csv, var)
            x = csv.Ns
            plot!(plt,
                log10.(x), log10.(y), linestyle = :dot, 
                markershape = shapes[i], 
                label = label,
                color = rainbow[i],
            )
            xmax = max(xmax, log10(maximum(x)))
            ymax = max(ymax, log10(maximum(y)))
            ymin = min(ymin, log10(minimum(y)))
            i += 1
        end
        trix = xmax + 0.5
        triy = ymax - 0.1
        sidelen = 0.2*(ymax-ymin)
        draw_triangle!(plt, trix, triy, 1, sidelen) 
        draw_triangle!(plt, trix, triy, 2, sidelen)
        savefig(plt, joinpath(export_path, "$(var).pdf"))
    end
    return
end

function draw_triangle!(plt, x, y, order = 1, sidelen = 1.0)
    a = sidelen
    shape = [(x-a,y), (x,y), (x, y-a*order), (x-a,y)]
    plot!(plt, shape, linecolor = :black, linestyle = :dash, label = :none)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end
