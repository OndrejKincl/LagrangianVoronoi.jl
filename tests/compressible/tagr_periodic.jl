module tagr

using WriteVTK, LinearAlgebra, Random, Match,  Parameters
using SmoothedParticles:rDwendland2
using LaTeXStrings, DataFrames, CSV, Plots, Measures


include("../../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi

const rho0 = 1.0
const xlims = (0.0, 1.0)
const ylims = (0.0, 1.0)
const t_end = 0.2
const Res = [400, 1000, Inf]
const Ns = [32, 48, 72, 108, 162]
const c0 = 1000.0
const gamma = 1.4
const P0 = rho0*c0^2/gamma

const l_char = 1.0
const v_char = 1.0

const export_path = "results/tagr"

function v_max(Re::Float64, t::Float64)::Float64
    return exp(-8.0*pi^2*t/Re)
end

function ic!(p::VoronoiPolygon, Re::Float64)
    p.v = v_exact(p.x, Re, 0.0)
    p.rho = rho0
    p.mass = p.rho*area(p)
    p.P = P_exact(p.x, Re, 0.0)
    p.e = 0.5*norm_squared(p.v) + p.P/(p.rho*(gamma - 1.0))
    p.mu = 1.0/Re
end

function v_exact(x::RealVector, Re::Float64, t::Float64)::RealVector
    u0 =  cos(2pi*x[1])*sin(2pi*x[2])
    v0 = -sin(2pi*x[1])*cos(2pi*x[2])
    return v_max(Re, t)*(u0*VECX + v0*VECY)
end

function P_exact(x::RealVector, Re::Float64, t::Float64)::Float64
    return 0.5*(v_max(Re, t)^2)*(sin(2pi*x[1])^2 + sin(2pi*x[2])^2 - 1.0)
end

mutable struct Simulation <: SimulationWorkspace
    dr::Float64
    dt::Float64
    Re::Float64
    grid::GridNSc
    solver::CompressibleSolver{PolygonNSc}
    v_err::Float64
    P_err::Float64
    E_err::Float64
    div::Float64
    rx::Relaxator{PolygonNSc}
    first_it::Bool
    E0::Float64
    Simulation(N::Int, Re::Number) = begin
        Re = Float64(Re)
        dr = 1.0/N
        dt = 0.1*min(dr, dr^2*Re)
        domain = Rectangle(xlims = xlims, ylims = ylims)
        grid = GridNSc(domain, dr, xperiodic = true, yperiodic = true)
        populate_hex!(grid)
        _ic! = (p -> ic!(p, Re))
        apply_unary!(grid, _ic!)
        solver = CompressibleSolver(grid)
        rx = Relaxator(grid)
        return new(dr, dt, Re, grid, solver, 0.0, 0.0, 0.0, 0.0, rx, true, 0.0)
    end
end

function step!(sim::Simulation, t::Float64)
    move!(sim.grid, sim.dt)
    stiffened_eos!(sim.grid, gamma, P0)
    find_pressure!(sim.solver, sim.dt)
    pressure_step!(sim.grid, sim.dt)
    find_D!(sim.grid, noslip = false)
    viscous_step!(sim.grid, sim.dt; artificial_visc = false)
    relaxation_step!(sim.rx, sim.dt)
    return
end

function postproc!(sim::Simulation, t::Float64)
    @show t
    grid = sim.grid
    sim.P_err = 0.0
    sim.v_err = 0.0
    sim.E_err = 0.0
    sim.div = 0.0
    p_avg = 0.0
    for p in grid.polygons
        p_avg += area(p)*p.P
    end
    for p in grid.polygons
        sim.E_err += p.mass*p.e
        sim.v_err += area(p)*norm_squared(p.v - v_exact(p.x, sim.Re, t))
        sim.P_err += area(p)*(p.P - p_avg - P_exact(p.x, sim.Re, t))^2
        sim.div += area(p)*(dot(p.D, MAT1)^2)
    end
    if sim.first_it
        sim.E0 = sim.E_err
    end
    sim.E_err -= sim.E0
    sim.P_err = sqrt(sim.P_err)
    sim.v_err = sqrt(sim.v_err)
    sim.div = sqrt(sim.div)
    @show sim.v_err
    @show sim.P_err
    @show sim.E_err
    sim.first_it = false
    return
end

function simple_test(Re::Number, N::Int)
    sim = Simulation(N, Re)
    run!(sim, sim.dt, t_end, step!; 
        postproc! = postproc!,
        nframes = 100, 
        path = joinpath(export_path, "N$N"),
        save_csv = false,
        save_points = false,
        save_grid = true,
        vtp_vars = (:P, :v, :rho, :quality, :dv)
    )
end

function get_convergence(Re::Number)
    @show Re
    println("***")
    path = joinpath(export_path, "$(Re)")
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
        sim = Simulation(N, Re)
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
    for Re in Res
        get_convergence(Re)
    end
    makeplots()
    return
end

function makeplots()
    rainbow = cgrad(:darkrainbow, length(Res), categorical = true)
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
        for Re in Res
            label = "$(Re)"
            path = joinpath(export_path, label, "convergence.csv")
            csv = CSV.read(path, DataFrame)
            y = getproperty(csv, var)
            x = csv.Ns
            plot!(plt,
                log10.(x), log10.(y), linestyle = :dot, 
                markershape = shapes[i], 
                label = Re == Inf ? "Re = Inf" : "Re = $(Int(Re))",
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
