# Gresho Vortex Benchmark

module tagr

using WriteVTK, LinearAlgebra, Random, Match,  Parameters, Polyester
using SmoothedParticles:rDwendland2
using LaTeXStrings, DataFrames, CSV, Plots, Measures

include("../../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi

const Res = (Inf, 1000.0, 400.0)
const Ns = [16, 24, 36, 54, 81, 121]

const l_char = 1.0
const v_char = 1.0

const export_path = "results/tagr"

const rho0 = 1.0
const xlims = (-0.5, 0.5)
const ylims = (-0.5, 0.5)

const t_end = 0.2
const nframes = 5
const c0 = 1000.0  # sound speed
const CFL = 0.5

const gamma = 1.4
const P0 = rho0*c0^2/gamma

# exact solution and initial velocity
function v_max(Re::Float64, t::Float64)::Float64
    return exp(-2.0*pi^2*t/Re)
end

function v_exact(x::RealVector, Re::Float64, t::Float64)::RealVector
    u0 =  cos(pi*x[1])*sin(pi*x[2])
    v0 = -sin(pi*x[1])*cos(pi*x[2])
    return v_max(Re, t)*(u0*VECX + v0*VECY)
end

function P_exact(x::RealVector, Re::Float64, t::Float64)::Float64
    return 0.5*v_max(Re, t)*(sin(pi*x[1])^2 + sin(pi*x[2])^2 - 1.0)
end

# enforce inital condition on a VoronoiPolygon
function ic!(p::VoronoiPolygon)
    p.v = v_exact(p.x, Inf, 0.0)
    p.P = P_exact(p.x, Inf, 0.0)
    p.rho = rho0
    p.area = area(p)
    p.mass = p.rho*p.area
    p.e = p.P/((gamma - 1.0)*p.rho) + 0.5*norm_squared(p.v) + P0/(gamma - 1.0)*(1.0/p.rho - 1.0/rho0)
end

function enforce_bc!(grid::VoronoiGrid, t::Float64, Re::Float64)
    @batch for p in grid.polygons
        if isboundary(p)
            p.v = v_exact(p.x, Re, t)
            #p.P = P_exact(p.x, Re, t)
            #p.e = p.P/((gamma - 1.0)*p.rho) + 0.5*norm_squared(p.v)
        end
    end
end

function energy(grid::VoronoiGrid)::Float64
    E = 0.0
    for p in grid.polygons  
        E += p.mass*p.e
    end
    return E
end

mutable struct Simulation <: SimulationWorkspace
    grid::GridNSc
    solver::CompressibleSolver
    N::Int64
    Re::Float64
    dr::Float64
    dt::Float64

    P_err::Float64
    v_err::Float64
    E_err::Float64
    E0::Float64
    Simulation(N::Number, Re::Number, structured::Bool) = begin
        dr = 1.0/N
        dt = CFL*dr/v_char
        domain = Rectangle(xlims = xlims, ylims = ylims)
        grid = GridNSc(domain, dr)
        structured ? populate_hex!(grid) : populate_vogel!(grid)
        remesh!(grid)
        apply_unary!(grid, ic!)
        E0 = energy(grid)
        return new(grid, CompressibleSolver(grid, dt, verbose=0), N, Re, dr, dt, 0.0, 0.0, 0.0, E0)
    end
end

function step!(sim::Simulation, t::Float64)
    find_D!(sim.grid, false)
    find_mu!(sim.grid, 0.0, 1.0/sim.Re)
    viscous_step!(sim.grid, sim.dt)
    #enforce_bc!(sim.grid, sim.dt, sim.Re)
    stiffened_eos!(sim.grid, gamma, rho0, P0)
    find_pressure!(sim.solver)
    pressure_step!(sim.grid, sim.dt)
    #enforce_bc!(sim.grid, sim.dt, sim.Re)
    lloyd_step!(sim.grid, sim.dt)
    move!(sim.grid, sim.dt)
    find_rho!(sim.grid)
    return
end

# find energy and l2 error
function postproc!(sim::Simulation, t::Float64) 
    sim.P_err = 0.0
    sim.v_err = 0.0
    for p in sim.grid.polygons
        p.v_err = p.v - v_exact(p.x, sim.Re, t)
        p.P_err = p.P - P_exact(p.x, sim.Re, t)
        boundary_filter(p.x) && continue
        sim.v_err += p.area*norm_squared(p.v - v_exact(p.x, sim.Re, t))
        sim.P_err += p.area*(p.P - P_exact(p.x, sim.Re, t))^2
    end
    sim.P_err = sqrt(sim.P_err)
    sim.v_err = sqrt(sim.v_err)
    sim.E_err = abs(energy(sim.grid) - sim.E0)
    @show sim.P_err
    @show sim.v_err
    @show sim.E_err
end

function boundary_filter(x::RealVector)::Bool
    return max(abs(x[1]), abs(x[2])) > 0.4
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
    v_errs = Float64[]
    P_errs = Float64[]
    E_errs = Float64[]
    for N in Ns
        @show N
        sim = Simulation(N, Re, structured)
        run!(sim, sim.dt, t_end, step!; 
            postproc! = postproc!,
            nframes = nframes, 
            path = path,
            save_csv = false,
            save_points = false,
            save_grid = true
        )
        push!(v_errs, sim.v_err)
        push!(P_errs, sim.P_err)
        push!(E_errs, sim.E_err)
        pvd[N] = export_grid(sim.grid, joinpath(path, "frame$(N).vtp"), :v, :P, :v_err, :P_err)
    end
    vtk_save(pvd)
    csv = DataFrame(Ns = Ns, v_errs = v_errs, P_errs = P_errs, E_errs = E_errs)
	CSV.write(joinpath(path, "convergence.csv"), csv)
    for var in (:v_errs, :P_errs, :E_errs)
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
    for structured in (false, ), Re in Res
        get_convergence(Re, structured)
    end
    makeplots()
    return
end

function makeplots()
    rainbow = cgrad(:darkrainbow, 2*length(Res), categorical = true)
    shapes = [:circ, :hex, :square, :star4, :star5, :utriangle, :dtriangle, :pentagon, :rtriangle, :ltriangle]
    for var in (:v_errs, :P_errs, :E_errs)
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
        for structured in (false,), Re in Res
            label = structured ? "ST$(Re)" : "UN$(Re)"
            path = joinpath(export_path, label, "convergence.csv")
            csv = CSV.read(path, DataFrame)
            y = getproperty(csv, var)
            x = csv.Ns
            plot!(plt,
                log10.(x), log10.(y), linestyle = :dot, 
                markershape = shapes[i], 
                label = (Re == Inf ? "Re = ∞" : "Re = $(Int(Re))"),
                #color = rainbow[i],
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