module gresho

using WriteVTK, LinearAlgebra, Random, Match, DataFrames, CSV, Plots, Parameters
using SmoothedParticles:rDwendland2
using LaTeXStrings
#using Krylov
using IterativeSolvers
using LinearAlgebra


include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi





const v_char = 1.0
const rho0 = 1.0
const xlims = (-0.5, 0.5)
const ylims = (-0.5, 0.5)
const N = 200 #resolution
const dr = 1.0/N

const dt = 0.2*dr/v_char
const t_end =  10*dt #0.5

const P_stab = 0.01*rho0*v_char^2
const h = 2.0*dr
const h_stab = h

const export_path = "results/gresho"

include("../utils/populate.jl")
include("../utils/isolver4.jl")

@with_kw mutable struct PhysFields
    mass::Float64 = 0.0
    v::RealVector = VEC0
    a::RealVector = VEC0
    P::Float64 = 0.0
    rho::Float64 = rho0
    div::Float64 = 0.0
end

function PhysFields(x::RealVector)
    return PhysFields(v = v_exact(x), P = randn())
end

function v_exact(x::RealVector)::RealVector
    omega = @match norm(x) begin
        r, if r < 0.2 end => 5.0
        r, if r < 0.4 end => 2.0/r - 5.0
        _ => 0.0
    end
    return omega*RealVector(-x[2], x[1])
end

function reset!(p::VoronoiPolygon)
    p.var.a = VEC0
    p.var.P = 0.0
end

function main()
    domain = Rectangle(xlims = xlims, ylims = ylims)
    grid = VoronoiGrid{PhysFields}(h, domain)
    populate_circ!(grid, dr)
    apply_unary!(grid, get_mass!)
    k_end = round(Int, t_end/dt)
    @info "implicit parallelization"
    BLAS.set_num_threads(1)
    #=
    A, b = assemble_system(
                grid,
                poi_diagonal, poi_edge, poi_vector; 
                filter = (p::VoronoiPolygon -> (p.var.bc_type == NOT_BC)), 
                constrained_average = true
     )
    P_vector = zeros(length(b))
    =#
    solver = MinresSolver(length(grid.polygons))
    @time for k = 0 : k_end
       #cg!(P_vector, A, b)
       solve!(solver, grid, dt)
       apply_unary!(grid, reset!)
    end
    @info "explicit parallelization"
    @time for k = 0 : k_end
        apply_binary!(grid, internal_force!)
        apply_unary!(grid, reset!)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end