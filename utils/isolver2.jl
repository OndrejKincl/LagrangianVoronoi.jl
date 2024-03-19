using Base.Threads
#using IterativeSolvers
using Krylov
include("multmat.jl")
#include("parallel_settings.jl")

# requires VoronoiPolygons with the following variables:
# mass::Float64
# v::RealVector
# a::RealVector
# bc_type::Int

# the implemented bc_types are:
const NOT_BC = 0
const DIRICHLET_BC = 1 #(homogen.)
const NEUMANN_BC = 2   #(homogen.)

function lr_ratio(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)::Float64
    l2 = norm_squared(e.v1 - e.v2)
    r2 = norm_squared(p.x - q.x)
    return sqrt(l2/r2)
end

function poi_edge(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)::Float64
    return -(0.5/p.var.rho + 0.5/q.var.rho)*lr_ratio(p,q,e)
end

function poi_diagonal(grid::VoronoiGrid, p::VoronoiPolygon)::Float64
    de = 0.0
    for e in p.edges
        if !isboundary(e)
            q = grid.polygons[e.label]
            if q.var.bc_type != NEUMANN_BC
                de += (0.5/p.var.rho + 0.5/q.var.rho)*lr_ratio(p, q, e)
            end
        end
    end
    return de
end

function poi_vector(grid::VoronoiGrid, p::VoronoiPolygon)::Float64
    # get the divergence
    div = 0.0
    for e in p.edges
        if !isboundary(e)
            m = 0.5*(e.v1 + e.v2)
            q = grid.polygons[e.label]
            div += lr_ratio(p,q,e)*dot(m - q.x, p.var.v - q.var.v)
        end
    end
    return -div/dt
end

function pressure_solver(grid::VoronoiGrid)::MinresSolver
    A, b = assemble_system(
        grid,
        poi_diagonal, poi_edge, poi_vector; 
        filter = (p::VoronoiPolygon -> (p.var.bc_type == NOT_BC)), 
        constrained_average = false
    )
    return MinresSolver(ThreadedMul(A), ThreadedVec(b))
end

function find_pressure!(grid::VoronoiGrid, solver::MinresSolver)
    # make the system
    A, b = assemble_system(
        grid,
        poi_diagonal, poi_edge, poi_vector; 
        filter = (p::VoronoiPolygon -> (p.var.bc_type == NOT_BC)), 
        constrained_average = false
    )
    P_vector = similar(b)
    @threads for i in eachindex(P_vector)
        @inbounds P_vector[i] = grid.polygons[i].var.P
    end
    # solve the system
    begin
        #P_vector = A\b
        minres!(solver, ThreadedMul(A), ThreadedVec(b), ThreadedVec(P_vector))
        #minres!(P_vector, A, b)
        #minres!(P_vector, ThreadedMul(A), b)
        #P_vector = minres(A, b)
    end
    # extract the pressure from p_vec
    x = solution(solver)
    @threads for p in grid.polygons
        p.var.P = 0.0
        if p.var.bc_type == NOT_BC
            @inbounds p.var.P = x[p.id]
        end
    end
end

function move!(p::VoronoiPolygon)
    p.x += dt*p.var.v
end

function accelerate!(p::VoronoiPolygon)
    p.var.v += dt*p.var.a
    p.var.a = VEC0
end

function pressure_force!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    z = 0.5*(p.x + q.x)
    p.var.a += (1.0/p.var.mass)*lr_ratio(p,q,e)*(
        (p.var.P - q.var.P)*(m - z) 
        + 0.5*(p.var.P + q.var.P)*(p.x - q.x) 
    ) 
end

function stabilizer!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64)
	p.var.v += -dt*q.var.mass*rDwendland2(h_stab,r)*(P_stab/p.var.rho^2 + P_stab/q.var.rho^2)*(p.x - q.x)
end

function no_slip!(p::VoronoiPolygon)
    if isboundary(p)
        p.var.v = VEC0
    end
end