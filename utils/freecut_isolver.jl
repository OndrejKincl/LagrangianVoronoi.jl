using Base.Threads
using IterativeSolvers
include("multmat.jl")
include("freecut.jl")

# requires VoronoiPolygons with the following variables:
# mass::Float64
# v::RealVector
# a::RealVector
# bc_type::Int

# the implemented bc_types are:
const NOT_BC = 0
const DIRICHLET_BC = 1 #(homogen.)
const NEUMANN_BC = 2   #(homogen.)
const FREE_PENALTY = 1.0

function lr_ratio(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)::Float64
    l = free_length(p,e)
    r = norm(p.x - q.x)
    return l/r
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
    #=
    if p.var.isfree
        de += 20.0*FREE_PENALTY/p.var.rho
    end
    =#
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

function find_pressure!(grid::VoronoiGrid; no_dirichlet::Bool = false)
    #@threads for p in grid.polygons
    #    p.var.bc_type = (p.var.isfree ? DIRICHLET_BC : NOT_BC)
    #end
    # make the system
    A, b = assemble_system(
        grid,
        poi_diagonal, poi_edge, poi_vector; 
        filter = (p::VoronoiPolygon -> (p.var.bc_type == NOT_BC)), 
        constrained_average = no_dirichlet
    )
    # solve the system
    begin
        P_vector = A\b
        #P_vector = minres(ThreadedMul(A), b)
    end
    # extract the pressure from p_vec
    @threads for p in grid.polygons
        p.var.P = 0.0
        if p.var.bc_type == NOT_BC
            p.var.P = P_vector[p.id] - P_stab
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
    p.var.a -= lr_ratio(p,q,e)/p.var.mass*(p.var.P - q.var.P)*(p.x - m)
end

function stabilizer!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64)
	p.var.v += -dt*q.var.mass*rDwendland2(h_stab,r)*(P_stab/p.var.rho^2 + P_stab/q.var.rho^2)*(p.x - q.x)
end

function no_slip!(p::VoronoiPolygon)
    if isboundary(p)
        p.var.v = VEC0
    end
end