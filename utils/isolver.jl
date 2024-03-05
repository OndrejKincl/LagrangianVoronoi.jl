using Base.Threads
using IterativeSolvers
include("multmat.jl")

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
    return -lr_ratio(p,q,e)
end

function poi_diagonal(grid::VoronoiGrid, p::VoronoiPolygon)::Float64
    de = 0.0
    for e in p.edges
        if !isboundary(e)
            q = grid.polygons[e.label]
            if q.var.bc_type != NEUMANN_BC
                de += lr_ratio(p, q, e)
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
    return -(rho/dt)*div
end

function find_pressure!(grid::VoronoiGrid; no_dirichlet::Bool = false)
    # make the system
    A, b = assemble_system(
        grid,
        poi_diagonal, poi_edge, poi_vector; 
        filter = (p::VoronoiPolygon -> (p.var.bc_type == NOT_BC)), 
        constrained_average = no_dirichlet
    )
    # solve the system
    begin
        #P_vector = A\b
        P_vector = minres(ThreadedMul(A), b)
    end
    # extract the pressure from p_vec
    @threads for p in grid.polygons
        p.var.P = 0.0
        if p.var.bc_type == NOT_BC
            p.var.P = P_vector[p.id]
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

function find_energy(grid::VoronoiGrid)::Float64
    E = 0.0
    for p in grid.polygons
        E += p.var.mass*LagrangianVoronoi.norm_squared(p.var.v)
    end
    return E
end

function pressure_force!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    z = 0.5*(p.x + q.x)
    qP = q.var.P
    if q.var.bc_type == NEUMANN_BC
        qP = p.var.P
    end
    p.var.a += -(1.0/p.var.mass)*lr_ratio(p,q,e)*(
        (q.var.P - p.var.P)*(m - z) 
        - 0.5*(p.var.P + q.var.P - 2*P_stab)*(p.x - q.x)
    ) 
end

function stabilizer!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64)
	p.var.v += -dt*q.var.mass*rDwendland2(h_stab,r)*(2*P_stab/rho^2)*(p.x - q.x)
end