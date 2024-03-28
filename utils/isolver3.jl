using Base.Threads
using SparseArrays

function lr_ratio(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)::Float64
    l2 = norm_squared(e.v1 - e.v2)
    r2 = norm_squared(p.x - q.x)
    return sqrt(l2/r2)
end

struct PressureSolver{T}
    grid::VoronoiGrid{T}
    verbose::Bool
    #P::Vector{Float64}
    PressureSolver(grid::VoronoiGrid{T}; verbose = false) where T = begin
        return new{T}(grid, verbose)
    end
end

function edge_element(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)::Float64
    return -lr_ratio(p,q,e)
end

function diagonal_element(grid::VoronoiGrid, p::VoronoiPolygon)::Float64
    return sum(e -> e.label == 0 ? 0.0 : lr_ratio(p, grid.polygons[e.label], e), p.edges, init = 0.0)
end

function vector_element(grid::VoronoiGrid, p::VoronoiPolygon)::Float64
    div = 0.0
    for e in p.edges
        if isboundary(e)
            continue
        end
        j = e.label
        q = grid.polygons[j]
        m = 0.5*(e.v1 + e.v2)
        z = 0.5*(p.x + q.x)
        div += lr_ratio(p,q,e)*(dot(p.var.v - q.var.v, m - z) - 0.5*dot(p.var.v + q.var.v, p.x - q.x))
    end
    return -p.var.rho*div
end


function find_pressure!(solver::PressureSolver, dt::Float64)
    A, b = assemble_system(solver.grid, diagonal_element, edge_element, vector_element, constrained_average = true)
    @threads for i in eachindex(b)
        b[i] = b[i]/dt
    end
    P = A\b
    @threads for p in solver.grid.polygons
        p.var.P = P[p.id]
    end
end

function get_mass!(p::VoronoiPolygon)
    p.var.mass = p.var.rho*area(p)
end

function move!(p::VoronoiPolygon)
    p.x += dt*p.var.v
end

function accelerate!(p::VoronoiPolygon)
    p.var.v += dt*p.var.a
    p.var.a = VEC0
end

function internal_force!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    z = 0.5*(p.x + q.x)
    p.var.a += (1.0/p.var.mass)*lr_ratio(p,q,e)*(p.var.P - q.var.P)*(m - p.x)
    #    (p.var.P - q.var.P)*(m - z) 
    #    + 0.5*(p.var.P + q.var.P)*(p.x - q.x) 
    #) 
end

function no_slip!(p::VoronoiPolygon)
    if isboundary(p)
        p.var.v = VEC0
    end
end

function tri_area(a::RealVector, b::RealVector, c::RealVector)::Float64
    return 0.5*abs(LagrangianVoronoi.cross2(b - a, c - a))
end

function centroid(p::VoronoiPolygon)::RealVector
    A = 0.0
    c = VEC0
    for e in p.edges
        dA = tri_area(p.x, e.v1, e.v2)
        A += dA
        c += dA*(p.x + e.v1 + e.v2)/3
    end
    return c/A
end


function stabilize!(grid::VoronoiGrid)
    @threads for p in grid.polygons
        LapP = 0.0 #laplacian of pressure
        for e in p.edges
            if isboundary(e)
                continue
            end
            q = grid.polygons[e.label]
            LapP -= lr_ratio(p,q,e)*(p.var.P - q.var.P)
        end
        if LapP > 0.0
            c = centroid(p)
            p.var.a += 1.5*LapP/(p.var.mass)*(c - p.x)
        end
    end
end
