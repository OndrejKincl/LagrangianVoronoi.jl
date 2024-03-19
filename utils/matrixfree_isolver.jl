using Base.Threads
import Base: eltype, size
import LinearAlgebra: mul!
using Base.Threads
import Base: *
using IterativeSolvers

include("multmat.jl")

function lr_ratio(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)::Float64
    l2 = norm_squared(e.v1 - e.v2)
    r2 = norm_squared(p.x - q.x)
    return sqrt(l2/r2)
end

struct VoronoiOperator{T}
    grid::VoronoiGrid{T}
    n::Int
    #edge_fun::Function
end

function mul!(y::ThreadedVec{Float64}, A::VoronoiOperator, x::ThreadedVec{Float64})::ThreadedVec{Float64}
    @threads for i in eachindex(y)
        @inbounds p = A.grid.polygons[i]
        @inbounds y[i] = 0.0
        for e in p.edges
            if isboundary(e)
                continue
            end
            j = e.label
            #@inbounds q = A.grid.polygons[j]
            @inbounds y[i] += e.lr_ratio*(x[i] - x[j])
        end
    end
    return y
end

*(A::VoronoiOperator, y::AbstractVector) = mul!(similar(y), A, y)
eltype(::VoronoiOperator) = Float64
size(A::VoronoiOperator) = (A.n, A.n)
size(A::VoronoiOperator, i::Int) = (i <= 2 ? A.n : 0)

function rhs_fun(grid::VoronoiGrid, p::VoronoiPolygon, dt::Float64)::Float64
    div = 0.0
    for e in p.edges
        if isboundary(e) 
            continue
        end
        @inbounds q = grid.polygons[e.label]
        m = 0.5*(e.v1 + e.v2)
        #div += lr_ratio(p,q,e)*dot(p.var.v - q.var.v, m - q.x)
        div += e.lr_ratio*dot(p.var.v - q.var.v, m - q.x)
    end
    return -div*(p.var.rho/dt)
end

struct PressureSolver{T}
    grid::VoronoiGrid{T}
    n::Int
    A::VoronoiOperator{T}
    b::ThreadedVec{Float64}
    P::ThreadedVec{Float64}
    PressureSolver{T}(grid::VoronoiGrid) where T = begin
        n = length(grid.polygons)
        b = ThreadedVec(zeros(n))
        P = ThreadedVec(zeros(n))
        A = VoronoiOperator{T}(grid, n) #, lr_ratio)
        return new{T}(grid, n, A, b, P)
    end
end

function find_pressure!(solver::PressureSolver, dt::Float64)
    @threads for i in 1:solver.n
        @inbounds p = solver.grid.polygons[i]
        precompute_lr_ratio!(solver.grid, p)
        @inbounds solver.P[i] = p.var.P
        @inbounds solver.b[i] = rhs_fun(solver.grid, p, dt)
    end
    minres!(solver.P, solver.A, solver.b)
    @threads for i in 1:solver.n
        @inbounds solver.grid.polygons[i].var.P = solver.P[i]
    end
end

function precompute_lr_ratio!(grid::VoronoiGrid, p::VoronoiPolygon)
    for i in eachindex(p.edges)
        e = p.edges[i]
        if isboundary(e)
            continue
        end
        q = grid.polygons[e.label]
        p.edges[i] = Edge(e.v1, e.v2, label = e.label, lr_ratio = lr_ratio(p, q, e))
    end
end

function internal_force!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    z = 0.5*(p.x + q.x)
    p.var.a += (1.0/p.var.mass)*lr_ratio(p,q,e)*(
        (p.var.P - q.var.P)*(m - z) 
        + 0.5*(p.var.P + q.var.P)*(p.x - q.x) 
    ) 
end

function move!(p::VoronoiPolygon)
    p.x += dt*p.var.v
end

function accelerate!(p::VoronoiPolygon)
    p.var.v += dt*p.var.a
    p.var.a = VEC0
end

function no_slip!(p::VoronoiPolygon)
    if isboundary(p)
        p.var.v = VEC0
    end
end