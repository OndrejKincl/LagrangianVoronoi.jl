using Base.Threads
using Krylov
include("multmat.jl")
include("../src/preallocvector.jl")
const CELL_SIZEHINT = 8

function lr_ratio(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)::Float64
    l2 = norm_squared(e.v1 - e.v2)
    r2 = norm_squared(p.x - q.x)
    return sqrt(l2/r2)
end

struct LaplaceOperator # actually minus Laplace 
    n::Int
    neighbors::Vector{PreAllocVector{Int}}
    lr_ratios::Vector{PreAllocVector{Float64}}
    LaplaceOperator(grid) = begin
        n = length(grid.polygons)
        neighbors = [PreAllocVector{Int}(CELL_SIZEHINT) for _ in 1:n]
        lr_ratios = [PreAllocVector{Float64}(CELL_SIZEHINT) for _ in 1:n]
        return new(n, neighbors, lr_ratios)
    end
end

function mul!(y::ThreadedVec{Float64}, A::LaplaceOperator, x::ThreadedVec{Float64})::ThreadedVec{Float64}
    init = 0.0
    @threads for i in 1:A.n
        @inbounds begin
            y[i] = init
            neighbors = A.neighbors[i]
            lr_ratios = A.lr_ratios[i]
            for k in 1:length(neighbors)
                j = neighbors[k]
                y[i] += lr_ratios[k]*(x[i] - x[j])
            end
        end
    end
    return y
end

Base.:*(A::LaplaceOperator, y::AbstractVector) = mul!(similar(y), A, y)
Base.eltype(::LaplaceOperator) = Float64
Base.size(A::LaplaceOperator) = (A.n, A.n)
Base.size(A::LaplaceOperator, i::Int) = (i <= 2 ? A.n : 0)

struct PressureSolver{T}
    A::LaplaceOperator
    b::ThreadedVec
    P::ThreadedVec
    ms::CgSolver
    grid::VoronoiGrid{T}
    PressureSolver(grid::VoronoiGrid{T}) where T = begin
        n = length(grid.polygons)
        A = LaplaceOperator(grid)
        b = ThreadedVec{Float64}(undef, n)
        P = ThreadedVec{Float64}(undef, n)
        ms = CgSolver(A, b)
        return new{T}(A, b, P, ms, grid)
    end
end

function refresh!(solver::PressureSolver{T}, dt::Float64) where T
    A = solver.A
    polygons = solver.grid.polygons
    @threads for i in 1:A.n
        @inbounds begin
            div = 0.0
            p = polygons[i]
            solver.P[i] = Float64(p.var.P)
            empty!(A.neighbors[i])
            empty!(A.lr_ratios[i])
            for e in p.edges
                if isboundary(e)
                    continue
                end
                j = e.label
                push!(A.neighbors[i], j)
                q = polygons[j]
                lrr = lr_ratio(p, q, e)
                push!(A.lr_ratios[i], lrr)
                m = 0.5*(e.v1 + e.v2)
                div += lrr*dot(p.var.v - q.var.v, m - q.x)
            end
            solver.b[i] = -div*(p.var.rho/dt)
        end
    end
end

function find_pressure!(solver::PressureSolver, dt::Float64)
    refresh!(solver, dt)
    cg!(solver.ms, solver.A, solver.b, solver.P)
    x = solution(solver.ms)
    polygons = solver.grid.polygons
    @threads for i in eachindex(x)
        polygons[i].var.P = x[i]
    end
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
    p.var.a += (1.0/p.var.mass)*lr_ratio(p,q,e)*(
        (p.var.P - q.var.P)*(m - z) 
        + 0.5*(p.var.P + q.var.P)*(p.x - q.x) 
    ) 
end

function no_slip!(p::VoronoiPolygon)
    if isboundary(p)
        p.var.v = VEC0
    end
end