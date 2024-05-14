import Base: eltype, size
import LinearAlgebra: mul!

struct CompressibleOperator 
    n::Int
    grid::GridNSc
    lrr::Vector{Vector{Float64}} # pre-computed lr ratios
    grad::Vector{RealVector}     # temporary storage for pressure gradient
    dt::Float64
    CompressibleOperator(grid::VoronoiGrid, dt::Float64) = begin
        n = length(grid.polygons)
        lrr = [PreAllocVector(Float64, POLYGON_SIZEHINT) for _ in 1:n]
        grad = [VEC0 for _ in 1:n]
        return new(n, grid, lrr, grad, dt)
    end
end

function mul!(y::ThreadedVec{Float64}, A::CompressibleOperator, x::ThreadedVec{Float64})::ThreadedVec{Float64}
    @batch for i in 1:A.n
        p = A.grid.polygons[i]
        grad[i] = VEC0
        k = 0
        for (q,e) in neighbors(p, A.grid) 
            j = e.label
            m = 0.5*(e.v1 + e.v2)
            grad[i] -= lrr[k += 1]*(x[i] - x[j])*(m - p.x)
        end
        grad[i] = grad[i]/p.mass
    end
    @batch for i in 1:A.n
        p = A.grid.polygons[i]
        k = 0
        for (q,e) in neighbors(p, A.grid)    
            lrr = A.lrr[k += 1]
            j = e.label
            m = 0.5*(e.v1 + e.v2)
            y[i] += lrr*dot(m - q.x, A.grad[i] - A.grad[j])
            y[i] -= lrr*dot(p.x - q.x, A.grad[i])
        end
        y[i] = x[i]/(p.rho*p.c^2) - (dt^2)*p.rho*y[i]/p.mass
    end
    return y
end

Base.:*(A::CompressibleOperator, y::AbstractVector) = mul!(similar(y), A, y)
Base.eltype(::CompressibleOperator) = Float64
Base.size(A::CompressibleOperator) = (A.n, A.n)
Base.size(A::CompressibleOperator, i::Int) = (i <= 2 ? A.n : 0)


struct CompressibleSolver
    A::CompressibleOperator
    b::ThreadedVec{Float64}
    P::ThreadedVec{Float64}
    ms::MinresSolver{Float64, Float64, ThreadedVec{Float64}}
    grid::GridNSc
    verbose::Bool
    CompressibleSolver(grid::GridNSc, dt::Float64; verbose = false) = begin
        n = length(grid.polygons)
        A = CompressibleOperator(grid, dt)
        b = ThreadedVec{Float64}(undef, n)
        P = ThreadedVec{Float64}(undef, n)
        ms = MinresSolver(A, b)
        return new(A, b, P, ms, grid, verbose, dt)
    end
end

function refresh!(solver::CompressibleSolver)
    A = solver.A
    polygons = solver.grid.polygons
    @batch for i in 1:A.n
        p = polygons[i]
        solver.P[i] = p.P
        empty!(A.lrr[i])
        for (q,e) in neighbors(p, solver.grid)
            push!(A.lrr[i], lr_ratio(p, q, e))
            m = 0.5*(e.v1 + e.v2)
            solver.b[i] += lrr*dot(m - q.x, p.v - q.v)
            solver.b[i] -= lrr*dot(p.x - q.x, p.v)
        end
        solver.b[i] = p.P/(p.rho*p.c^2) - A.dt*p.rho*solver.b[i]/p.mass
    end
end


function find_pressure!(solver::CompressibleSolver)
    refresh!(solver)
    minres!(solver.ms, solver.A, solver.b, solver.P; verbose = Int(solver.verbose), atol = 1e-6, rtol = 1e-6, itmax = 1000)
    x = solution(solver.ms)
    if !solver.ms.stats.solved
        @warn "solver did not converge"
    end
    polygons = solver.grid.polygons
    @batch for i in eachindex(x)
        @inbounds polygons[i].P = x[i]
    end
end