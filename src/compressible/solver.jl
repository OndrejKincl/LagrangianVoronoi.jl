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

@inbounds function mul!(y::ThreadedVec{Float64}, A::CompressibleOperator, x::ThreadedVec{Float64})::ThreadedVec{Float64}
    @batch for i in 1:A.n
        y[i] = 0.0
        p = A.grid.polygons[i]
        k = 0
        lrrs = A.lrr[i]
        for (q,e) in neighbors(p, A.grid)    
            lrr = lrrs[k += 1]
            j = e.label
            y[i] += lrr*(x[i] - x[j])/(0.5*(p.rho + q.rho))
        end
        y[i] = p.area*x[i]/(A.dt^2*p.rho*p.c2) + y[i]
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
    ms::CgSolver{Float64, Float64, ThreadedVec{Float64}}
    grid::GridNSc
    verbose::Bool
    CompressibleSolver(grid::GridNSc, dt::Float64; verbose = false) = begin
        n = length(grid.polygons)
        A = CompressibleOperator(grid, dt)
        b = ThreadedVec{Float64}(undef, n)
        P = ThreadedVec{Float64}(undef, n)
        ms = CgSolver(A, b)
        return new(A, b, P, ms, grid, verbose)
    end
end

@inbounds function refresh!(solver::CompressibleSolver)
    A = solver.A
    polygons = solver.grid.polygons
    @batch for i in 1:A.n
        p = polygons[i]
        solver.P[i] = p.P
        empty!(A.lrr[i])
        solver.b[i] = 0.0
        for (q,e) in neighbors(p, solver.grid)
            lrr = lr_ratio(p,q,e)
            push!(A.lrr[i], lrr)
            m = 0.5*(e.v1 + e.v2)
            z = 0.5*(p.x + q.x)
            solver.b[i] += lrr*dot(m - z, p.v - q.v)
            solver.b[i] -= lrr*dot(p.x - q.x, 0.5*(p.v + q.v))
        end
        solver.b[i] = p.area*p.P/(A.dt^2*p.rho*p.c2) - solver.b[i]/(A.dt)
    end
end


function find_pressure!(solver::CompressibleSolver)
    refresh!(solver)
    cg!(solver.ms, solver.A, solver.b, solver.P; verbose = Int(solver.verbose), atol = 1e-8, rtol = 1e-8, itmax = 1000)
    x = solution(solver.ms)
    if !solver.ms.stats.solved
        throw("solver did not converge")
    end
    polygons = solver.grid.polygons
    @batch for i in eachindex(x)
        @inbounds polygons[i].P = x[i]
    end
end