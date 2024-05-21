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
        p = A.grid.polygons[i]
        A.grad[i] = VEC0
        k = 0
        lrrs = A.lrr[i]
        for (q,e) in neighbors(p, A.grid) 
            lrr = lrrs[k += 1]
            j = e.label
            m = 0.5*(e.v1 + e.v2)
            A.grad[i] -= lrr*(x[i] - x[j])*(m - p.x)
        end
        A.grad[i] = A.grad[i]/p.mass
    end
    @batch for i in 1:A.n
        y[i] = 0.0
        p = A.grid.polygons[i]
        k = 0
        lrrs = A.lrr[i]
        for (q,e) in neighbors(p, A.grid)    
            lrr = lrrs[k += 1]
            j = e.label
            m = 0.5*(e.v1 + e.v2)
            z = 0.5*(p.x + q.x)
            y[i] += lrr*dot(m - z, A.grad[i] - A.grad[j])
            y[i] -= lrr*dot(p.x - q.x, 0.5*(A.grad[i] + A.grad[j]))
        end
        y[i] = p.mass*x[i]/(A.dt*p.rho*p.c)^2 - y[i]
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
        solver.b[i] = p.mass*p.P/(A.dt*p.rho*p.c)^2 - solver.b[i]/(A.dt)
    end
end


function find_pressure!(solver::CompressibleSolver)
    refresh!(solver)
    cg!(solver.ms, solver.A, solver.b, solver.P; verbose = Int(solver.verbose), atol = 1e-8, rtol = 1e-8, itmax = 1000)
    x = solution(solver.ms)
    if !solver.ms.stats.solved
        @warn "solver did not converge"
    end
    polygons = solver.grid.polygons
    @batch for i in eachindex(x)
        @inbounds polygons[i].P = x[i]
    end
end

function posdef_test(solver::CompressibleSolver, tol = 1e-6)
    A = solver.A
    x = ThreadedVec{Float64}(undef, A.n)
    y = ThreadedVec{Float64}(undef, A.n)
    for i in 1:A.n
        x[i] = rand()
    end
    mul!(y, A, x)
    dot = LinearAlgebra.dot(y, x)
    @show dot
    if dot < -tol
        @warn "is not posdef"
    else
        @info "is posdef"
    end
end

function symm_test(solver::CompressibleSolver, tol = 1e-6)
    A = solver.A
    x1 = ThreadedVec{Float64}(undef, A.n)
    x2 = ThreadedVec{Float64}(undef, A.n)
    y = ThreadedVec{Float64}(undef, A.n)
    for i in 1:A.n
        x1[i] = rand()
        x2[i] = rand()
    end
    mul!(y, A, x1)
    dot1 = LinearAlgebra.dot(y, x2)
    mul!(y, A, x2)
    dot2 = LinearAlgebra.dot(y, x1)
    err = dot1 - dot2
    @show err
    if err > tol
        @warn "is not symmetric"
    else
        @info "is symmetric"
    end
end