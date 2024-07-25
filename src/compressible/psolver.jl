mutable struct CompressibleOperator{T}
    n::Int
    grid::VoronoiGrid{T}
    Gy::Vector{RealVector}
    dt::Float64
    lrr_cache::Lrr_cache
    CompressibleOperator(grid::VoronoiGrid{T}, dt::Float64 = 1.0) where T <: VoronoiPolygon = begin
        n = length(grid.polygons)
        Gy = [VEC0 for _ in 1:n]
        return new{T}(n, grid, Gy, dt, new_lrr_cache(grid))
    end
end

function mul!(y::ThreadedVec{Float64}, A::CompressibleOperator, x::ThreadedVec{Float64})::ThreadedVec{Float64}
    # compute the pressure gradients
    grid = A.grid
    @batch for i in eachindex(grid.polygons)
        @inbounds begin
            p = grid.polygons[i]
            A.Gy[i] = VEC0
            for (k,(q,e)) in enumerate(neighbors(p, grid))
                lrr = A.lrr_cache[i][k]
                j = e.label
                m = 0.5*(e.v1 + e.v2)
                A.Gy[i] -= lrr*(m - p.x)*(x[i] - x[j])
            end
            A.Gy[i] /= p.mass
        end
    end
    # use the pressure gradient to find y
    @batch for i in eachindex(grid.polygons)
        @inbounds begin
            p = grid.polygons[i]
            y[i] = p.area*x[i]/(p.rho*p.c2*A.dt^2)
            for (k,(q,e)) in enumerate(neighbors(p, grid))
                lrr = A.lrr_cache[i][k]
                j = e.label
                m = 0.5*(e.v1 + e.v2)
                z = 0.5*(p.x + q.x)
                y[i] -= lrr*(dot(A.Gy[i] - A.Gy[j], m - z) - 0.5*dot(A.Gy[i] + A.Gy[j], p.x - q.x))
            end
        end
    end
    return y
end

Base.:*(A::CompressibleOperator, y::AbstractVector) = mul!(similar(y), A, y)
Base.eltype(::CompressibleOperator) = Float64
Base.size(A::CompressibleOperator) = (A.n, A.n)
Base.size(A::CompressibleOperator, i::Int) = (i <= 2 ? A.n : 0)

struct CompressibleSolver{T}
    A::CompressibleOperator{T}
    b::ThreadedVec{Float64}
    P::ThreadedVec{Float64}
    ms::MinresSolver{Float64, Float64, ThreadedVec{Float64}}
    grid::VoronoiGrid{T}
    verbose::Bool
    CompressibleSolver(grid::VoronoiGrid{T}, dt::Float64 = 1.0; verbose = false) where T <: VoronoiPolygon = begin
        n = length(grid.polygons)
        A = CompressibleOperator(grid, dt)
        b = ThreadedVec{Float64}(undef, n)
        P = ThreadedVec{Float64}(undef, n)
        ms = MinresSolver(A, b)
        return new{T}(A, b, P, ms, grid, verbose)
    end
end

function refresh!(solver::CompressibleSolver, dt::Float64)
    grid = solver.grid
    solver.A.dt = dt
    b = solver.b
    P = solver.P
    @batch for i in eachindex(grid.polygons)
        @inbounds begin
            p = grid.polygons[i]
            b[i] = p.area*p.P/(p.rho*p.c2*dt^2)
            P[i] = p.P # serves as an initial guess
            for (q,e) in neighbors(p, grid)
                lrr = lr_ratio(p,q,e)
                m = 0.5*(e.v1 + e.v2)
                z = 0.5*(p.x + q.x)
                b[i] -= (lrr/dt)*(dot(p.v - q.v, m - z) - 0.5*dot(p.v + q.v, p.x - q.x))
            end
        end
    end
    refresh!(solver.A.lrr_cache, grid)
end

function find_pressure!(solver::CompressibleSolver, dt::Float64)
    refresh!(solver, dt)
    minres!(solver.ms, solver.A, solver.b, solver.P; verbose = Int(solver.verbose), atol = 1e-6, rtol = 1e-6, itmax = 1000)
    x = solution(solver.ms)
    polygons = solver.grid.polygons
    @batch for i in eachindex(x)
        @inbounds polygons[i].P = x[i]
    end
end