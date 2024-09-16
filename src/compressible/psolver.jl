struct CompressibleOperator{T}
    n::Int
    neighbors::Vector{Vector{Int}}
    lr_ratios::Vector{Vector{Float64}}
    diagonal::Vector{Float64}
    CompressibleOperator(grid::VoronoiGrid{T}) where T <: VoronoiPolygon = begin
        n = length(grid.polygons)
        neighbors = [PreAllocVector(Int, POLYGON_SIZEHINT) for _ in 1:n]
        lr_ratios = [PreAllocVector(Float64, POLYGON_SIZEHINT) for _ in 1:n]
        diagonal = [1.0 for _ in 1:n]
        return new{T}(n, neighbors, lr_ratios, diagonal)
    end
end

function refresh!(A::CompressibleOperator, grid::VoronoiGrid, dt::Float64)
    @batch for i in 1:A.n
        begin
            p = grid.polygons[i]
            empty!(A.neighbors[i])
            empty!(A.lr_ratios[i])
            A.diagonal[i] = p.mass/(p.rho^2*p.c2*dt^2)
            for (q,e) in neighbors(p, grid)
                pq = get_arrow(p.x, q.x, grid)
                push!(A.neighbors[i], e.label)
                push!(A.lr_ratios[i], lr_ratio(pq,e)*(0.5/p.rho + 0.5/q.rho))
            end
        end
    end
end

function mul!(y::ThreadedVec{Float64}, A::CompressibleOperator, x::ThreadedVec{Float64})::ThreadedVec{Float64}
    @batch for i in 1:A.n
        @inbounds begin
            y[i] = A.diagonal[i]*x[i]
            for (k,j) in enumerate(A.neighbors[i])
                lrr = A.lr_ratios[i][k]
                y[i] += lrr*(x[i] - x[j])
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
    GP::Vector{RealVector}
    ms::MinresSolver{Float64, Float64, ThreadedVec{Float64}}
    grid::VoronoiGrid{T}
    verbose::Bool
    CompressibleSolver(grid::VoronoiGrid{T}, dt::Float64 = 1.0; verbose = false) where T <: VoronoiPolygon = begin
        n = length(grid.polygons)
        A = CompressibleOperator(grid)
        b = ThreadedVec{Float64}(undef, n)
        P = ThreadedVec{Float64}(undef, n)
        GP = zeros(RealVector, n)
        ms = MinresSolver(A, b)
        return new{T}(A, b, P, GP, ms, grid, verbose)
    end
end

function refresh!(solver::CompressibleSolver, dt::Float64, gp_step::Bool)
    grid = solver.grid
    b = solver.b
    P = solver.P
    GP = solver.GP
    @batch for i in eachindex(grid.polygons)
        @inbounds begin
            p = grid.polygons[i]
            b[i] = p.mass*p.P/(p.rho^2*p.c2*dt^2)
            P[i] = p.P # serves as an initial guess
            GP[i] = VEC0
            for (q,e) in neighbors(p, grid)
                pq = get_arrow(p.x,q.x,grid)
                lrr = lr_ratio(pq,e)
                m = 0.5*(e.v1 + e.v2)
                mz = 0.5*get_arrow(m, p.x, grid) + 0.5*get_arrow(m, q.x, grid)
                b[i] -= (lrr/dt)*(dot(p.v - q.v, mz) - 0.5*dot(p.v + q.v, pq))
                GP[i] -= lrr*(p.P - q.P)*(m - p.x)
            end
            GP[i] /= p.mass
        end
    end
    # explicit part of the pressure laplacian
    if gp_step
        @batch for i in eachindex(grid.polygons)
            @inbounds begin
                p = grid.polygons[i]
                for (q,e) in neighbors(p, grid)
                    pq = get_arrow(p.x,q.x,grid)
                    lrr = lr_ratio(pq,e)
                    m = 0.5*(e.v1 + e.v2)
                    mz = 0.5*get_arrow(m, p.x, grid) + 0.5*get_arrow(m, q.x, grid)
                    j = e.label
                    b[i] += lrr*dot(GP[i] - GP[j], mz)
                end
            end
        end
    end
end

function find_pressure!(solver::CompressibleSolver, dt::Float64, niter::Int64 = 5)
    refresh!(solver.A, solver.grid, dt)
    for it in 1:niter
        refresh!(solver, dt, (it > 1))
        minres!(solver.ms, solver.A, solver.b, solver.P; verbose = Int(solver.verbose), atol = 1e-6, rtol = 1e-6, itmax = 1000)
        x = solution(solver.ms)
        for i in eachindex(x)
            @inbounds solver.grid.polygons[i].P = x[i]
        end
    end
end