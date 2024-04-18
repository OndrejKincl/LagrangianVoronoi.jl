using Base.Threads
using Krylov
import Base: eltype, size
import LinearAlgebra: mul!

struct LaplaceOperator # actually minus Laplace 
    n::Int
    neighbors::Vector{PreAllocVector{Int}}
    lr_ratios::Vector{PreAllocVector{Float64}}
    ones::ThreadedVec{Float64}
    LaplaceOperator(grid) = begin
        n = length(grid.polygons)
        neighbors = [PreAllocVector{Int}(POLYGON_SIZEHINT) for _ in 1:n]
        lr_ratios = [PreAllocVector{Float64}(POLYGON_SIZEHINT) for _ in 1:n]
        ones = ThreadedVec([1.0 for _ in 1:n])
        return new(n, neighbors, lr_ratios, ones)
    end
end

function mul!(y::ThreadedVec{Float64}, A::LaplaceOperator, x::ThreadedVec{Float64})::ThreadedVec{Float64}
    init = dot(A.ones, x)
    @batch for i in 1:A.n
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

struct JacobiPreconditioner # diagonal preconditioning 
    n::Int
    diag::Vector{Float64}
    JacobiPreconditioner(grid) = begin
        n = length(grid.polygons)
        diag = zeros(n)
        return new(n, diag)
    end
end

function mul!(y::ThreadedVec{Float64}, M::JacobiPreconditioner, x::ThreadedVec{Float64})::ThreadedVec{Float64}
    @batch for i in 1:M.n
        @inbounds y[i] = x[i]*M.diag[i]
    end
    return y
end

struct PressureSolver{T}
    A::LaplaceOperator
    b::ThreadedVec{Float64}
    M::JacobiPreconditioner
    P::ThreadedVec{Float64}
    ms::MinresSolver{Float64, Float64, ThreadedVec{Float64}}
    grid::VoronoiGrid{T}
    verbose::Bool
    PressureSolver(grid::VoronoiGrid{T}; verbose = false) where T = begin
        n = length(grid.polygons)
        A = LaplaceOperator(grid)
        M = JacobiPreconditioner(grid)
        b = ThreadedVec{Float64}(undef, n)
        P = ThreadedVec{Float64}(undef, n)
        ms = MinresSolver(A, b)
        return new{T}(A, b, M, P, ms, grid, verbose)
    end
end

function refresh!(solver::PressureSolver{T}, dt::Float64, constant_density::Bool) where T
    A = solver.A
    M = solver.M
    polygons = solver.grid.polygons
    @batch for i in 1:A.n
        @inbounds begin
            div = 0.0
            grad_P = VEC0
            grad_rho = VEC0
            p = polygons[i]
            solver.P[i] = p.P
            empty!(A.neighbors[i])
            empty!(A.lr_ratios[i])
            M.diag[i] = 0.0
            for e in p.edges
                if isboundary(e)
                    continue
                end
                j = e.label
                push!(A.neighbors[i], j)
                q = polygons[j]
                lrr = lr_ratio(p, q, e)
                push!(A.lr_ratios[i], lrr)
                M.diag[i] += lrr
                m = 0.5*(e.v1 + e.v2)
                z = 0.5*(p.x + q.x)
                div += lrr*(dot(p.v - q.v, m - z) - 0.5*dot(p.v + q.v, p.x - q.x))
                if !constant_density
                    z = 0.5*(p.x + q.x)
                    grad_P -= lrr*(p.P - q.P)*(m - z) 
                    grad_P -= 0.5*lrr*(p.P + q.P)*(p.x - q.x)
                    grad_rho += lrr*(p.rho - q.rho)*(m - q.x)
                end
            end
            solver.b[i] = -div*(p.rho/dt) + dot(grad_P, grad_rho)/p.rho
            M.diag[i] = 1.0/M.diag[i]
        end
    end
    # remove avg values from P an b
    #sP = dot(solver.P, A.ones)/A.n
    #sb = dot(solver.b, A.ones)/A.n
    #axpy!(-sP, A.ones, solver.P)
    #axpy!(-sb, A.ones, solver.P)
end


function find_pressure!(solver::PressureSolver, dt::Float64;
    constant_density::Bool = false)
    refresh!(solver, dt, constant_density)
    minres!(solver.ms, solver.A, solver.b, solver.P; verbose = Int(solver.verbose), atol = 1e-6, rtol = 1e-6, M = solver.M, itmax = 1000)
    x = solution(solver.ms)
    polygons = solver.grid.polygons
    @batch for i in eachindex(x)
        @inbounds polygons[i].P = x[i]
    end
end