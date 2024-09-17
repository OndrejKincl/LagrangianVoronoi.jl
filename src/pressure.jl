import Base: eltype, size
import LinearAlgebra: mul!

# update velocity by pressure gradient
function pressure_step!(grid::VoronoiGrid, dt::Float64)  
    @batch for p in grid.polygons
        for (q,e,y) in neighbors(p, grid)
            lrr = lr_ratio(p.x-y,e)
            m = midpoint(e)
            p.v += dt*lrr/p.mass*(p.P - q.P)*(m - p.x)
        end
    end
    @batch for p in grid.polygons
        for (q,e,y) in neighbors(p, grid)
            lrr = lr_ratio(p.x-y,e)
            m = midpoint(e)
            p.e -= dt*lrr/p.mass*(dot(m - p.x, p.P*p.v) - dot(m - y, q.P*q.v))
        end
    end
end

# internal energy
function eint(p::VoronoiPolygon)::Float64 
    return p.e - 0.5*norm_squared(p.v)
end

# ideal gas equation of state
function ideal_eos!(grid::VoronoiGrid, gamma::Float64 = 1.4; Pmin::Float64 = 0.0) 
    @batch for p in grid.polygons
        p.rho = p.mass/area(p)
        p.P = (gamma - 1.0)*p.rho*eint(p)
        p.c2 = gamma*max(p.P, Pmin)/p.rho
    end
end

# stiffened gas equation of state
function stiffened_eos!(grid::VoronoiGrid, gamma::Float64 = 1.4, P0::Float64 = 0.0)
    @batch for p in grid.polygons
        p.rho = p.mass/area(p)
        p.P = (gamma - 1.0)*p.rho*eint(p)
        p.c2 = gamma*(p.P + P0)/p.rho
    end
end

function gravity_step!(grid::VoronoiGrid, g::RealVector, dt::Float64)
    @batch for p in grid.polygons
        p.v += dt*g
        p.e += dot(dt*g, p.v)
    end
end

# Operator for the pressure system
struct PressureOperator{T}
    n::Int
    neighbors::Vector{Vector{Int}}
    lr_ratios::Vector{Vector{Float64}}
    diagonal::Vector{Float64}
    PressureOperator(grid::VoronoiGrid{T}) where T <: VoronoiPolygon = begin
        n = length(grid.polygons)
        neighbors = [PreAllocVector(Int, POLYGON_SIZEHINT) for _ in 1:n]
        lr_ratios = [PreAllocVector(Float64, POLYGON_SIZEHINT) for _ in 1:n]
        diagonal = [1.0 for _ in 1:n]
        return new{T}(n, neighbors, lr_ratios, diagonal)
    end
end

function refresh!(A::PressureOperator, grid::VoronoiGrid, dt::Float64)
    @batch for i in 1:A.n
        begin
            p = grid.polygons[i]
            empty!(A.neighbors[i])
            empty!(A.lr_ratios[i])
            A.diagonal[i] = p.mass/(p.rho^2*p.c2*dt^2)
            for (q,e,y) in neighbors(p, grid)
                push!(A.neighbors[i], e.label)
                push!(A.lr_ratios[i], lr_ratio(p.x - y, e)*(0.5/p.rho + 0.5/q.rho))
            end
        end
    end
end

function mul!(y::ThreadedVec{Float64}, A::PressureOperator, x::ThreadedVec{Float64})::ThreadedVec{Float64}
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

Base.:*(A::PressureOperator, y::AbstractVector) = mul!(similar(y), A, y)
Base.eltype(::PressureOperator) = Float64
Base.size(A::PressureOperator) = (A.n, A.n)
Base.size(A::PressureOperator, i::Int) = (i <= 2 ? A.n : 0)

struct PressureSolver{T}
    A::PressureOperator{T}
    b::ThreadedVec{Float64}
    P::ThreadedVec{Float64}
    GP::Vector{RealVector}
    ms::MinresSolver{Float64, Float64, ThreadedVec{Float64}}
    grid::VoronoiGrid{T}
    verbose::Bool
    PressureSolver(grid::VoronoiGrid{T}; verbose = false) where T <: VoronoiPolygon = begin
        n = length(grid.polygons)
        A = PressureOperator(grid)
        b = ThreadedVec{Float64}(undef, n)
        P = ThreadedVec{Float64}(undef, n)
        GP = zeros(RealVector, n)
        ms = MinresSolver(A, b)
        return new{T}(A, b, P, GP, ms, grid, verbose)
    end
end

function refresh!(solver::PressureSolver, dt::Float64, gp_step::Bool)
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
            for (q,e,y) in neighbors(p, grid)
                lrr = lr_ratio(p.x-y,e)
                m = midpoint(e)
                z = midpoint(p.x,y)
                b[i] -= (lrr/dt)*(dot(p.v - q.v, m-z) - 0.5*dot(p.v + q.v, p.x-y))
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
                for (q,e,y) in neighbors(p, grid)
                    lrr = lr_ratio(p.x-y,e)
                    m = midpoint(e)
                    z = midpoint(p.x,y)
                    j = e.label
                    b[i] += lrr*dot(GP[i] - GP[j], m-z)
                end
            end
        end
    end
end

function find_pressure!(solver::PressureSolver, dt::Float64, niter::Int64 = 5)
    refresh!(solver.A, solver.grid, dt)
    for it in 1:niter
        refresh!(solver, dt, (it > 1))
        minres!(solver.ms, solver.A, solver.b, solver.P; verbose = Int(solver.verbose), atol = 1e-6, rtol = 1e-6, itmax = 1000)
        P = solution(solver.ms)
        for i in eachindex(P)
            @inbounds solver.grid.polygons[i].P = P[i]
        end
    end
end