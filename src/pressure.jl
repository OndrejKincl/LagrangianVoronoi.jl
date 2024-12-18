import Base: eltype, size
import LinearAlgebra: mul!
import SmoothedParticles:wendland2

"""
    pressure_step!(grid::VoronoiGrid, dt::Float64)  

Update the velocity and energy by the pressure field. This assumes that pressure was already determined.
"""
function pressure_step!(grid::VoronoiGrid, dt::Float64)  
    @batch for p in grid.polygons
        for (q,e,y) in neighbors(p, grid)
            lrr = lr_ratio(p.x-y,e)
            m = midpoint(e)
            p.v += dt/(p.mass)*lrr*(p.P - q.P)*(m - p.x)
        end
    end
    @batch for p in grid.polygons
        for (q,e,y) in neighbors(p, grid)
            lrr = lr_ratio(p.x-y,e)
            m = midpoint(e)
            p.e -= dt*lrr/(p.mass)*(dot(m - p.x, p.P*p.v) - dot(m - y, q.P*q.v))
        end
    end
end

"""
    eint(p::VoronoiPolygon)::Float64 

Return the internal energy of a Voronoi Polygon.
"""
function eint(p::VoronoiPolygon)::Float64 
    return p.e - 0.5*norm_squared(p.v)
end

function find_rho!(grid::VoronoiGrid) 
    @batch for p in grid.polygons
        p.rho = p.mass/area(p)
    end
end

"""
    ideal_eos!(grid::VoronoiGrid, gamma = 1.4; Pmin) 

Compute pressure and sound speed from internal energy and density using ideal gas equation of state.
Number `gamma` is adiabatic index and `Pmin` can specify least possible value pressure.
In the semi-implicit scheme, this value of pressure is used as an initial condition for an implicit solver.
"""
function ideal_eos!(grid::VoronoiGrid, gamma::Float64 = 1.4; Pmin::Float64 = 0.0) 
    @batch for p in grid.polygons
        p.rho = p.mass/area(p)
        p.P = (gamma - 1.0)*p.rho*eint(p)
        p.c2 = gamma*max(p.P, Pmin)/p.rho
    end
end

"""
    stiffened_eos!(grid::VoronoiGrid, gamma = 1.4; P0) 

Compute pressure and sound speed from internal energy and density using ideal gas equation of state.
Number `gamma` is adiabatic index and `P0` is the stiffness constant.
Stiffened equation of state fares better for flows with very low Mach number.
"""
function stiffened_eos!(grid::VoronoiGrid, gamma::Float64 = 1.4, P0::Float64 = 0.0)
    @batch for p in grid.polygons
        p.rho = p.mass/area(p)
        p.P = (gamma - 1.0)*p.rho*eint(p)
        p.c2 = gamma*(p.P + P0)/p.rho
    end
end

"""
    gravity_step!(grid::VoronoiGrid, g::RealVector, dt::Float64)

Update velocity field and specific energy by a gravitational force.
"""
function gravity_step!(grid::VoronoiGrid, g::RealVector, dt::Float64)
    @batch for p in grid.polygons
        p.v += dt*g
        #p.e += dot(dt*g, p.v)
    end
end

"""
    PressureOperator(grid::VoronoiGrid)

Construct the operator for pressure system. It is symmetric and positive-definite.
""" 
struct PressureOperator{T}
    n::Int
    neighbors::Vector{FastVector{Int}}
    lr_ratios::Vector{FastVector{Float64}}
    diagonal::Vector{Float64}
    PressureOperator(grid::VoronoiGrid{T}) where T <: VoronoiPolygon = begin
        n = length(grid.polygons)
        neighbors = [FastVector{Int}(POLYGON_SIZEHINT) for _ in 1:n]
        lr_ratios = [FastVector{Float64}(POLYGON_SIZEHINT) for _ in 1:n]
        diagonal = [1.0 for _ in 1:n]
        return new{T}(n, neighbors, lr_ratios, diagonal)
    end
end

# Update the pressure operator when the mesh or physical field change.
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

"""
    PressureSolver(grid::VoronoiGrid; verbose::Bool)

Construct the solver for pressure system.
""" 
struct PressureSolver{T}
    A::PressureOperator{T}
    b::ThreadedVec{Float64}
    P::ThreadedVec{Float64}
    GP::Vector{RealVector}
    ms::MinresSolver{Float64, Float64, ThreadedVec{Float64}}
    grid::VoronoiGrid{T}
    verbose::Bool
    PressureSolver(grid::VoronoiGrid{T}; verbose::Bool = false) where T <: VoronoiPolygon = begin
        n = length(grid.polygons)
        A = PressureOperator(grid)
        b = ThreadedVec{Float64}(undef, n)
        P = ThreadedVec{Float64}(undef, n)
        GP = zeros(RealVector, n)
        ms = MinresSolver(A, b)
        return new{T}(A, b, P, GP, ms, grid, verbose)
    end
end

# Update the pressure solver when the mesh or physical field change.
function refresh!(solver::PressureSolver, dt::Float64, gp_step::Bool, boundary_velocity::Function)
    grid = solver.grid
    b = solver.b
    P = solver.P
    GP = solver.GP
    @batch for i in eachindex(grid.polygons)
        @inbounds begin
            p = grid.polygons[i]
            A = area(p)
            b[i] = A*p.P/(p.rho*p.c2*dt^2)
            P[i] = p.P # serves as an initial guess
            GP[i] = VEC0
            for (q,e,y) in neighbors(p, grid)
                lrr = lr_ratio(p.x-y,e)
                m = midpoint(e)
                b[i] -= (lrr/dt)*dot(p.v - q.v, m-y)
                GP[i] -= lrr*(p.P - q.P)*(m - p.x)
            end
            for e in boundaries(p)
                dS = RealVector(e.v1[2]-e.v2[2], e.v2[1]-e.v1[1])
                vbc = boundary_velocity(midpoint(e), e.label)
                b[i] -= dot(dS, vbc - p.v)/dt
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

function zero_vbc(_::RealVector, _::Int)::RealVector
    return VEC0
end

"""
    find_pressure!(solver::PressureSolver, dt::Float64, niter::Int64)

Find the estimated value of pressure field at the next time step. 
Integer `niter` is the number fixed point iteratrions.
""" 
function find_pressure!(solver::PressureSolver, dt::Float64, niter::Int64 = 10; boundary_velocity::Function = zero_vbc)
    refresh!(solver.A, solver.grid, dt)
    for it in 1:niter
        refresh!(solver, dt, (it > 1), boundary_velocity)
        minres!(solver.ms, solver.A, solver.b, solver.P; verbose = Int(solver.verbose), atol = 1e-6, rtol = 1e-6, itmax = 1000)
        P = solution(solver.ms)
        @batch for i in eachindex(P)
            @inbounds solver.grid.polygons[i].P = P[i]
        end
    end
end