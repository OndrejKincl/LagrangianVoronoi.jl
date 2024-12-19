mutable struct ThreadedMat
    rows::Vector{FastVector{Int}}
    vals::Vector{FastVector{Float64}}
    n::Int
    ThreadedMat(n::Int) = begin
        rows = [FastVector{Int}(POLYGON_SIZEHINT) for _ in 1:n]
        vals = [FastVector{Float64}(POLYGON_SIZEHINT) for _ in 1:n]
        return new(rows, vals, n)
    end
end
ThreadedMat(grid::VoronoiGrid) = ThreadedMat(length(grid.polygons))

function mul!(y::ThreadedVec{Float64}, A::ThreadedMat, x::ThreadedVec{Float64})::ThreadedVec{Float64}
    @batch for i in 1:A.n
         begin
            y[i] = 0.0
            for (k,j) in enumerate(A.rows[i])
                y[i] += A.vals[i][k]*x[j]
            end
        end
    end
    return y
end

Base.:*(A::ThreadedMat, y::AbstractVector) = mul!(similar(y), A, y)
Base.eltype(::ThreadedMat) = Float64
Base.size(A::ThreadedMat) = (A.n, A.n)
Base.size(A::ThreadedMat, i::Int) = (i <= 2 ? A.n : 0)

function clear!(A::ThreadedMat)
    @batch for i in 1:A.n
        empty!(A.rows[i])
        empty!(A.vals[i])
    end 
end

"""
    ViscousSolver(grid::VoronoiGrid{T}; verbose::Int=0) where T

This structure implements an implicit viscous fractional operator. Useful for multiphase problems
or when the viscosity is relatively large.
"""
mutable struct ViscousSolver{T}
    grid::VoronoiGrid{T}
    A::ThreadedMat   # lhs operator
    u0::ThreadedVec{Float64}  # initial guess
    u::ThreadedVec{Float64}   # solution
    b::ThreadedVec{Float64}   # rhs
    ms::MinresSolver{Float64, Float64, ThreadedVec{Float64}}
    verbose::Int
    ViscousSolver(grid::VoronoiGrid{T}; verbose::Int=0) where T = begin
        n = length(grid.polygons)
        A = ThreadedMat(2n)
        u0 = ThreadedVec{Float64}(undef,2n)  
        u = ThreadedVec{Float64}(undef,2n)
        b = ThreadedVec{Float64}(undef,2n)
        ms = MinresSolver(A, b)
        return new{T}(grid, A, u0, u, b, ms, verbose)
    end
end

function add_element!(A::ThreadedMat, i::Int, j::Int, val::Float64)
    push!(A.rows[i], j)
    push!(A.vals[i], val)
end

function v_index(i::Int)::NTuple{2,Int}
    i0 = 2*(i-1)
    return (i0+1, i0+2)
end

function add_v_miniblock!(A::ThreadedMat, i::Int, j::Int, vv::Float64)
    i1, i2 = v_index(i)
    j1, j2 = v_index(j)
    add_element!(A, i1, j1, vv)
    add_element!(A, i2, j2, vv)
end

function assemble!(solver::ViscousSolver, dt::Float64)
    grid = solver.grid
    A = solver.A
    b = solver.b
    u0 = solver.u0
    n = length(grid.polygons)
    clear!(A)
    @batch for i in 1:n
        p = grid.polygons[i]
        i1, i2 = v_index(i)
        # init rhs
        b[i1] = p.mass/dt*p.v[1]
        b[i2] = p.mass/dt*p.v[2]
        # init first guess
        u0[i1] = p.v[1]
        u0[i2] = p.v[2]
        # self-interacting elements
        vv = p.mass/dt
        for (q,e,y) in neighbors(p, grid)
            x = p.x
            j = e.label
            lrr = lr_ratio(x-y,e)
            _vv = -0.5*lrr*(p.mu + q.mu)
            add_v_miniblock!(A, i, j, _vv)
            vv -= _vv
        end
        add_v_miniblock!(A, i, i, vv)
    end
end

"""
    viscous_step!(solver::ViscousSolver, dt::Float64)

Apply viscousity in an implicit way. The viscous coefficient `mu` must be initialized for
every cell.
"""
function viscous_step!(solver::ViscousSolver, dt::Float64)
    assemble!(solver, dt)
    minres!(
        solver.ms, 
        solver.A, solver.b, solver.u0; 
        verbose = solver.verbose, 
        atol = 1e-6, rtol = 1e-6, itmax = 2000
    )
    u = solution(solver.ms)
    @batch for i in eachindex(solver.grid.polygons)
        p = solver.grid.polygons[i]
        i1, i2 = v_index(i)
        p.v = RealVector(u[i1], u[i2])
    end
end