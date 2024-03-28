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
    grid::VoronoiGrid{PhysFields}
    grads::Vector{RealVector}
    lr_ratios::Vector{PreAllocVector{Float64}}
    LaplaceOperator(grid) = begin
        n = length(grid.polygons)
        #neighbors = [PreAllocVector{Int}(CELL_SIZEHINT) for _ in 1:n]
        lr_ratios = [PreAllocVector{Float64}(CELL_SIZEHINT) for _ in 1:n]
        grads = zeros(RealVector, n)
        #mass = zeros(n)
        #ones = ThreadedVec([1.0 for _ in 1:n])
        return new(n, grid, grads, lr_ratios)
    end
end

function mul!(y::ThreadedVec{Float64}, A::LaplaceOperator, x::ThreadedVec{Float64})::ThreadedVec{Float64}
    grid = A.grid
    @threads for i in 1:A.n
        @inbounds begin
            A.grads[i] = VEC0
            p = grid.polygons[i]
            k = 0
            for e in p.edges
                if isboundary(e)
                    continue
                end
                k += 1
                j = e.label
                
                lrr = A.lr_ratios[i][k]
                #q = grid.polygons[j]
                #lrr = lr_ratio(p, q, e)
                m = 0.5*(e.v1 + e.v2)
                A.grads[i] -= lrr*(x[i] - x[j])*(m - p.x)
            end
            A.grads[i] = A.grads[i]/p.var.mass
        end
    end
    @threads for i in 1:A.n
        @inbounds begin
            div = 0.0
            p = grid.polygons[i]
            k = 0
            for e in p.edges
                if isboundary(e)
                    continue
                end
                k += 1
                j = e.label
                q = grid.polygons[j]
                m = 0.5*(e.v1 + e.v2)
                z = 0.5*(p.x + q.x)
                #lrr = lr_ratio(p, q, e)
                lrr = A.lr_ratios[i][k]
                div += lrr*dot(A.grads[i] - A.grads[j], m - z)
                div -= 0.5*lrr*dot(A.grads[i] + A.grads[j], p.x - q.x)
            end
            y[i] = -div #/area(p)
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
    @threads for i in 1:M.n
        @inbounds begin
            y[i] = x[i]*M.diag[i]
        end
    end
    return y
end

struct PressureSolver{T}
    A::LaplaceOperator
    b::ThreadedVec
    M::JacobiPreconditioner
    P::ThreadedVec
    ms::MinresSolver
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
    @threads for i in 1:A.n
        @inbounds begin
            div = 0.0
            #grad_P = VEC0
            #grad_rho = VEC0
            p = polygons[i]
            solver.P[i] = p.var.P
            #empty!(A.neighbors[i])
            empty!(A.lr_ratios[i])
            M.diag[i] = 0.0
            for e in p.edges
                if isboundary(e)
                    continue
                end
                j = e.label
                #push!(A.neighbors[i], j)
                q = polygons[j]
                lrr = lr_ratio(p, q, e)
                push!(A.lr_ratios[i], lrr)
                M.diag[i] += lrr
                m = 0.5*(e.v1 + e.v2)
                z = 0.5*(p.x + q.x)
                #div += lrr*dot(p.var.v - q.var.v, m - q.x)
                div += lrr*(dot(p.var.v - q.var.v, m - z) - 0.5*dot(p.var.v + q.var.v, p.x - q.x))
                #=
                if !constant_density
                    z = 0.5*(p.x + q.x)
                    grad_P -= lrr*(p.var.P - q.var.P)*(m - z) 
                    grad_P -= 0.5*lrr*(p.var.P + q.var.P)*(p.x - q.x)
                    grad_rho += lrr*(p.var.rho - q.var.rho)*(m - q.x)
                end
                =#
                #div /= area(p)
            end
            solver.b[i] = -div*(p.var.rho/dt)# + dot(grad_P, grad_rho)/p.var.rho
            M.diag[i] = 1.0/M.diag[i]
        end
    end
    # remove avg values from P an b
    #=
    ones = ThreadedVec([1.0 for _ in 1:n])
    sP = dot(solver.P, A.ones)/A.n
    sb = dot(solver.b, A.ones)/A.n
    axpy!(-sP, A.ones, solver.P)
    axpy!(-sb, A.ones, solver.P)
    =#
end


function find_pressure!(solver::PressureSolver, dt::Float64;
    constant_density::Bool = false)
    refresh!(solver, dt, constant_density)
    minres!(solver.ms, solver.A, solver.b, solver.P; verbose = Int(solver.verbose)) #, M = solver.M)
    x = solution(solver.ms)
    polygons = solver.grid.polygons
    @threads for i in eachindex(x)
        polygons[i].var.P = x[i]
    end
end

function get_mass!(p::VoronoiPolygon)
    p.var.mass = p.var.rho*area(p)
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
    p.var.a += (1.0/p.var.mass)*lr_ratio(p,q,e)*(p.var.P - q.var.P)*(m - p.x)
    #    (p.var.P - q.var.P)*(m - z) 
    #    + 0.5*(p.var.P + q.var.P)*(p.x - q.x) 
    #) 
end

function no_slip!(p::VoronoiPolygon)
    if isboundary(p)
        p.var.v = VEC0
    end
end