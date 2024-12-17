"""
    find_dv!(grid::VoronoiGrid, dt::Float64, alpha::Float64 = 1.0)

Determine `dv`, the repair velocity. This is the first step of mesh relaxation procedure and
is needed to maintain sufficiently high mesh quality. The repair velocity is chosen to be proportional to Frobenius norm of velocity deformation tensor `D` and inversely
proportional to the mesh quality squared. There is also multiplicative dimensionless
parameter `alpha` with default value `alpha = 1`. Higher `alpha` causes the Voronoi cells
to be rounder (= better) but also less genuinly Lagrangian.
"""
function find_dv!(grid::VoronoiGrid, dt::Float64, alpha::Float64 = 1.0)
    @batch for p in grid.polygons
        p.dv = VEC0
        c = centroid(p)
        rmax = 0.0
        rmin = Inf
        for (q,e,y) in neighbors(p, grid)
            r = norm(p.x-y)
            rmax = max(rmax, r)
            rmin = min(rmin, r)
        end
        p.quality = rmin/rmax
        lambda = alpha*norm(p.D)/(p.quality^2)
        p.dv = lambda/(1.0 + dt*lambda)*(c - p.x)
    end
end

"""
    relaxation_step!(grid::VoronoiGrid, dt::Float64; rusanov::Bool = true)

Update the variables by taking into account the repair velocity `dv`. The repair velocity should be known
before the `relalaxation_step!` is called. See `find_dv!`. The mass, momentum and energy of each cell
is updated by solving one step of an advection problem. A keyword boolean argument `rusanov` controls the use of a
Rusanov approximate Riemann solver. The Rusanov flux is recommended to prevent the generation of new extrema (like negative density) and decrease of entropy.
This is a second step of the mesh relaxation procedure (or maybe third, if your simulation involves a multiphase projector).
"""
function relaxation_step!(grid::VoronoiGrid, dt::Float64; rusanov::Bool = true)
    @batch for p in grid.polygons
        p.momentum = p.mass*p.v
        p.energy = p.mass*p.e
        for (q,e,y) in neighbors(p, grid)
            if !(p.phase == q.phase)
                continue
            end
            lrr = lr_ratio(p.x-y,e)
            m = midpoint(e)
            z = midpoint(p.x,y)
            
            pdvpq = dot(p.dv, p.x-y)
            qdvpq = dot(q.dv, p.x-y)
            pdvmz = dot(p.dv, m-z)
            qdvmz = dot(q.dv, m-z)
            p.mass += dt*lrr*((pdvmz*p.rho - qdvmz*q.rho) - 0.5*(pdvpq*p.rho + qdvpq*q.rho))
            p.momentum += dt*lrr*((pdvmz*p.rho*p.v - qdvmz*q.rho*q.v) - 0.5*(pdvpq*p.rho*p.v + qdvpq*q.rho*q.v))
            p.energy += dt*lrr*((pdvmz*p.rho*p.e - qdvmz*q.rho*q.e) - 0.5*(pdvpq*p.rho*p.e + qdvpq*q.rho*q.e))

            if rusanov
                # Rusanov riemann solver for pure advection
                # this term helps with entropy growth and against new extrema generation
                a = max(norm(p.dv), norm(q.dv))
                l = len(e)
                p.mass += 0.5*dt*l*a*(q.rho - p.rho)
                p.momentum += 0.5*dt*l*a*(q.rho*q.v - p.rho*p.v)
                p.energy += 0.5*dt*l*a*(q.rho*q.e - p.rho*p.e)
            end
        end
    end
    @batch for p in grid.polygons
        p.v = p.momentum/p.mass
        p.e = p.energy/p.mass
        p.x += dt*p.dv
    end
    remesh!(grid)
end

"""
    MultiphaseProjector(grid::VoronoiGrid)

A linear operator which appears in the multiphase projection system. It is symmetric and positive-definite.
"""
struct MultiphaseProjector{T}
    n::Int
    grid::VoronoiGrid{T}
    tmp_vec::Vector{RealVector}
    MultiphaseProjector(grid::VoronoiGrid{T}) where T <: VoronoiPolygon = begin
        n = length(grid.polygons)
        tmp_vec = [VEC0 for _ in 1:n]
        return new{T}(n, grid, tmp_vec)
    end
end

function mul!(res::ThreadedVec{Float64}, A::MultiphaseProjector, x::ThreadedVec{Float64})::ThreadedVec{Float64}
    @batch for i in 1:A.n
        A.tmp_vec[i] = VEC0
        p = A.grid.polygons[i]
        @inbounds begin
            for (q,e,y) in neighbors(p, A.grid)
                if p.phase != q.phase
                    lrr = lr_ratio(p.x-y,e)
                    m = midpoint(e)
                    j = e.label
                    A.tmp_vec[i] -= lrr*(x[i] - x[j])*(m - p.x)
                end
            end
        end
        A.tmp_vec[i] /= area(p)
    end
    @batch for i in 1:A.n
        res[i] = 0.0
        p = A.grid.polygons[i]
        @inbounds begin
            for (q,e,y) in neighbors(p, A.grid)
                if p.phase != q.phase
                    lrr = lr_ratio(p.x-y,e)
                    m = midpoint(e)
                    z = midpoint(p.x, y)
                    j = e.label
                    res[i] -= lrr*(dot(A.tmp_vec[i] - A.tmp_vec[j], m-z) - 0.5*dot(A.tmp_vec[i] + A.tmp_vec[j], p.x-y))
                end
            end
        end
    end
    return res
end

Base.:*(A::MultiphaseProjector, y::AbstractVector) = mul!(similar(y), A, y)
Base.eltype(::MultiphaseProjector) = Float64
Base.size(A::MultiphaseProjector) = (A.n, A.n)
Base.size(A::MultiphaseProjector, i::Int) = (i <= 2 ? A.n : 0)

"""
    MultiphaseSolver(grid::VoronoiGrid, quality_threshold::Float64 = 0.25)

Construct a solver for the multiphase projection step. 
A keyword argument `quality_threshold` determines when `dv` should not be projected because the quality of the cell is too poor.
In these problematic cases, stability of the simulation takes priority over physical accuracy. 
"""
struct MultiphaseSolver{T}
    n::Int
    grid::VoronoiGrid{T}
    A::MultiphaseProjector{T}
    b::ThreadedVec{Float64}
    q::ThreadedVec{Float64}
    mr::MinresSolver{Float64, Float64, ThreadedVec{Float64}}
    quality_threshold::Float64
    MultiphaseSolver(grid::VoronoiGrid{T}; quality_threshold::Float64 = 0.25) where T <: VoronoiPolygon = begin
        n = length(grid.polygons)
        A = MultiphaseProjector(grid)
        b = ThreadedVec{Float64}(undef, n)
        q = ThreadedVec{Float64}(undef, n)
        mr = MinresSolver(A, b)
        return new{T}(n, grid, A, b, q, mr, quality_threshold)
    end
end

function refresh!(solver::MultiphaseSolver)
    grid = solver.grid
    @batch for i in 1:solver.n
        solver.q[i] = 0.0
        solver.b[i] = 0.0
        p = grid.polygons[i]
        for (q,e,y) in neighbors(p, grid)
            if p.phase != q.phase
                lrr = lr_ratio(p.x-y,e)
                m = midpoint(e)
                z = midpoint(p.x,y)
                solver.b[i] -= lrr*(dot(p.dv - q.dv, m-z) - 0.5*dot(p.dv + q.dv, p.x-y))
            end
        end
    end
end

"""
    multiphase_projection!(solver::MultiphaseSolver)

Solve the linear system to find an orthogonal projection of `dv` to a constraint space
which gurantees conservation of area for every fluid phase. You only need this for multiphase flows.
You can also run multiphase problems without it but expect some artificial density ridges between
phases of different densities.
"""
function multiphase_projection!(solver::MultiphaseSolver)
    refresh!(solver)
    minres!(solver.mr, solver.A, solver.b, atol = 1e-4, rtol = 1e-4, itmax = 200)
    stats = statistics(solver.mr)
    if stats.solved == false
        @warn("multiphase projector did not converge within tolerance")
        #return
    end
    res = solution(solver.mr)
    grid = solver.grid
    @batch for i in eachindex(res)
        @inbounds begin
            p = grid.polygons[i]
            if p.quality < solver.quality_threshold
                continue
            end
            for (q,e,y) in neighbors(p, grid)
                if p.phase != q.phase
                    j = e.label
                    m = midpoint(e)
                    p.dv += lr_ratio(p.x-y,e)*(res[i] - res[j])*(m - p.x)/area(p)
                end
            end
        end
    end
    return
end