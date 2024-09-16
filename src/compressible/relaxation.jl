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

function mul!(y::ThreadedVec{Float64}, A::MultiphaseProjector, x::ThreadedVec{Float64})::ThreadedVec{Float64}
    @batch for i in 1:A.n
        A.tmp_vec[i] = VEC0
        p = A.grid.polygons[i]
        @inbounds begin
            for (q,e) in neighbors(p, A.grid)
                if p.phase != q.phase
                    pq = get_arrow(p.x,q.x,A.grid)
                    lrr = lr_ratio(pq,e)
                    m = 0.5*(e.v1 + e.v2)
                    j = e.label
                    A.tmp_vec[i] -= lrr*(x[i] - x[j])*(m - p.x)
                end
            end
        end
        A.tmp_vec[i] /= area(p)
    end
    @batch for i in 1:A.n
        y[i] = 0.0
        p = A.grid.polygons[i]
        @inbounds begin
            for (q,e) in neighbors(p, A.grid)
                if p.phase != q.phase
                    pq = get_arrow(p.x,q.x,A.grid)
                    lrr = lr_ratio(pq,e)
                    m = 0.5*(e.v1 + e.v2)
                    mz = 0.5*get_arrow(m, p.x, A.grid) + 0.5*get_arrow(m, q.x, A.grid)
                    j = e.label
                    y[i] -= lrr*(dot(A.tmp_vec[i] - A.tmp_vec[j], mz) - 0.5*dot(A.tmp_vec[i] + A.tmp_vec[j], pq))
                end
            end
        end
    end
    
    return y
end

Base.:*(A::MultiphaseProjector, y::AbstractVector) = mul!(similar(y), A, y)
Base.eltype(::MultiphaseProjector) = Float64
Base.size(A::MultiphaseProjector) = (A.n, A.n)
Base.size(A::MultiphaseProjector, i::Int) = (i <= 2 ? A.n : 0)

struct Relaxator{T}
    n::Int
    grid::VoronoiGrid{T}
    A::MultiphaseProjector{T}
    b::ThreadedVec{Float64}
    q::ThreadedVec{Float64}
    mr::MinresSolver{Float64, Float64, ThreadedVec{Float64}}
    alpha::Float64
    rusanov::Bool
    multiprojection::Bool
    Relaxator(grid::VoronoiGrid{T}; alpha::Float64 = 20.0, rusanov = true, multiprojection = false) where T <: VoronoiPolygon = begin
        n = length(grid.polygons)
        A = MultiphaseProjector(grid)
        b = ThreadedVec{Float64}(undef, n)
        q = ThreadedVec{Float64}(undef, n)
        mr = MinresSolver(A, b)
        return new{T}(n, grid, A, b, q, mr, alpha, rusanov, multiprojection)
    end
end

function refresh!(rx::Relaxator)
    grid = rx.grid
    @batch for i in 1:rx.n
        rx.q[i] = 0.0
        rx.b[i] = 0.0
        p = grid.polygons[i]
        for (q,e) in neighbors(p, grid)
            if p.phase != q.phase
                pq = get_arrow(p.x,q.x,grid)
                lrr = lr_ratio(pq,e)
                m = 0.5*(e.v1 + e.v2)
                mz = 0.5*get_arrow(m, p.x, grid) + 0.5*get_arrow(m, q.x, grid)
                rx.b[i] -= lrr*(dot(p.dv - q.dv, mz) - 0.5*dot(p.dv + q.dv, pq))
            end
        end
    end
end

function multiphase_projection!(rx::Relaxator)
    refresh!(rx)
    minres!(rx.mr, rx.A, rx.b, atol = 1e-4, rtol = 1e-4, itmax = 200)
    stats = statistics(rx.mr)
    if stats.solved == false
        @warn("multiphase projector did not converge within tolerance")
        #return
    end
    y = solution(rx.mr)
    grid = rx.grid
    @batch for i in eachindex(y)
        @inbounds begin
            p = grid.polygons[i]
            if p.quality < 0.25
                continue
            end
            for (q,e) in neighbors(p, grid)
                if p.phase != q.phase
                    j = e.label
                    pq = get_arrow(p.x,q.x,grid)
                    m = 0.5*(e.v1 + e.v2)
                    p.dv += lr_ratio(pq,e)*(y[i] - y[j])*(m - p.x)/area(p)
                end
            end
        end
    end
    return
end

function relaxation_step!(rx::Relaxator, dt::Float64)
    grid = rx.grid
    @batch for p in grid.polygons
        p.dv = VEC0
        p.momentum = p.mass*p.v
        p.energy = p.mass*p.e
        #c = centroid(p)1
        #cx_dist = norm(c - p.x)
        #tol = 0.05*sqrt(area(p))
        #if cx_dist > tol
        #    p.dv = ((1.0 - tol/cx_dist)/dt)*(c - p.x)
        #end
        c = centroid(p)
        #r_char = sqrt(area(p))
        rmax = 0.0
        rmin = Inf
        for (q,e) in neighbors(p, grid)
            pq = get_arrow(p.x,q.x,grid)
            r = norm(pq)
            rmax = max(rmax, r)
            rmin = min(rmin, r)
        end
        p.quality = rmin/rmax
        #p.badness = norm(c - p.x)/sqrt(area(p))
        #lambda = rx.alpha*norm(p.D)
        lambda = norm(p.D)/(p.quality^2)
        p.dv = lambda/(1.0 + dt*lambda)*(c - p.x)
        #p.dv += 0.1*sqrt(area(p))*randn(RealVector)/(p.quality^2)
    end
    if rx.multiprojection
        multiphase_projection!(rx)
    end
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            if !(p.phase == q.phase)
                continue
            end
            pq = get_arrow(p.x,q.x,grid)
            lrr = lr_ratio(pq,e)
            m = 0.5*(e.v1 + e.v2)
            mz = 0.5*get_arrow(m, p.x, grid) + 0.5*get_arrow(m, q.x, grid)
            
            pdvpq = dot(p.dv, pq)
            qdvpq = dot(q.dv, pq)
            pdvmz = dot(p.dv, mz)
            qdvmz = dot(q.dv, mz)
            p.mass += dt*lrr*((pdvmz*p.rho - qdvmz*q.rho) - 0.5*(pdvpq*p.rho + qdvpq*q.rho))
            p.momentum += dt*lrr*((pdvmz*p.rho*p.v - qdvmz*q.rho*q.v) - 0.5*(pdvpq*p.rho*p.v + qdvpq*q.rho*q.v))
            p.energy += dt*lrr*((pdvmz*p.rho*p.e - qdvmz*q.rho*q.e) - 0.5*(pdvpq*p.rho*p.e + qdvpq*q.rho*q.e))

            if rx.rusanov
                # Rusanov-like term helps with entropy
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
    #remesh!(grid)
end