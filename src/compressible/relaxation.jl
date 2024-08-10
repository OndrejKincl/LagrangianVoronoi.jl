function relaxation_step!(grid::VoronoiGrid, dt::Float64, alpha::Float64 = 20.0; rusanov::Bool = true)
    @batch for p in grid.polygons
        p.momentum = p.mass*p.v
        p.energy = p.mass*p.e
        lambda = alpha*norm(p.D)
        p.dv = lambda/(1.0 + dt*lambda)*(centroid(p) - p.x)
    end
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            if !(p.phase == q.phase)
                continue
            end
            lrr = lr_ratio(p,q,e)
            mz = 0.5*(e.v1 + e.v2) - 0.5*(p.x + q.x)
            pq = p.x - q.x
            pdvpq = dot(p.dv, pq)
            qdvpq = dot(q.dv, pq)
            pdvmz = dot(p.dv, mz)
            qdvmz = dot(q.dv, mz)
            p.mass += dt*lrr*((pdvmz*p.rho - qdvmz*q.rho) - 0.5*(pdvpq*p.rho + qdvpq*q.rho))
            p.momentum += dt*lrr*((pdvmz*p.rho*p.v - qdvmz*q.rho*q.v) - 0.5*(pdvpq*p.rho*p.v + qdvpq*q.rho*q.v))
            p.energy += dt*lrr*((pdvmz*p.rho*p.e - qdvmz*q.rho*q.e) - 0.5*(pdvpq*p.rho*p.e + qdvpq*q.rho*q.e))

            if rusanov
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
