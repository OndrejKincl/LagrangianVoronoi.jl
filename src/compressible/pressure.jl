function pressure_step!(grid::VoronoiGrid, dt::Float64)
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            m = 0.5*(e.v1 + e.v2)
            pq = get_arrow(p.x, q.x, grid)
            p.v += dt*lr_ratio(pq,e)/p.mass*(p.P - q.P)*(m - p.x)
        end
    end
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            pq = get_arrow(p.x, q.x, grid)
            lrr = lr_ratio(pq,e)
            m = 0.5*(e.v1 + e.v2)
            mp = get_arrow(m, p.x, grid)
            mq = get_arrow(m, q.x, grid)
            #z = 0.5*(p.x + q.x)
            #p.e += 0.5*dt*lrr/p.mass*dot(p.x - q.x, p.P*q.v + q.P*p.v)
            #p.e -= dt*lrr/p.mass*dot(m - z, p.P*q.v - q.P*p.v)
            p.e -= dt*lrr/p.mass*(dot(mp, q.P*p.v) - dot(mq, p.P*q.v))
        end
    end
end

function ideal_eos!(grid::VoronoiGrid, gamma::Float64 = 1.4, Pmin = 1e-6)
    @batch for p in grid.polygons
        p.rho = p.mass/area(p)
        eps = p.e - 0.5*norm_squared(p.v)
        p.P = (gamma - 1.0)*p.rho*eps
        p.c2 = gamma*max(p.P, Pmin)/p.rho
    end
end

function stiffened_eos!(grid::VoronoiGrid, gamma::Float64 = 1.4, P0::Float64 = 0.0)
    @batch for p in grid.polygons
        p.rho = p.mass/area(p)
        eps = p.e - 0.5*norm_squared(p.v)
        p.P = (gamma - 1.0)*p.rho*eps
        p.c2 = gamma*(p.P + P0)/p.rho
    end
end

function gravity_step!(grid::VoronoiGrid, g::RealVector, dt::Float64)
    @batch for p in grid.polygons
        p.v += dt*g
        p.e += dot(dt*g, p.v)
    end
end