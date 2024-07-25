function pressure_step!(grid::VoronoiGrid, dt::Float64)
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            lrr = lr_ratio(p,q,e)
            m = 0.5*(e.v1 + e.v2)
            z = 0.5*(p.x + q.x)
            p.e -= dt*lrr/mass*dot(m - z, p.P*q.v - q.P*p.v)
            p.e += 0.5*dt*lrr/mass*dot(p.x - q.x, p.P*q.v + q.P*p.v)
        end
    end
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            lrr = lr_ratio(p,q,e)
            m = 0.5*(e.v1 + e.v2)
            p.v -= dt*lrr/mass*(p.P - q.P)*(m - p.x)
        end
    end
end