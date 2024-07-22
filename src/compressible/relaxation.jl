function relaxation_step!(grid::VoronoiGrid, dt::Float64; alpha = 20.0)
    # get all extensives and dv
    @batch for p in grid.polygons
        #p.area = area(p)
        #p.mass = p.rho*p.area
        p.momentum = p.mass*p.v
        p.energy = p.mass*p.e
        if isboundary(p)
            continue
        end
        lambda = alpha*norm(p.D)
        c = centroid(p)
        p.dv = lambda*(c - p.x)/(1.0 + dt*lambda)
    end
    # udpate the extensive (+ conservative) variables
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            lrr = lr_ratio(p,q,e)
            m = 0.5*(e.v1 + e.v2)
            z = 0.5*(p.x + q.x)
            
            qdot = -dt*lrr*dot(m - z, q.dv)
            pdot = -dt*lrr*dot(m - z, p.dv)
            p.mass += qdot*p.rho - pdot*q.rho
            p.momentum += qdot*(p.rho*p.v) - pdot*(q.rho*q.v)
            p.energy += qdot*(p.rho*p.e) - pdot*(q.rho*q.e)

            qdot = -0.5*dt*lrr*dot(p.x - q.x, q.dv)
            pdot = -0.5*dt*lrr*dot(p.x - q.x, p.dv)
            p.mass += qdot*p.rho + pdot*q.rho
            p.momentum += qdot*(p.rho*p.v) + pdot*(q.rho*q.v)
            p.energy += qdot*(p.rho*p.e) + pdot*(q.rho*q.e)
        end
    end
    # recover the intensive variables and move
    @batch for p in grid.polygons
        p.rho = p.mass/p.area
        p.v = p.momentum/p.mass
        p.e = p.energy/p.mass
        p.x += dt*p.dv
    end
    remesh!(grid)
end