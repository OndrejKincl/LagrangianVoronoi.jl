function relaxation_step!(grid::VoronoiGrid, dt::Float64, alpha::Float64 = 20.0)
    @batch for p in grid.polygons
        p.momentum = p.mass*p.v
        p.energy = p.mass*p.e
        lambda = alpha*norm(p.D)
        p.dv = lambda/(1.0 + dt*lambda)*(centroid(p) - p.x)
        drho = VEC0
        rho_min = p.rho
        rho_max = p.rho
        for (q,e) in neighbors(p,grid)
            m = 0.5*(e.v1 + e.v2)
            drho += -lr_ratio(p,q,e)*dot(m - p.x, p.dv)*(p.rho - q.rho)
            rho_min = min(q.rho, rho_min)
            rho_max = max(q.rho, rho_max)
        end
        drho *= dt*p.rho/p.mass
        p.phi_rho = 1.0
        (drho > 0.0) && p.phi_rho = min(p.phi_rho, (rho_max - p.rho)/drho)
        (drho < 0.0) && p.phi_rho = min(p.phi_rho, (rho_min - p.rho)/drho)
    end
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            lrr = lr_ratio(p,q,e)
            m = 0.5*(e.v1 + e.v2)
            dot_p = dot(m - p.x, p.dv)
            dot_q = dot(m - q.x, q.dv)
            p.mass += min(p.phi_rho, q.phi_rho)*dt*lrr*(dot_p*q.rho - dot_q*p.rho)
            p.momentum += dt*lrr*(dot_p*(q.rho*q.v) - dot_q*(p.rho*p.v)) 
            p.energy += dt*lrr*(dot_p*(q.rho*q.e) - dot_q*(p.rho*p.e))
        end
    end
    @batch for p in grid.polygons
        p.v = p.momentum/p.mass
        p.e = p.energy/p.mass
        p.x += dt*p.dv
    end
    #remesh!(grid)
end

