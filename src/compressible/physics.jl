function find_dv!(grid::VoronoiGrid, t_relax::Float64, dt::Float64)#dv0::Float64, dt::Float64, noise::Float64 = 0.0)
    @batch for p in grid.polygons
        c = centroid(p)
        L_char = sqrt(p.area)
        #p.dv = isboundary(p) ? VEC0 : dv0*(c - p.x)/(L_char + dt*dv0) + dv0*noise*randn(RealVector)
        p.dv = isboundary(p) ? VEC0 : (c - p.x)/(dt + t_relax)
    end
end

function dv_step!(grid::VoronoiGrid, dt::Float64)
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            lrr = lr_ratio(p, q, e)
            m = 0.5*(e.v1 + e.v2)
            z = 0.5*(p.x + q.x)
            
            #p.M -= dt*lrr*dot(m - z, p.rho*p.dv - q.rho*q.dv) 
            p.U -= dt*lrr*(dot(m - z, p.rho*p.dv)*p.v - dot(m - z, q.rho*q.dv)*q.v)
            p.E -= dt*lrr*dot(m - z, p.rho*p.e*p.dv - q.rho*q.e*q.dv)

            pq = p.x - q.x
            #p.M -= 0.5*dt*lrr*dot(pq, p.rho*p.dv + q.rho*q.dv)
            p.U -= 0.5*dt*lrr*(dot(pq, p.dv)*p.v + dot(pq, q.dv)*q.v)
            p.E -= 0.5*dt*lrr*dot(pq, p.rho*p.e*p.dv + q.rho*q.e*q.dv)
        end
    end
    
    @batch for p in grid.polygons
        p.x += dt*p.dv
    end
    remesh!(grid)
    find_rho!(grid)
end

function viscous_step!(grid::VoronoiGrid, dt::Float64, dr::Float64, nu::Float64 = 0.0; no_slip = true)
    #find viscous stress
    @batch for p in grid.polygons
        D = MAT0
        for (q,e) in neighbors(p, grid)
            lrr = lr_ratio(p, q, e)
            m = 0.5*(e.v1 + e.v2)
            z = 0.5*(p.x + q.x)
            D += lrr*outer(m - z, p.v - q.v)
            sgn = (no_slip ? 1.0 : -1.0)
            D -= sgn*lrr*outer(p.x - q.x, 0.5*(p.v + q.v))
        end
        D = 0.5/p.area*(D + transpose(D))
        div = dot(D, MAT1)
        mu = p.rho*(nu+ 0.5*dr^2*max(-div,0.0))
        p.S = 2.0*mu*D
    end
    #accelerate
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            m = 0.5*(e.v1 + e.v2)
            z = 0.5*(p.x + q.x)
            lrr = lr_ratio(p,q,e)
            p.U -= dt*lrr*(p.S - q.S)*(m - p.x)
            p.E -= dt*lrr*dot(m - z, p.S*q.v - q.S*p.v)
            p.E -= dt*lrr*dot(p.x - q.x, 0.5*(p.S*q.v + q.S*p.v))
        end
    end
end

function ideal_eos!(grid::VoronoiGrid, gamma::Float64, Pmin::Float64 = 0.0)
    @batch for p in grid.polygons
        p.v = p.U/p.M
        p.e = p.E/p.M
        p.P = (gamma-1.0)*p.rho*(p.e - 0.5*norm_squared(p.v))
        p.c2 = gamma*max(p.P, Pmin)/p.rho
    end
end

function pressure_step!(grid::VoronoiGrid, dt::Float64)
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            m = 0.5*(e.v1 + e.v2)
            p.U += dt*lr_ratio(p,q,e)*(p.P - q.P)*(m - p.x)
        end
        p.v = p.U/p.M
    end
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            lrr = lr_ratio(p,q,e)
            m = 0.5*(e.v1 + e.v2)
            z = 0.5*(p.x + q.x)
            p.E += dt*lrr*dot(m - z, p.P*q.v - q.P*p.v)
            p.E += dt*lrr*dot(p.x - q.x, 0.5*(p.P*q.v + q.P*p.v))
        end
        p.e = p.E/p.M
    end
end

#move!

function find_rho!(grid::VoronoiGrid)
    @batch for p in grid.polygons
        p.area = area(p)
        p.rho = p.M/p.area
    end
end