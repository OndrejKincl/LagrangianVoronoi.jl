function find_D!(grid::VoronoiGrid; noslip::Bool = true)
    @batch for p in grid.polygons
        p.D = MAT0
        for (q,e) in neighbors(p, grid)
            m = 0.5*(e.v1 + e.v2)
            p.D += lr_ratio(p,q,e)*outer(p.v - q.v, m - q.x)
            if noslip
                p.D -= lr_ratio(p,q,e)*outer(p.v, p.x - q.x) 
            end
        end
        A = p.mass/p.rho
        p.D /= A
        p.D = 0.5*(p.D + transpose(p.D))
    end
end

function viscous_step!(grid::VoronoiGrid, dt::Float64; artificial_visc = true)
    @batch for p in grid.polygons
        divv = dot(p.D, MAT1)
        mu = p.mu
        if artificial_visc && (divv < 0.0)
            mu -= divv*p.rho*grid.dr^2
        end
        p.S = 2.0*mu*(p.D - divv*MAT1/3)
    end
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            m = 0.5*(e.v1 + e.v2)
            p.v -= dt*lr_ratio(p,q,e)/p.mass*(p.S - q.S)*(m - p.x)
        end
    end
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            m = 0.5*(e.v1 + e.v2)
            p.e += dt*lr_ratio(p,q,e)/p.mass*(dot(m - p.x, q.S*p.v) - dot(m - q.x, p.S*q.v))
        end
    end
    
    
end