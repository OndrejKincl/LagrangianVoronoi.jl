function find_D!(grid::VoronoiGrid; noslip::Bool = true)
    @batch for p in grid.polygons
        p.D = MAT0
        for (q,e,y) in neighbors(p, grid)
            m = midpoint(e)
            lrr = lr_ratio(p.x-y, e)
            p.D += lrr*outer(p.v - q.v, m - y)
            if noslip
                p.D -= lrr*outer(p.v, p.x - y) 
            end
        end
        p.D /= area(p)
        p.D = 0.5*(p.D + transpose(p.D))
    end
end

# viscous stress
function getS(p::VoronoiPolygon, dr::Float64)::RealMatrix 
    divv = dot(p.D, MAT1)
    mu = p.mu
    if divv < 0.0
        mu -= divv*p.rho*dr^2
    end
    return 2.0*mu*(p.D - divv*MAT1/3)
end

function viscous_step!(grid::VoronoiGrid, dt::Float64; artificial_viscosity = true)
    avdr = artificial_viscosity ? grid.dr : 0.0
    @batch for p in grid.polygons
        for (q,e,y) in neighbors(p, grid)
            m = midpoint(e)
            p.v -= dt*lr_ratio(p.x-y,e)/p.mass*(getS(p, avdr) - getS(q, avdr))*(m - p.x)
        end
    end
    @batch for p in grid.polygons
        for (q,e,y) in neighbors(p, grid)
            m = midpoint(e)
            p.e += dt*lr_ratio(p.x-y,e)/p.mass*(dot(m - p.x, getS(p, avdr)*p.v) - dot(m - y, getS(q, avdr)*q.v))
        end
    end
end