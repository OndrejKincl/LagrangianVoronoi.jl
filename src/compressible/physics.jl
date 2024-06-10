function pforce!(grid::VoronoiGrid, dt::Float64)
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            m = 0.5*(e.v1 + e.v2)
            p.v += (dt/p.mass)*lr_ratio(p,q,e)*(p.P - q.P)*(m - p.x)
        end
    end
end

function viscous_step!(grid::GridNSc, dt::Float64, dr::Float64, gamma::Float64, nu::Float64 = 0.0; no_slip = true)
    #find velocity deformation
    @batch for p in grid.polygons
        D = MAT0
        for (q,e) in neighbors(p, grid)
            lrr = lr_ratio(p, q, e)
            m = 0.5*(e.v1 + e.v2)
            z = 0.5*(p.x + q.x)
            D += lrr*outer(m - z, p.v - q.v)
            if no_slip
                D -= lrr*outer(p.x - q.x, 0.5*(p.v + q.v))
            else
                D += lrr*outer(p.x - q.x, 0.5*(p.v - q.v))
            end
        end
        D = 0.5/p.area*(D + transpose(D))
        div = dot(D, MAT1)
        mu = p.rho*(nu + 0.5*dr^2*max(-div,0.0))
        p.S = 2.0*mu*D
        # pressure update
        # p.P += 2.0*dt*(gamma - 1.0)*dot(D, p.S)
    end
    #accelerate
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            m = 0.5*(e.v1 + e.v2)
            z = 0.5*(p.x + q.x)
            lrr = lr_ratio(p,q,e)
            p.v -= (dt/p.mass)*lrr*(p.S - q.S)*(m - p.x)
            p.e -= dt/p.mass*lrr*dot(m - z, p.S*q.v - q.S*p.v)
            p.e -= dt/p.mass*lrr*dot(p.x - q.x, 0.5*(p.S*q.v + q.S*p.v))
        end
    end
end

function energy_interaction(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge, dt::Float64)
    lrr = lr_ratio(p,q,e)
    m = 0.5*(e.v1 + e.v2)
    z = 0.5*(p.x + q.x)
    p.e += dt/p.mass*lrr*dot(m - z, p.P*q.v - q.P*p.v)
    p.e += dt/p.mass*lrr*dot(p.x - q.x, 0.5*(p.P*q.v + q.P*p.v))
end

function energy_balance!(grid::GridNSc, dt::Float64)
    apply_binary!(grid, (p,q,e) -> energy_interaction(p,q,e,dt))
end

