using SmoothedParticles:wendland2, rDwendland2

function find_D!(grid::VoronoiGrid, no_slip = true)
    #find viscous stress
    @batch for p in grid.polygons
        p.D = MAT0
        for (q,e) in neighbors(p, grid)
            lrr = lr_ratio(p, q, e)
            m = 0.5*(e.v1 + e.v2)
            z = 0.5*(p.x + q.x)
            #p.D -= lrr*(outer(p.v - q.v, m - p.x))
            p.D += lrr*(outer(p.v - q.v, m - z) - 0.5*outer(p.v + q.v, p.x - q.x))
        end
        p.D = 0.5/p.area*(p.D + transpose(p.D))
    end
end

function find_mu!(grid::VoronoiGrid, dr::Float64, nu::Float64 = 0.0)
    @batch for p in grid.polygons
        div = dot(p.D, MAT1)
        p.mu = p.rho*nu
        p.mu += 0.75*dr^2*max(-div,0.0)
    end
end

function viscous_step!(grid::VoronoiGrid, dt::Float64)
    # accelerate
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            m = 0.5*(e.v1 + e.v2)
            lrr = lr_ratio(p,q,e)
            p.v -= dt/p.mass*lrr*(2.0*p.mu*p.D - 2.0*q.mu*q.D)*(m - p.x)
        end
    end
    # update energy
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            m = 0.5*(e.v1 + e.v2)
            z = 0.5*(p.x + q.x)
            lrr = lr_ratio(p,q,e)
            pSv = 2.0*p.mu*(p.D*p.v- 1.0/3*dot(p.D, MAT1)*p.v)
            qSv = 2.0*q.mu*(q.D*q.v- 1.0/3*dot(q.D, MAT1)*q.v)
            p.e -= dt/p.mass*lrr*dot(m - z, pSv - qSv)
            p.e -= dt/p.mass*lrr*dot(p.x - q.x, 0.5*(pSv + qSv))
        end
    end
end

function ideal_eos!(grid::VoronoiGrid, gamma::Float64, Pmin::Float64 = 0.0)
    @batch for p in grid.polygons
        p.P = (gamma-1.0)*p.rho*(p.e - 0.5*norm_squared(p.v))
        p.c2 = gamma*max(p.P, Pmin)/p.rho
    end
end

function stiffened_eos!(grid::VoronoiGrid, gamma::Float64, rho0::Float64, P0::Float64)
    @batch for p in grid.polygons
        p.P = (gamma-1.0)*p.rho*(p.e - p.u - 0.5*norm_squared(p.v)) #+ P0*(1.0 - p.rho/rho0)
        p.c2 = (gamma*p.P + P0)/p.rho
    end
end

#find_pressure!

function pressure_step!(grid::VoronoiGrid, dt::Float64)
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            m = 0.5*(e.v1 + e.v2)
            p.v += dt/p.mass*lr_ratio(p,q,e)*(p.P - q.P)*(m - p.x)
        end
    end
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            lrr = lr_ratio(p,q,e)
            m = 0.5*(e.v1 + e.v2)
            z = 0.5*(p.x + q.x)
            p.e += dt/p.mass*lrr*dot(m - z, p.P*q.v - q.P*p.v)
            p.e += dt/p.mass*lrr*dot(p.x - q.x, 0.5*(p.P*q.v + q.P*p.v))
        end
    end
end

function update_energy!(grid::VoronoiGrid, gamma::Float64)
    @batch for p in grid.polygons
        p.e = 0.5*norm_squared(p.v) + p.P/(p.rho*(gamma - 1.0))
    end
end

#=
function lloyd_step!(grid::VoronoiGrid, dt::Float64, rate::Float64 = 20.0)
    @batch for p in grid.polygons
        a = rate*norm(p.D, 2)*dt
        p.x = (p.x + a*centroid(p))/(1.0 + a)
    end
end
=#


function find_lloyd_potential!(grid::VoronoiGrid, alpha::Float64)
    @batch for p in grid.polygons
        if isboundary(p)
            continue
        end
        p.u = 0.0
        for e in p.edges
            A = 0.5*abs(cross2(e.v1 - p.x, e.v2 - p.x))
            # this triangle rule is exact for quad polys
            p.u -=  (9*A/16)*norm_squared(-2/3*p.x + 1/3*e.v1 + 1/3*e.v2)
            p.u += (25*A/48)*norm_squared(-2/5*p.x + 1/5*e.v1 + 1/5*e.v2)
            p.u += (25*A/48)*norm_squared(-4/5*p.x + 3/5*e.v1 + 1/5*e.v2)
            p.u += (25*A/48)*norm_squared(-4/5*p.x + 1/5*e.v1 + 3/5*e.v2)
        end
        p.u = 0.5*alpha/p.mass*p.u
    end
end


function lloyd_step!(grid::VoronoiGrid, dt::Float64, alpha::Float64)
    # update the potential
    #find_lloyd_potential!(grid, alpha)    
    # update the velocities
    @batch for p in grid.polygons
        if isboundary(p)
            continue
        end
        lambda = dt*alpha
        p.v += lambda*(centroid(p) - p.x)
    end
    # update the energies
    #=
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            a = e.v1
            b = e.v2
            m = 0.5*e.v1 + 0.5*e.v2
            lrr = lr_ratio(p,q,e)
            # simpson rule is exact for cubic polys
            p.e += dt*0.5/6.0*alpha*lrr/p.mass*(norm_squared(a - q.x)*dot(a - p.x, p.v) - norm_squared(a - p.x)*dot(a - q.x, q.v))
            p.e += dt*2.0/6.0*alpha*lrr/p.mass*(norm_squared(m - q.x)*dot(m - p.x, p.v) - norm_squared(m - p.x)*dot(m - q.x, q.v))
            p.e += dt*0.5/6.0*alpha*lrr/p.mass*(norm_squared(b - q.x)*dot(b - p.x, p.v) - norm_squared(b - p.x)*dot(b - q.x, q.v))
        end
    end
    =#
end

#move!

function SPH_find_rho!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64, h::Float64)
	p.rho_SPH += q.mass*wendland2(h,r)
    return
end

function SPH_update_v!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64, P::Float64, h::Float64, dt::Float64)
	p.v -= dt*q.mass*rDwendland2(h,r)*(P/p.rho_SPH^2 + P/q.rho_SPH^2)*(p.x - q.x)
    return
end

function SPH_update_e!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64, P::Float64, h::Float64, dt::Float64)
	p.e -= dt*q.mass*rDwendland2(h,r)*dot(P*q.v/p.rho_SPH^2 + P*p.v/q.rho_SPH^2, p.x - q.x)
    return
end

function SPH_stabilizer!(grid::VoronoiGrid, P::Float64, h::Float64, dt::Float64)
    @batch for p in grid.polygons
        p.rho_SPH = p.mass*wendland2(h,0.0)
    end
    apply_local!(grid, (p::VoronoiPolygon,q::VoronoiPolygon,r::Float64) -> SPH_find_rho!(p,q,r,h), h)
    apply_local!(grid, (p::VoronoiPolygon,q::VoronoiPolygon,r::Float64) -> SPH_update_v!(p,q,r,P,h,dt), h)
    apply_local!(grid, (p::VoronoiPolygon,q::VoronoiPolygon,r::Float64) -> SPH_update_e!(p,q,r,P,h,dt), h)
    @batch for p in grid.polygons
        p.u = -P/p.rho_SPH
    end
end

function find_rho!(grid::VoronoiGrid)
    @batch for p in grid.polygons
        p.area = area(p)
        p.rho = p.mass/p.area
    end
end