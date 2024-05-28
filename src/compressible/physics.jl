function pressure_force!(grid::GridNSc, dt::Float64; stabilize = true)
    apply_binary!(grid, pressure_interaction!)
    if stabilize
        @batch for p in grid.polygons
            div = 0.0 #divergence of velocity
            for (q,e) in neighbors(p, grid)
                q = grid.polygons[e.label]
                lrr = lr_ratio(p, q, e)
                m = 0.5*(e.v1 + e.v2)
                z = 0.5*(p.x + q.x)
                div += lrr*dot(m - z, p.v - q.v)
                div -= lrr*dot(p.x - q.x, 0.5*(p.v + q.v))
            end
            LapP = div*p.rho/dt
            if LapP > 0.0
                c = centroid(p)
                A = 1.5*LapP/(p.mass)*(c - p.x)
                v0 = p.v + dt*p.a
                v1 = v0 + dt*A
                if norm_squared(v0) >= norm_squared(v1)
                    p.a += A
                end
            end
        end
    end
    accelerate!(grid, dt)
end

function viscous_step!(grid::GridNSc, dt::Float64, dr::Float64, nu::Float64 = 0.0)
    #find velocity deformation
    @batch for p in grid.polygons
        p.D = MAT0
        for (q,e) in neighbors(p, grid)
            lrr = lr_ratio(p, q, e)
            m = 0.5*(e.v1 + e.v2)
            z = 0.5*(p.x + q.x)
            p.D += lrr*outer(m - z, p.v - q.v)
            p.D -= lrr*outer(p.x - q.x, 0.5*(p.v + q.v))
        end
        p.D = 0.5*(p.rho/p.mass)*(p.D + transpose(p.D))
        pdiv = min(dot(p.D, MAT1), 0.0)
        p.mu = p.rho*(nu - 0.5*dr^2*pdiv) 
    end
    #accelerate
    @batch for p in grid.polygons
        pS = 2.0*p.mu*p.D
        for (q,e) in neighbors(p, grid)
            m = 0.5*(e.v1 + e.v2)
            p.v -= (dt/p.mass)*lr_ratio(p,q,e)*(pS - 2.0*q.mu*q.D)*(m - p.x)
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

function ideal_eos!(grid::GridNSc, gamma::Float64)
    @batch for p in grid.polygons
        p.rho = p.mass/area(p)
        p.P = (gamma - 1.0)*p.rho*(p.e - 0.5*norm_squared(p.v))
        p.c = sqrt(abs(gamma*p.P/p.rho))
    end
end

function ideal_pressurefix!(grid::GridNSc, gamma::Float64, dt)
    @batch for p in grid.polygons
        p.P += 2.0*p.mu*dt*(gamma - 1.0)*dot(p.D, p.D)
        p.c = sqrt(abs(gamma*p.P/p.rho))
    end
end