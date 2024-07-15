function ideal_gas_law(rho::Float64, u::RealVector, e::Float64; gamma::Float64 = 1.4)::Tuple{Float64, Float64}
    p = (gamma - 1.0)*(e - 0.5*norm_squared(u)/rho)
    p = max(p, 0.0)
    a = sqrt(gamma*max(p/rho, 0.0))
    return (p, a)
end

function physical_step!(grid::VoronoiGrid; gamma::Float64 = 1.4)
    @batch for p in grid.polygons
        p.Mdot = 0.0
        p.Pdot = VEC0
        p.Edot = 0.0
        for (q,e) in neighbors(p, grid)
            l = len(e)
            n = q.x - p.x
            r = norm(n)
            n /= r
            m = 0.5*(e.v1 + e.v2)
            z = 0.5*(p.x + q.x)
            rho_L, u_L, e_L = interpolate_vars(p, m)
            rho_R, u_R, e_R = interpolate_vars(q, m)
            p_L, a_L = ideal_gas_law(rho_L, u_L, e_L, gamma=gamma) 
            p_R, a_R = ideal_gas_law(rho_R, u_R, e_R, gamma=gamma) 
            vnL = dot(u_L/rho_L, n)
            vnR = dot(u_R/rho_R, n)
            wn = 0.5*dot(n, p.w + q.w) + dot(m - z, p.w - q.w)/r
            a_max = max(a_L, a_R)
            p.Mdot -= l*(0.5*(vnL - wn)*rho_L + 0.5*(vnR - wn)*rho_R - a_max*(rho_R - rho_L))
            p.Pdot -= l*(0.5*(vnL - wn)*u_L   + 0.5*(vnR - wn)*u_R   - a_max*(u_R - u_L) + 0.5*(p_L + p_R)*n)
            p.Edot -= l*(0.5*(vnL - wn)*e_L   + 0.5*(vnR - wn)*e_R   - a_max*(e_R - e_L) + 0.5*(p_L*vnL + p_R*vnR))
        end
    end
end

function update!(grid::VoronoiGrid, dt::Float64)
    @batch for p in grid.polygons
        if !isboundary(p)
            p.M += dt*p.Mdot
            p.P += dt*p.Pdot
            p.E += dt*p.Edot
            p.x += dt*p.w
        end
    end
    remesh!(grid)
end
