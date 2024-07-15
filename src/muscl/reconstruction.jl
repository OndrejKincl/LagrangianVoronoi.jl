function get_intensives!(grid::VoronoiGrid)
    @batch for p in grid.polygons
        p.A = area(p)
        p.c = centroid(p)
        p.rho = p.M/p.A
        p.u   = p.P/p.A
        p.e   = p.E/p.A
    end
end

function interpolate_vars(p::VoronoiPolygon, x::RealVector)::Tuple{Float64, RealVector, Float64}
    dx = x - p.x
    rho = p.rho + dot(dx, p.Grho)
    u = p.u + p.Gu*dx
    e = p.e + dot(dx, p.Ge)
    return (rho, u, e) 
end

function reconstruction!(grid::VoronoiGrid; slopelimiter::Bool = true, repair_alpha = 20.0)
    @batch for p in grid.polygons
        p.Grho = VEC0
        p.Ge   = VEC0
        p.Gu   = MAT0
        p.D    = MAT0
        R      = MAT0

        rho_max = p.rho 
        rho_min = p.rho
        u_max   = p.u
        u_min   = p.u
        e_min   = p.e
        e_max   = p.e

        for (q,e) in neighbors(p, grid)
            dc = p.c - q.c
            weight = len(e)/norm(dc)
            R += weight*outer(dc, dc)
            p.Grho += weight*(p.rho - q.rho)*dc
            p.Gu += weight*outer(p.u - q.u, dc)
            p.D += weight*outer(p.u/p.rho - q.u/q.rho, dc)
            p.Ge += weight*(p.e - q.e)*dc

            rho_max = max(rho_max, q.rho)
            rho_min = min(rho_min, q.rho)
            u_max   = max.(u_max, q.u)
            u_min   = min.(u_min, q.u)
            e_max   = max(e_max, q.e)
            e_min   = min(e_min, q.e)
        end
        
        R = inv(R)
        p.Grho = R*p.Grho
        p.Gu = p.Gu*R
        p.Ge = R*p.Ge
        p.D = p.D*R
        p.D = 0.5*(p.D + transpose(p.D))
        p.w = p.u/p.rho + repair_alpha*norm(p.D)*(p.c - p.x)

        if slopelimiter
            phi_rho = 1.0
            phi_u1 = 1.0
            phi_u2 = 1.0
            phi_e = 1.0
            for (q,e) in neighbors(p, grid)
                m = 0.5*(e.v1 + e.v2)
                rho, u, e = interpolate_vars(p, m)
                (rho > rho_max) && (phi_rho = min(phi_rho, (rho_max - p.rho)/(rho - p.rho)))
                (rho < rho_min) && (phi_rho = min(phi_rho, (rho_min - p.rho)/(rho - p.rho)))
                (u[1] > u_max[1]) && (phi_u1 = min(phi_u1, (u_max[1] - p.u[1])/(u[1] - p.u[1])))
                (u[1] < u_min[1]) && (phi_u1 = min(phi_u1, (u_min[1] - p.u[1])/(u[1] - p.u[1])))
                (u[2] > u_max[2]) && (phi_u2 = min(phi_u2, (u_max[2] - p.u[2])/(u[2] - p.u[2])))
                (u[2] < u_min[2]) && (phi_u2 = min(phi_u2, (u_min[2] - p.u[2])/(u[2] - p.u[2])))
                (e > e_max) && (phi_e = min(phi_e, (e_max - p.e)/(e - p.e)))
                (e < e_min) && (phi_e = min(phi_e, (e_min - p.e)/(e - p.e)))
            end
            p.Grho = phi_rho*p.Grho
            p.Gu = RealMatrix(phi_u1, 0.0, 0.0, phi_u2)*p.Gu
            p.Ge = phi_e*p.Ge
        end
    end
end