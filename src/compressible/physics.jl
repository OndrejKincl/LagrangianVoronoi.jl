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