function ideal_temperature!(grid::VoronoiGrid)
    @batch for p in grid.polygons
        eps = p.e - 0.5*norm_squared(p.v)
        p.T = eps/p.cV 
    end
end

function fourier_step!(grid::VoronoiGrid, dt::Float64)
    @batch for p in grid.polygons
        for (q,e) in neighbors(p,grid)
            pq = get_arrow(p.x, q.x, grid)
            lrr = lr_ratio(pq,e)
            k = 0.5*(p.k + q.k)
            p.e -= dt*k/p.mass*lrr*(p.T - q.T)
        end
    end
end

