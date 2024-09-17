function ideal_temperature!(grid::VoronoiGrid)
    @batch for p in grid.polygons
        p.T = eint(p)/p.cV 
    end
end

function fourier_step!(grid::VoronoiGrid, dt::Float64)
    @batch for p in grid.polygons
        for (q,e,y) in neighbors(p,grid)
            lrr = lr_ratio(p.x-y,e)
            k = 0.5*(p.k + q.k)
            p.e -= dt*k/p.mass*lrr*(p.T - q.T)
        end
    end
end

