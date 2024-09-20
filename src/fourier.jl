"""
    ideal_temperature!(grid::VoronoiGrid)

Assign temperature to every Voronoi polygon based on ideal gas law.

Required variables in Voronoi Polygon: `v`, `e`, `cV`.
"""
function ideal_temperature!(grid::VoronoiGrid)
    @batch for p in grid.polygons
        p.T = eint(p)/p.cV 
    end
end

"""
    fourier_step!(grid::VoronoiGrid, dt::Float64)

Update the energy of Voronoi polygon by Fourier heat conduction. 
This assumes that every polygon `p` has its value of heat conductivity `p.k` assigned by the initial condition
or otherwise. 

Required variables in Voronoi Polygon: `mass`, `k`, `T`.
"""
function fourier_step!(grid::VoronoiGrid, dt::Float64)
    @batch for p in grid.polygons
        for (q,e,y) in neighbors(p,grid)
            lrr = lr_ratio(p.x-y,e)
            k = 0.5*(p.k + q.k)
            p.e -= dt*k/p.mass*lrr*(p.T - q.T)
        end
    end
end

