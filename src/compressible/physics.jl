function ideal_eos!(grid::VoronoiGrid, gamma::Float64)
    @batch for p in grid.polygons
        p.area = area(p)
        p.c2 = gamma*p.P/p.rho
        p.rho = p.mass/p.area
    end
end

function hforce!(grid::VoronoiGrid, dt::Float64)
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            m = 0.5*(e.v1 + e.v2)
            p.v += (dt/p.area)*lr_ratio(p,q,e)*(p.h - q.h)*(m - p.x)
        end
    end
end

function pforce!(grid::VoronoiGrid, dt::Float64)
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            m = 0.5*(e.v1 + e.v2)
            p.v += (dt/p.mass)*lr_ratio(p,q,e)*(p.P - q.P)*(m - p.x)
        end
    end
end