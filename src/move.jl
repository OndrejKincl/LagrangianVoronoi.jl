function move!(grid::VoronoiGrid, dt::Float64)
    @batch for p in grid.polygons
        if any(isnan, p.v)
            throw("Velocity field invalidated.") 
        end
        new_x = periodic_proj(grid, p.x + dt*p.v)
        if isinside(grid.boundary_rect, new_x)
            p.x = new_x
        else # try to project v to tangent space
            for e in boundaries(p)
                n = normal_vector(e)
                p.v -= dot(p.v, n)*n
            end
            new_x = periodic_proj(grid, p.x + dt*p.v)
            if isinside(grid.boundary_rect, new_x)
                p.x = new_x
            else # give up and halt the particle
                p.v = VEC0
            end
        end
    end
    remesh!(grid)
end