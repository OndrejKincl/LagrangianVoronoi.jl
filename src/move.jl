"""
    move!(grid::VoronoiGrid, dt::Float64)

Update the positions of all polygons, moving each by `dt*p.v`. 
The function ensures that points will never escape the computational domain (this would lead to undefined behavior).
If the grid is peridoic, points will be wrapped around automatically.
The grid is remeshed after this update.

Required variables: `v`
"""
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