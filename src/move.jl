"""
    move!(grid::VoronoiGrid, dt::Float64)

Update the positions of all polygons, moving each by `dt*p.v`. 
The function ensures that points will never escape the computational domain (this would lead to undefined behavior).
If the grid is peridoic, points will be wrapped around automatically.
The grid is remeshed after this update.
"""
function move!(grid::VoronoiGrid, dt::Float64)
    @batch for p in grid.polygons
        try_move!(grid, p, dt) && continue
        for e in boundaries(p) # project v to tangent space
            n = normal_vector(e)
            p.v -= dot(p.v, n)*n
        end
        try_move!(grid, p, dt) && continue
        p.v = VEC0
        try_move!(grid, p, dt)
    end
    remesh!(grid)
end

function try_move!(grid::VoronoiGrid, p::VoronoiPolygon, dt::Float64)::Bool
    if any(isnan, p.v)
        throw("Velocity field invalidated.") 
    end
    _x = periodic_wrap(grid, p.x + dt*p.v)
    if isinside(grid.boundary_rect, _x)
        p.x = _x
        return true
    end
    return false
end