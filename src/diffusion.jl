"""
    find_D!(grid::VoronoiGrid)

Compute the velocity deformation tensor 
`` D = \\frac{1}{2}(\\nabla v + \\nabla v^T) ``
and assign it to every polygon.

Required variables: `v`, `D`.
"""
function find_D!(grid::VoronoiGrid)
    @batch for p in grid.polygons
        p.D = MAT0
        for (q,e,y) in neighbors(p, grid)
            m = midpoint(e)
            lrr = lr_ratio(p.x-y, e)
            p.D += lrr*outer(p.v - q.v, m - y)
        end
        p.D /= area(p)
        p.D = 0.5*(p.D + transpose(p.D))
    end
end

# S = viscous stress
function getS(p::VoronoiPolygon, dr::Float64)::RealMatrix 
    divv = dot(p.D, MAT1)
    mu = p.mu
    if divv < 0.0
        mu -= divv*p.rho*dr^2
    end
    return 2.0*mu*(p.D - divv*MAT1/3)
end

"""
    viscous_step!(grid::VoronoiGrid, dt::Float64; artificial_viscosity::Bool = true)

Applies one forward viscous step of size `dt` to all Voronoi polygons.
It assumes that the velocity deformation tensor is already computed using the `find_D!` function.
It is also assumed that every polygon `p` has its dynamic viscosity `p.mu` assigned by the initial condition or otherwise.
A keyword parameter controls whether Stone-Norman viscosity tensor is used to damp oscillations near shocks (yes by default).

Required variables: `v`, `D`.
"""
function viscous_step!(grid::VoronoiGrid, dt::Float64; artificial_viscosity::Bool = true)
    avdr = artificial_viscosity ? grid.dr : 0.0
    @batch for p in grid.polygons
        for (q,e,y) in neighbors(p, grid)
            m = midpoint(e)
            p.v -= dt*lr_ratio(p.x-y,e)/p.mass*(getS(p, avdr) - getS(q, avdr))*(m - p.x)
        end
    end
    @batch for p in grid.polygons
        for (q,e,y) in neighbors(p, grid)
            m = midpoint(e)
            p.e += dt*lr_ratio(p.x-y,e)/p.mass*(dot(m - p.x, getS(p, avdr)*p.v) - dot(m - y, getS(q, avdr)*q.v))
        end
    end
end