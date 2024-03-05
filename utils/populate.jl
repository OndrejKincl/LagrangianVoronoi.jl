using LinearAlgebra

function get_mass!(p::VoronoiPolygon)
    p.var.mass = rho*area(p)
end

function populate_circ!(grid::VoronoiGrid, dr::Float64; charfun = (::RealVector -> true), center = VEC0)
    r_max = maximum([norm(x - center) for x in verts(grid.boundary_rect)])
    for r in (0.5*dr):dr:r_max
        k_max = round(Int, 2.0*pi*r*N)
        for k in 1:k_max
            theta = 2.0*pi*k/k_max
            x = center + RealVector(r*cos(theta), r*sin(theta))
            if charfun(x) && isinside(grid.boundary_rect, x)
                push!(grid.polygons, VoronoiPolygon{PhysFields}(x))
            end
        end
    end
    remesh!(grid)
    apply_unary!(grid, get_mass!)
	return
end