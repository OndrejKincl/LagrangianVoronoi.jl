using LinearAlgebra

function get_mass!(p::VoronoiPolygon)
    p.mass = p.rho*area(p)
end

function populate_circ!(grid::VoronoiGrid{T}; charfun = (::RealVector -> true), center::RealVector = VEC0) where T
    r_max = maximum([norm(x - center) for x in verts(grid.boundary_rect)])
    for r in (0.5*grid.dr):grid.dr:r_max
        k_max = round(Int, 2.0*pi*r/grid.dr)
        for k in 1:k_max
            theta = 2.0*pi*k/k_max
            x = center + RealVector(r*cos(theta), r*sin(theta))
            if charfun(x) && isinside(grid.boundary_rect, x)
                push!(grid.polygons, VoronoiPolygon{T}(x))
            end
        end
    end
    remesh!(grid)
	return
end

function populate_rect!(grid::VoronoiGrid{T}; charfun = (::RealVector -> true)) where T
    x1_max = grid.boundary_rect.xmax[1]
    x2_max = grid.boundary_rect.xmax[2]
    x1_min = grid.boundary_rect.xmin[1]
    x2_min = grid.boundary_rect.xmin[2]
    N = round(Int, (x1_max - x1_min)/grid.dr)
    M = round(Int, (x2_max - x2_min)/grid.dr)
    for x1 in range(x1_min, x1_max, N)
        for x2 in range(x2_min, x2_max, M)
            x = RealVector(x1, x2)
            if charfun(x) && isinside(grid.boundary_rect, x)
                push!(grid.polygons, VoronoiPolygon{T}(x))
            end
        end
    end
    remesh!(grid)
	return
end

function populate_rand!(grid::VoronoiGrid{T}; charfun = (::RealVector -> true)) where T
    x1_max = grid.boundary_rect.xmax[1]
    x2_max = grid.boundary_rect.xmax[2]
    x1_min = grid.boundary_rect.xmin[1]
    x2_min = grid.boundary_rect.xmin[2]
    N = round(Int, abs(x1_max - x1_min)*abs(x2_max - x2_min)/(grid.dr^2))
    for _ in 1:N
        s1 = rand()
        s2 = rand()
        x = (s1*x1_max + (1-s1)*x1_min)*VECX + (s2*x2_max + (1-s2)*x2_min)*VECY
        if charfun(x) && isinside(grid.boundary_rect, x)
            push!(grid.polygons, VoronoiPolygon{T}(x))
        end
    end
    remesh!(grid)
	return
end

function populate_vogel!(grid::VoronoiGrid{T}; charfun = (::RealVector -> true), center::RealVector = VEC0) where T
    r_max = maximum([norm(x - center) for x in verts(grid.boundary_rect)])
    N = round(Int, pi*r_max*r_max/(grid.dr^2))
    for i in 1:N
        r = r_max*sqrt(i/N)
        theta = 2.39996322972865332*i
        x = RealVector(center[1] + r*cos(theta), center[2] + r*sin(theta))
        if charfun(x) && isinside(grid.boundary_rect, x)
            push!(grid.polygons, VoronoiPolygon{T}(x))
        end
    end
    remesh!(grid)
	return
end

function populate_lloyd!(grid::VoronoiGrid{T}; charfun = (::RealVector -> true), niterations = 10) where T
    populate_rand!(grid, charfun=charfun)
    remesh!(grid)
    cs = zeros(RealVector, length(grid.polygons))
    for it in 1:niterations
        for (i,p) in enumerate(grid.polygons)
            cs[i] = centroid(p)
        end
        for (i,p) in enumerate(grid.polygons)
            p.x = cs[i]
        end
        remesh!(grid)
    end
	return
end

