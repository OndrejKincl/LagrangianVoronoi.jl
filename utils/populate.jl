using LinearAlgebra

function get_mass!(p::VoronoiPolygon)
    p.var.mass = p.var.rho*area(p)
end

function populate_circ!(grid::VoronoiGrid, dr::Float64; charfun = (::RealVector -> true), center::RealVector = VEC0)
    #center = 0.5*grid.boundary_rect.xmin + 0.5*grid.boundary_rect.xmax
    r_max = maximum([norm(x - center) for x in verts(grid.boundary_rect)])
    for r in (0.5*dr):dr:r_max
        k_max = round(Int, 2.0*pi*r/dr)
        for k in 1:k_max
            theta = 2.0*pi*k/k_max
            x = center + RealVector(r*cos(theta), r*sin(theta))
            if charfun(x) && isinside(grid.boundary_rect, x)
                push!(grid.polygons, VoronoiPolygon{PhysFields}(x))
            end
        end
    end
    remesh!(grid)
	return
end

function populate_rect!(grid::VoronoiGrid, dr::Float64; charfun = (::RealVector -> true))
    x1_max = grid.boundary_rect.xmax[1]# + 0.5*dr
    x2_max = grid.boundary_rect.xmax[2]# - 0.5*dr
    x1_min = grid.boundary_rect.xmin[1]# + 0.5*dr
    x2_min = grid.boundary_rect.xmin[2]# - 0.5*dr
    N = round(Int, (x1_max - x1_min)/dr)
    M = round(Int, (x2_max - x2_min)/dr)
    for x1 in range(x1_min, x1_max, N)
        for x2 in range(x2_min, x2_max, M)
            x = RealVector(x1, x2)
            if charfun(x) && isinside(grid.boundary_rect, x)
                push!(grid.polygons, VoronoiPolygon{PhysFields}(x))
            end
        end
    end
    remesh!(grid)
	return
end

function populate_rand!(grid::VoronoiGrid, dr::Float64; charfun = (::RealVector -> true))
    x1_max = grid.boundary_rect.xmax[1]
    x2_max = grid.boundary_rect.xmax[2]
    x1_min = grid.boundary_rect.xmin[1]
    x2_min = grid.boundary_rect.xmin[2]
    N = round(Int, abs(x1_max - x1_min)*abs(x2_max - x2_min)/(dr*dr))
    for _ in 1:N
        s1 = rand()
        s2 = rand()
        x = (s1*x1_max + (1-s1)*x1_min)*VECX + (s2*x2_max + (1-s2)*x2_min)*VECY
        if charfun(x)
            push!(grid.polygons, VoronoiPolygon{PhysFields}(x))
        end
    end
    remesh!(grid)
	return
end

function populate_vogel!(grid::VoronoiGrid, dr::Float64; charfun = (::RealVector -> true), center::RealVector = VEC0)
    x1_max = grid.boundary_rect.xmax[1]
    x2_max = grid.boundary_rect.xmax[2]
    x1_min = grid.boundary_rect.xmin[1]
    x2_min = grid.boundary_rect.xmin[2]
    radius = sqrt(max((x1_max - center[1])^2, (x1_min - center[1])^2) + max((x2_max - center[2])^2, (x2_min - center[2])^2)) 
    N = round(Int, pi*radius*radius/(dr*dr))
    for i in 1:N
        r = radius*sqrt(i/N)
        theta = 2.39996322972865332*i
        x = RealVector(center[1] + r*cos(theta), center[2] + r*sin(theta))
        if charfun(x)
            push!(grid.polygons, VoronoiPolygon{PhysFields}(x))
        end
    end
    remesh!(grid)
	return
end
