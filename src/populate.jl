function _everywhere(::RealVector)::Bool
    return true
end

function _donothing(::VoronoiPolygon)
    return
end

"""
    populate_circ!(grid::VoronoiGrid{T}; charfun, center, ic!)

Populate computational domain with polygons arranged in concentric circles. 
Keyword parameters:
* `charfun`: the characteristic function; only those areas where `charfun(x) = true` are populated
* `center`: the center of each circle
* `ic!`: the initial condition; `ic!(p)` is called on every polygon *after* the mesh is generated
"""
function populate_circ!(grid::VoronoiGrid{T}; charfun::Function = _everywhere, center::RealVector = VEC0, ic!::Function = _donothing) where T <: VoronoiPolygon
    r_max = maximum([norm(x - center) for x in verts(grid.boundary_rect)])
    for r in (0.5*grid.dr):grid.dr:r_max
        k_max = round(Int, 2.0*pi*r/grid.dr)
        for k in 1:k_max
            theta = 2.0*pi*k/k_max
            x = center + RealVector(r*cos(theta), r*sin(theta))
            if charfun(x) && isinside(grid.boundary_rect, x)
                push!(grid.polygons, T(x=x))
            end
        end
    end
    remesh!(grid)
    @batch for p in grid.polygons
        ic!(p)
    end
	return
end

"""
    populate_rect!(grid::VoronoiGrid{T}; charfun, ic!)

Populate computational domain with Cartesian grid.
Keyword parameters:
* `charfun`: the characteristic function; only those areas where `charfun(x) = true` are populated
* `ic!`: the initial condition; `ic!(p)` is called on every polygon *after* the mesh is generated
"""
function populate_rect!(grid::VoronoiGrid{T}; charfun::Function = _everywhere, ic!::Function = _donothing) where T <: VoronoiPolygon
    x1_max = grid.boundary_rect.xmax[1]
    x2_max = grid.boundary_rect.xmax[2]
    x1_min = grid.boundary_rect.xmin[1]
    x2_min = grid.boundary_rect.xmin[2]
    N = round(Int, (x1_max - x1_min)/grid.dr)
    M = round(Int, (x2_max - x2_min)/grid.dr)
    for x1 in range(x1_min, x1_max, N)
        for x2 in range(x2_min, x2_max, M)
            x = RealVector(x1, x2) + 0.5*RealVector(grid.dr, grid.dr)
            if charfun(x) && isinside(grid.boundary_rect, x)
                push!(grid.polygons, T(x=x))
            end
        end
    end
    remesh!(grid)
    @batch for p in grid.polygons
        ic!(p)
    end
	return
end

"""
    populate_rand!(grid::VoronoiGrid{T}; charfun, ic!)

Populate computational domain with polygons arranged randomly. This initializing method is not recommended because the mesh will have very low quality.
Keyword parameters:
* `charfun`: the characteristic function; only those areas where `charfun(x) = true` are populated
* `ic!`: the initial condition; `ic!(p)` is called on every polygon *after* the mesh is generated
"""
function populate_rand!(grid::VoronoiGrid{T}; charfun::Function = _everywhere, ic!::Function = _donothing) where T <: VoronoiPolygon
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
            push!(grid.polygons, T(x=x))
        end
    end
    remesh!(grid)
    @batch for p in grid.polygons
        ic!(p)
    end
	return
end

"""
    populate_vogel!(grid::VoronoiGrid{T}; charfun, center, ic!)

Populate computational domain with polygons arranged on Vogel spiral.
Keyword parameters:
* `charfun`: the characteristic function; only those areas where `charfun(x) = true` are populated
* `center`: the center of the spiral
* `ic!`: the initial condition; `ic!(p)` is called on every polygon *after* the mesh is generated
"""
function populate_vogel!(grid::VoronoiGrid{T}; charfun::Function = _everywhere, center::RealVector = VEC0, ic!::Function = _donothing) where T <: VoronoiPolygon
    r_max = maximum([norm(x - center) for x in verts(grid.boundary_rect)])
    N = round(Int, pi*r_max*r_max/(grid.dr^2))
    for i in 1:N
        r = r_max*sqrt(i/N)
        theta = 2.39996322972865332*i
        x = RealVector(center[1] + r*cos(theta), center[2] + r*sin(theta))
        if charfun(x) && isinside(grid.boundary_rect, x)
            push!(grid.polygons, T(x=x))
        end
    end
    remesh!(grid)
    @batch for p in grid.polygons
        ic!(p)
    end
	return
end

"""
    populate_lloyd!(grid::VoronoiGrid{T}; charfun, niterations, ic!)

Populate computational domain with polygons arranged in concentric circles. 
Keyword parameters:
* `charfun`: the characteristic function; only those areas where `charfun(x) = true` are populated
* `center`: the center of each circle
* `ic!`: the initial condition; `ic!(p)` is called on every polygon *after* the mesh is generated
"""
function populate_lloyd!(grid::VoronoiGrid{T}; charfun::Function = _everywhere, niterations::Int = 100, ic!::Function = _donothing) where T <: VoronoiPolygon
    populate_rand!(grid, charfun=charfun)
    for _ in 1:niterations
        remesh!(grid)
        @batch for p in grid.polygons
            p.x = centroid(p)
        end
    end
    remesh!(grid)
    @batch for p in grid.polygons
        ic!(p)
    end
	return
end

"""
    populate_hex!(grid::VoronoiGrid{T}; charfun, ic!)

Populate computational domain with polygons arranged in hexagonal grid. Use this initializing method when you are not sure.
Keyword parameters:
* `charfun`: the characteristic function; only those areas where `charfun(x) = true` are populated
* `ic!`: the initial condition; `ic!(p)` is called on every polygon *after* the mesh is generated
"""
function populate_hex!(grid::VoronoiGrid{T}; charfun::Function = _everywhere, ic!::Function = _donothing) where T <: VoronoiPolygon
    a = (4/3)^(1/4)*grid.dr
    b = (3/4)^(1/4)*grid.dr
    x1_max = grid.boundary_rect.xmax[1]
    x2_max = grid.boundary_rect.xmax[2]
    x1_min = grid.boundary_rect.xmin[1]
    x2_min = grid.boundary_rect.xmin[2]
    i_min = Int64(floor(x1_min/a)) - 1
    j_min = Int64(floor(x2_min/b))
    i_max = Int64(ceil(x1_max/a))
    j_max = Int64(ceil(x2_max/b))
	for i in i_min:i_max, j in j_min:j_max
        x1 = (i + (j%2)/2)*a
        x2 = j*b
        x = RealVector(x1, x2)
        if charfun(x) && isinside(grid.boundary_rect, x)
            push!(grid.polygons, T(x=x))
        end
	end
    remesh!(grid)
    @batch for p in grid.polygons
        ic!(p)
    end
	return
end

