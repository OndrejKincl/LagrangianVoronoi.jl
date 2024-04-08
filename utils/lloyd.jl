function tri_area(a::RealVector, b::RealVector, c::RealVector)::Float64
    return 0.5*abs(LagrangianVoronoi.cross2(b - a, c - a))
end

function centroid(p::VoronoiPolygon)::RealVector
    A = 0.0
    c = VEC0
    for e in p.edges
        dA = tri_area(p.x, e.v1, e.v2)
        A += dA
        c += dA*(p.x + e.v1 + e.v2)/3
    end
    return c/A
end

function lloyd_relaxation!(p::VoronoiPolygon, tau_r::Float64)
    p.var.lloyd_dx = VEC0
    if !isboundary(p)
        c = centroid(p)
        p.var.lloyd_dx = dt/(tau_r + dt)*(c - p.x)
    end
end

function lloyd_correction!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    p.var.lloyd_dv += lr_ratio(p,q,e)*dot(p.x - m, p.var.lloyd_dx)*(p.var.v - q.var.v)  
end

function lloyd_update!(p::VoronoiPolygon)
    p.var.v += p.var.lloyd_dv/area(p)
    p.var.lloyd_dv = VEC0
    p.x += p.var.lloyd_dx
end

function lloyd_stabilization!(grid::VoronoiGrid, tau_r::Float64)
    apply_unary!(grid, p -> lloyd_relaxation!(p, tau_r))
    apply_binary!(grid, lloyd_correction!)
    apply_unary!(grid, lloyd_update!)
    remesh!(grid)
end


function populate_lloyd!(grid::VoronoiGrid, dr::Float64, charfun = (::RealVector -> true); niterations::Int = 10, randomness::Float64 = 0.4)
    x1_max = grid.boundary_rect.xmax[1]
    x2_max = grid.boundary_rect.xmax[2]
    x1_min = grid.boundary_rect.xmin[1]
    x2_min = grid.boundary_rect.xmin[2]
    # place points
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
    #randomly push everything
    for p in grid.polygons
        if !isboundary(p)
            p.x += randomness*dr*((rand() - 0.5)*VECX + (rand() - 0.5)*VECY)
        end
    end

    remesh!(grid)
    # perform lloyd iterations
    for _ in 1:niterations
        lloyd_stabilization!(grid, 0.0)
        remesh!(grid)
    end
	return
end