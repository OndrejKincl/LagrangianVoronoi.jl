function phase_centroid(p::VoronoiPolygon, grid::VoronoiGrid, h::Float64)::RealVector
    Cx = VEC0
    C = 0.0
    for e in p.edges
        dA = tri_area(p.x, e.v1, e.v2)
        for y in (
            2/3*p.x + 1/6*e.v1 + 1/6*e.v2, 
            1/6*p.x + 2/3*e.v1 + 1/6*e.v2,
            1/6*p.x + 1/6*e.v1 + 2/3*e.v2
            )
            w = (dA/3)*findcolor(grid, y, p.phase, h)
            Cx += w*y
            C += w
        end
    end
    return Cx/C
end

function findcolor(grid::VoronoiGrid, x::RealVector, phase::Int, h::Float64)
    C = 0.0
    S = 0.0
    key0 = findkey(grid.cell_list, x)
    for node in grid.cell_list.magic_path
        if (node.rr > h)
            break
        end
        key = key0 + node.key
        if !(checkbounds(Bool, grid.cell_list.cells, key))
            continue
        end
        for i in grid.cell_list.cells[key]
            q = grid.polygons[i]
            y = x + get_arrow(q.x, x, grid)
            r = norm(x-y)
            if r < h
                ker = wendland2(h, r)
                qA = q.mass/q.rho
                C += (q.phase == phase)*ker*qA
                S += ker*qA
            end
        end
    end
    return C/S
end

function phase_preserving_remapping!(grid::VoronoiGrid, dt::Float64, h::Float64; quality_treshold::Float64 = 0.3)
    if mesh_quality(grid) < quality_treshold
        @batch for p in grid.polygons
            c = phase_centroid(p, grid, h)
            p.dv = (c - p.x)/dt
        end
        relaxation_step!(grid, dt)
    end
end

function mesh_quality(grid::VoronoiGrid)::Float64
    @batch for p in grid.polygons
        rmax = 0.0
        rmin = Inf
        for (_,_,y) in neighbors(p, grid)
            r = norm(p.x-y)
            rmax = max(rmax, r)
            rmin = min(rmin, r)
        end
        p.quality = rmin/rmax
    end
    return minimum(p::VoronoiPolygon -> p.quality, grid.polygons)
end

function st_tensor(p::VoronoiPolygon)::RealMatrix
    cg = norm(p.cgrad)
    n = p.cgrad/(cg + eps())
    return p.st*cg*(MAT1 - outer(n, n))
end

function surface_tension!(grid::VoronoiGrid, dt::Float64, h::Float64)
    # compute the gradient of coloring function
    @batch for p in grid.polygons
        p.cgrad = VEC0
        key0 = findkey(grid.cell_list, p.x)
        for node in grid.cell_list.magic_path
            if (node.rr > h)
                break
            end
            key = key0 + node.key
            if !(checkbounds(Bool, grid.cell_list.cells, key))
                continue
            end
            for i in grid.cell_list.cells[key]
                q = grid.polygons[i]
                y = p.x + get_arrow(q.x, p.x, grid)
                r = norm(p.x-y)
                if r < h && q.phase == 0
                    qA = q.mass/q.rho
                    ker = rDwendland2(h, r)
                    p.cgrad += qA*ker*(p.x - y)
                end
            end
        end
    end
    @batch for p in grid.polygons
        for (q,e,y) in neighbors(p, grid)
            m = midpoint(e)
            p.v -= dt*lr_ratio(p.x-y,e)/p.mass*(st_tensor(p) - st_tensor(q))*(m - p.x)
        end
    end
end

function isinterface(p::VoronoiPolygon, grid::VoronoiGrid)::Bool
    for (q,_,_) in neighbors(p, grid)
        if (q.phase != p.phase) return true end
    end
    return false
end