function apply_unary!(grid::VoronoiGrid, fun::Function)
    if !hasmethod(fun, (VoronoiPolygon, ))
        throw(ArgumentError("functional argument must be fun(::VoronoiPolygon)"))
    end
    @batch for p in grid.polygons
        fun(p)
    end
end

function apply_binary!(grid::VoronoiGrid, fun::Function)
    if !hasmethod(fun, (VoronoiPolygon, VoronoiPolygon, Edge))
        throw(ArgumentError("functional argument must be fun(::VoronoiPolygon, ::VoronoiPolygon, ::Edge)"))
    end    
    @batch for p in grid.polygons
        for e in p.edges
            if isboundary(e)
                continue
            end
            q = grid.polygons[e.label]
            fun(p, q, e)
        end
    end
end

function apply_local!(grid::VoronoiGrid, fun::Function, threshold_dist::Float64)
    if !hasmethod(fun, (VoronoiPolygon, VoronoiPolygon, Float64))
        throw(ArgumentError("functional argument must be fun(::VoronoiPolygon, ::VoronoiPolygon, ::Float64)"))
    end 
    @batch for p in grid.polygons
        x = p.x
        key0 = findkey(grid.cell_list, x)
        for node in grid.cell_list.magic_path
            rr = node.rr
            offset = node.key
            if (rr > threshold_dist^2)
                break
            end
            key = key0 + offset
            if !(checkbounds(Bool, grid.cell_list.cells, key))
                continue
            end
            for i in grid.cell_list.cells[key]
                q = grid.polygons[i]
                if (p.x == q.x)
                   continue
                end
                r = norm(p.x - q.x)
                if r < threshold_dist
                    fun(p, q, r)
                end
            end
        end
    end
end

