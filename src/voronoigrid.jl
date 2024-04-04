mutable struct VoronoiGrid{T}
    h::Float64
    boundary_rect::Rectangle
    cell_list::CellList
    polygons::Vector{VoronoiPolygon{T}}
    index_containers::Vector{PreAllocVector{Int}}
    rr_max::Float64
    VoronoiGrid{T}(h::Number, boundary_rect::Rectangle) where T = begin
        cell_list = CellList(h, boundary_rect)
        polygons = VoronoiPolygon{T}[]
        index_containers = [PreAllocVector{Int}(POLYGON_SIZEHINT) for _ in 1:Threads.nthreads()]
        return new{T}(
            h, 
            boundary_rect, 
            cell_list, 
            polygons, 
            index_containers,
            h
        )
    end
end 



# cut polygon poly using all other polygon seeds as cutting tools
@inbounds function voronoicut!(grid::VoronoiGrid, poly::VoronoiPolygon)
    poly.isbroken = false
    x = poly.x
    prr = influence_rr(poly)
    key0 = findkey(grid.cell_list, x)
    neighbors = _getNeighbors(grid, poly)
    for node in grid.cell_list.magic_path
        rr = node.rr
        offset = node.key
        if (rr > prr)
            break
        end
        if (rr > grid.rr_max)
            poly.isbroken = true
            break
        end
        key = key0 + offset
        if !(checkbounds(Bool, grid.cell_list.cells, key))
            continue
        end
        for i in grid.cell_list.cells[key]
            if i in neighbors
                continue
            end
            y = grid.polygons[i].x
            if (poly.x == y) || (norm_squared(poly.x-y) > prr)
                continue
            end
            if voronoicut!(poly, y, i)
                prr = influence_rr(poly)
            end
        end
    end
end

@inbounds function remesh!(grid::VoronoiGrid)::Nothing
    # clear the cell list
    @batch for cell in grid.cell_list.cells
        empty!(cell)
    end
    # reset voronoi cells
    @batch for i in eachindex(grid.polygons)
        poly = grid.polygons[i]
        # remember old neighbors
        neighbors = _getNeighbors(grid, poly)
        reset!(poly, grid.boundary_rect)
        # cut against old neighbors
        for j in neighbors
            if checkbounds(Bool, grid.polygons, j)
                voronoicut!(poly, grid.polygons[j].x, j)
            end
        end
        insert!(grid.cell_list, poly.x, i)
    end
    # cut all voronoi cells
    @batch for i in eachindex(grid.polygons)
        voronoicut!(grid, grid.polygons[i])
        sort_edges!(grid.polygons[i])
    end
    return
end

function _getNeighbors(grid::VoronoiGrid, poly::VoronoiPolygon)
    container = grid.index_containers[Threads.threadid()]
    empty!(container)
    for e in poly.edges
        if !isboundary(e)
            push!(container, e.label)
        end
    end
    return container
end

function limit_cell_diameter!(grid::VoronoiGrid, diameter::Float64)
    grid.rr_max = diameter^2
end

function nearest_polygon(grid::VoronoiGrid, x::RealVector)::VoronoiPolygon
    key0 = findkey(grid.cell_list, x)
    rr_best = Inf
    i_best = 0
    for node in grid.cell_list.magic_path
        rr_min = node.rr
        offset = node.key
        if (rr_min > rr_best) || (rr_min > grid.rr_max)
            break
        end
        key = key0 + offset
        if !(checkbounds(Bool, grid.cell_list.cells, key))
            continue
        end
        for i in grid.cell_list.cells[key]
            p = grid.polygons[i]
            prr = norm_squared(p.x - x)
            if prr < rr_best
                rr_best = prr
                i_best = i
            end
        end
    end
    return grid.polygons[i_best]
end

function point_value(grid::VoronoiGrid, x::RealVector, fun::Function)
    p = nearest_polygon(grid, x)
    L = ls_reconstruction(LinearExpansion, grid, p, fun) 
    return fun(p) + dot(L, x - p.x)
end