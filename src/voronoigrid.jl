mutable struct VoronoiGrid{T}
    dr::Float64
    h::Float64
    rr_max::Float64
    boundary_rect::Rectangle
    extended_rect::Rectangle
    cell_list::CellList
    polygons::Vector{T}
    index_containers::Vector{Vector{Int}}
    xperiod::Bool
    yperiod::Bool
    VoronoiGrid{T}(boundary_rect::Rectangle, dr::Float64;
    xperiod::Bool = false, yperiod::Bool = false) where T = begin
        h = 2dr
        r_max = 10dr
        extended_xmin = boundary_rect.xmin - r_max*xperiod*VECX - r_max*yperiod*VECY
        extended_xmax = boundary_rect.xmax + r_max*xperiod*VECX + r_max*yperiod*VECY
        extended_rect = Rectangle(extended_xmin, extended_xmax)
        cell_list = CellList(h, extended_rect)
        polygons = T[]
        index_containers = [PreAllocVector(Int, POLYGON_SIZEHINT) for _ in 1:Threads.nthreads()]
        
        return new{T}(
            dr,
            h,
            100*dr^2,
            boundary_rect, 
            extended_rect,
            cell_list, 
            polygons, 
            index_containers,
            xperiod,
            yperiod
        )
    end
end

@inbounds function voronoicut!(grid::VoronoiGrid{T}, poly::T) where T <: VoronoiPolygon
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
            throw("The Voronoi Mesh has been destroyed.")
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
    for cell in grid.cell_list.cells
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

function _getNeighbors(grid::VoronoiGrid{T}, poly::T) where T <: VoronoiPolygon
    container = grid.index_containers[Threads.threadid()]
    empty!(container)
    for e in poly.edges
        if !isboundary(e)
            push!(container, e.label)
        end
    end
    return container
end

function nearest_polygon(grid::VoronoiGrid{T}, x::RealVector)::T where T <: VoronoiPolygon
    key0 = findkey(grid.cell_list, x)
    rr_best = Inf
    i_best = 0
    for node in grid.cell_list.magic_path
        rr_min = node.rr
        offset = node.key
        if (rr_min > rr_best)
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