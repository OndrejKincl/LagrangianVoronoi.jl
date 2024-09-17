mutable struct VoronoiGrid{T}
    dr::Float64
    h::Float64
    rr_max::Float64
    boundary_rect::Rectangle
    extended_rect::Rectangle
    cell_list::CellList
    polygons::Vector{T}
    index_containers::Vector{Vector{Int}}
    xperiodic::Bool
    yperiodic::Bool
    xperiod::Float64
    yperiod::Float64
    period_tol::Float64
    VoronoiGrid{T}(boundary_rect::Rectangle, dr::Float64;
    h = 2dr, r_max = 10dr, period_tol = 10dr,
    xperiodic::Bool = false, yperiodic::Bool = false) where T = begin
        extended_xmin = boundary_rect.xmin - period_tol*xperiodic*VECX - period_tol*yperiodic*VECY
        extended_xmax = boundary_rect.xmax + period_tol*xperiodic*VECX + period_tol*yperiodic*VECY
        xperiod = boundary_rect.xmax[1] - boundary_rect.xmin[1]
        yperiod = boundary_rect.xmax[2] - boundary_rect.xmin[2]
        extended_rect = Rectangle(extended_xmin, extended_xmax)
        cell_list = CellList(h, extended_rect)
        polygons = T[]
        index_containers = [PreAllocVector(Int, POLYGON_SIZEHINT) for _ in 1:Threads.nthreads()]
        return new{T}(
            dr,
            h,
            r_max^2,
            boundary_rect, 
            extended_rect,
            cell_list, 
            polygons, 
            index_containers,
            xperiodic,
            yperiodic,
            xperiod,
            yperiod,
            period_tol,
        )
    end
end

@inbounds function voronoicut!(grid::VoronoiGrid{T}, poly::T) where T <: VoronoiPolygon
    x = poly.x
    prr = influence_rr(poly)
    key0 = findkey(grid.cell_list, x)
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
            q = grid.polygons[i]
            y = x + get_arrow(q.x, x, grid)
            if (x == y) || (norm_squared(x-y) > prr)
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
    # reset voronoi cells and add them to the cell list
    @batch for i in eachindex(grid.polygons)
        poly = grid.polygons[i]
        reset!(poly, grid.extended_rect)
        insert_periodic!(grid, poly.x, i)
    end
    # cut all voronoi cells
    @batch for i in eachindex(grid.polygons)
        voronoicut!(grid, grid.polygons[i])
        sort_edges!(grid.polygons[i])
    end
    return
end

# get shortest vector from y to x
# is equal to x - y on non-periodic domains
function get_arrow(x::RealVector, y::RealVector, grid::VoronoiGrid)
    v = x - y
    if grid.xperiodic && (abs(v[1]) > 0.5*grid.xperiod)
        v -= sign(v[1])*grid.xperiod*VECX
    end
    if grid.yperiodic && (abs(v[2]) > 0.5*grid.yperiod)
        v -= sign(v[2])*grid.yperiod*VECY
    end
    return v
end

function insert_periodic!(grid::VoronoiGrid, x::RealVector, label::Int)
    cl = grid.cell_list
    insert!(cl, x, label)
    if grid.xperiodic
        insert!(cl, x + grid.xperiod*VECX, label)
        insert!(cl, x - grid.xperiod*VECX, label)
    end
    if grid.yperiodic
        insert!(cl, x + grid.yperiod*VECY, label)
        insert!(cl, x - grid.yperiod*VECY, label)
    end
    if grid.xperiodic && grid.yperiodic
        insert!(cl, x + grid.xperiod*VECX + grid.yperiod*VECY, label)
        insert!(cl, x + grid.xperiod*VECX - grid.yperiod*VECY, label)
        insert!(cl, x - grid.xperiod*VECX + grid.yperiod*VECY, label)
        insert!(cl, x - grid.xperiod*VECX - grid.yperiod*VECY, label)
    end
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

function periodic_proj(grid::VoronoiGrid, Z::RealVector)::RealVector
    Zx = Z[1]
    Zy = Z[2]
    if grid.xperiodic
        xperiod = abs(grid.boundary_rect.xmax[1] - grid.boundary_rect.xmin[1])
        Zx = (Z[1] - grid.boundary_rect.xmin[1])%xperiod
        Zx = (Zx + xperiod)%xperiod + grid.boundary_rect.xmin[1]
    end
    if grid.yperiodic   
        yperiod = abs(grid.boundary_rect.xmax[2] - grid.boundary_rect.xmin[2])
        Zy = (Z[2] - grid.boundary_rect.xmin[2])%yperiod
        Zy = (Zy + yperiod)%yperiod + grid.boundary_rect.xmin[2]
    end
    return Zx*VECX + Zy*VECY
end