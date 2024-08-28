struct Cutter
    x::RealVector
    label::Int
end

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
    cutters::FastVector{Cutter}
    period_tol::Float64
    VoronoiGrid{T}(boundary_rect::Rectangle, dr::Float64;
    h = 2dr, r_max = 10dr, period_tol = 10dr,
    xperiodic::Bool = false, yperiodic::Bool = false) where T = begin
        extended_xmin = boundary_rect.xmin - period_tol*xperiodic*VECX - period_tol*yperiodic*VECY
        extended_xmax = boundary_rect.xmax + period_tol*xperiodic*VECX + period_tol*yperiodic*VECY
        xperiod = boundary_rect.xmax[1] - boundary_rect.xmin[1]
        yperiod = boundary_rect.xmax[2] - boundary_rect.xmin[2]
        extended_rect = Rectangle(extended_xmin, extended_xmax)
        n = round(Int, area(extended_rect)/(dr^2))
        cutters = FastVector{Cutter}(n)
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
            cutters,
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
            y = grid.cutters[i].x
            label = grid.cutters[i].label
            if (poly.x == y) || (norm_squared(poly.x-y) > prr)
                continue
            end
            if voronoicut!(poly, y, label)
                prr = influence_rr(poly)
            end
        end
    end
end

function make_cutter!(grid::VoronoiGrid, x::RealVector, label::Int)
    c = Cutter(x, label)
    push!(grid.cutters, c)
    j = length(grid.cutters)
    insert!(grid.cell_list, c.x, j)
end

@inbounds function remesh!(grid::VoronoiGrid)::Nothing
    # empty the cutter list
    empty!(grid.cutters)
    # clear the cell list
    for cell in grid.cell_list.cells
        empty!(cell)
    end
    # reset voronoi cells, generate cutters and add them to the cell list
    for i in eachindex(grid.polygons)
        poly = grid.polygons[i]
        reset!(poly, grid.extended_rect)
        make_cutter!(grid, poly.x, i)
    end
    # edge periodic maps
    if grid.xperiodic
        N = length(grid.cutters)
        for i in 1:N
            c = grid.cutters[i]
            if (grid.boundary_rect.xmax[1] - grid.period_tol < c.x[1])
                make_cutter!(grid, c.x - grid.xperiod*VECX, c.label)
            end
            if (grid.boundary_rect.xmin[1] + grid.period_tol > c.x[1])
                make_cutter!(grid, c.x + grid.xperiod*VECX, c.label)
            end 
        end
    end
    if grid.yperiodic
        N = length(grid.cutters)
        for i in 1:N
            c = grid.cutters[i]
            if (grid.boundary_rect.xmax[2] - grid.period_tol < c.x[2])
                make_cutter!(grid, c.x - grid.yperiod*VECY, c.label)
            end
            if (grid.boundary_rect.xmin[2] + grid.period_tol > c.x[2])
                make_cutter!(grid, c.x + grid.yperiod*VECY, c.label)
            end 
        end
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