"""
    VoronoiGrid{T}(boundary_rect::Rectangle, dr::Float64, kwargs...)

This struct contains all information about geometry, the Voronoi mesh and a cell list. 
Type variable `T` specifies the Voronoi Polygon, `boundary_rect` defines the computational domain and `dr` is the default resolution 
(a particle will typically occupy an area of size `dr^2`). 

Keyword arguments:
* `xperiodic::Bool`: Is our domain periodic in the horizontal direction?
* `yperiodic::Bool`: Is our domain periodic in the vertical direction?
* `h` the size of cells in the cell_list
* `r_max` the maximum possible distance between neighboring cells
"""
mutable struct VoronoiGrid{T}
    dr::Float64
    h::Float64
    rr_max::Float64
    boundary_rect::Rectangle
    extended_rect::Rectangle
    cell_list::CellList
    polygons::Vector{T}
    xperiodic::Bool
    yperiodic::Bool
    xperiod::Float64
    yperiod::Float64
    VoronoiGrid{T}(boundary_rect::Rectangle, dr::Float64;
    h = 2dr, r_max = 10dr,
    xperiodic::Bool = false, yperiodic::Bool = false) where T = begin
        extended_xmin = boundary_rect.xmin - r_max*xperiodic*VECX - r_max*yperiodic*VECY
        extended_xmax = boundary_rect.xmax + r_max*xperiodic*VECX + r_max*yperiodic*VECY
        xperiod = boundary_rect.xmax[1] - boundary_rect.xmin[1]
        yperiod = boundary_rect.xmax[2] - boundary_rect.xmin[2]
        extended_rect = Rectangle(extended_xmin, extended_xmax)
        cell_list = CellList(h, extended_rect)
        polygons = T[]
        return new{T}(
            dr,
            h,
            r_max^2,
            boundary_rect, 
            extended_rect,
            cell_list, 
            polygons,
            xperiodic,
            yperiodic,
            xperiod,
            yperiod
        )
    end
end

# crop a polygon `poly` with respect to all other seed generators to create the Voronoi cell.
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

"""
    remesh!(grid::VoronoiGrid)

Generate all Voronoi cells by performing all Voronoi cuts. This is called automatically when the VoronoiGrid
is populated and each the Voronoi generators move in space.
"""
 function remesh!(grid::VoronoiGrid)
    # clear the cell list
    @batch for cell in grid.cell_list.cells
        empty!(cell)
    end
    # reset voronoi cells and add them to the cell list
    @inbounds begin
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
    end
    return
end

"""
    get_arrow(x::RealVector, y::RealVector, grid::VoronoiGrid)::RealVector

Return a vector from `y` to `x`. Equal to `x - y` on non-periodic domains. 
For periodic rectangle, the shortest possible vector is returned.
"""
function get_arrow(x::RealVector, y::RealVector, grid::VoronoiGrid)::RealVector
    v = x - y
    if grid.xperiodic && (abs(v[1]) > 0.5*grid.xperiod)
        v -= sign(v[1])*grid.xperiod*VECX
    end
    if grid.yperiodic && (abs(v[2]) > 0.5*grid.yperiod)
        v -= sign(v[2])*grid.yperiod*VECY
    end
    return v
end

# Insert a `label` at `x` to the cell list of `grid`. 
# If `x` is near boundary, multiple copies need to be inserted, so that 
# other polygons find the label in their neighborhood. 
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

"""
    nearest_polygon(grid::VoronoiGrid{T}, x::RealVector)::T

Find the nearest polygon to the reference point `x`. 
"""
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

# Find representant of Z in the periodic equivalence class such that it lies within 
# the domain boundaries.
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