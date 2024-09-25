import Base:iterate


struct PolygonNeighborIterator{T}
    p::T
    grid::VoronoiGrid{T}
end

"""
    neighbors(p::VoronoiPolygon, grid::VoronoiGrid)

Create an iterator through all Voronoi polygons that neighbor the polygon `p`. 
Use it in a for loop to iterate through all triplets `(q,e,y)` where 

* `q` = the neighboring polygon
* `e` = the edge connecting `p` and `q`
* `y` = the position of `q` as a neighbor of `p` (equivalent to `q.x` for non-periodic grids)
"""
function neighbors(p::T, grid::VoronoiGrid{T})::PolygonNeighborIterator{T} where T <: VoronoiPolygon
    return PolygonNeighborIterator{T}(p, grid)
end

function Base.iterate(itr::PolygonNeighborIterator, i::Int = 0)::Union{Nothing, Tuple{Tuple{VoronoiPolygon, Edge, RealVector}, Int}}
    while (i < length(itr.p.edges))
        i += 1
        e = itr.p.edges[i]
        if isboundary(e) continue end
        q = itr.grid.polygons[e.label]
        y = itr.p.x + get_arrow(q.x, itr.p.x, itr.grid)
        return ((q, e, y), i)
    end
    return nothing
end

# iterator through all boundary edges
struct PolygonBoundaryIterator{T}
    p::T
end

"""
    boundaries(p::VoronoiPolygon)

Create an iterator through all edges of 'p' that lie on the boundary of the domain. 
This can be used to implement boundary forces.
"""
function boundaries(p::T)::PolygonBoundaryIterator{T} where T <: VoronoiPolygon
    return PolygonBoundaryIterator{T}(p)
end

function Base.iterate(itr::PolygonBoundaryIterator, i::Int = 0)::Union{Nothing, Tuple{Edge, Int}}
    while (i < length(itr.p.edges))
        i += 1
        e = itr.p.edges[i]
        if !isboundary(e) continue end
        return (e, i)
    end
    return nothing
end