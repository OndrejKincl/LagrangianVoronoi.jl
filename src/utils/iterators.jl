import Base:iterate

# iterator through all neighboring cells
struct PolygonNeighborIterator{T}
    p::T
    grid::VoronoiGrid{T}
end

function neighbors(p::T, grid::VoronoiGrid{T})::PolygonNeighborIterator{T} where T <: VoronoiPolygon
    return PolygonNeighborIterator{T}(p, grid)
end

function Base.iterate(itr::PolygonNeighborIterator, i::Int = 0)::Union{Nothing, Tuple{Tuple{VoronoiPolygon, Edge}, Int}}
    while (i < length(itr.p.edges))
        i += 1
        e = itr.p.edges[i]
        if isboundary(e) continue end
        q = itr.grid.polygons[e.label]
        return ((q, e), i)
    end
    return nothing
end

# iterator through all boundary edges
struct PolygonBoundaryIterator{T}
    p::T
end

function boundaries(p::T) where T <: VoronoiPolygon
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