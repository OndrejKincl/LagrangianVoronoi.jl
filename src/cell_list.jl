const CELL_SIZEHINT = 8
const Cell = Vector{Int}
const Key = CartesianIndex{2}
const KEYNULL = Key(0, 0)

struct PathNode
    key::Key
    rr::Float64
end

mutable struct CellList
    cells::Array{Cell, 2}
    locks::Array{ReentrantLock, 2}
    # values for calculating location keys
	origin::RealVector
    h::Float64
    magic_path::Vector{PathNode}
    CellList(h::Float64, boundary_rect::Rectangle) = begin
        if (h <= 0.0)
            throw(ArgumentError("h must be positive"))
        end
        # make to box slightly bigger to avoid problems on boundary
        origin = boundary_rect.xmin - RealVector(h,h)
        n1 = floor(Int, (boundary_rect.xmax[1] - boundary_rect.xmin[1])/h) + 3
		n2 = floor(Int, (boundary_rect.xmax[2] - boundary_rect.xmin[2])/h) + 3
        cells = [PreAllocVector(Cell, CELL_SIZEHINT) for _ in 1:n1, _ in 1:n2]
        locks = [ReentrantLock() for _ in 1:n1, _ in 1:n2]
        # the path which is used for visiting neighbors
        magic_path = PathNode[]
        for i2 in (1-n1):(n1-1)
            for i1 in (1-n2):(n2-1)
                key = Key(i1, i2)
                # distance of this cell to (0,0) cell
                rr = h^2*(max(0, abs(i1)-1)^2 + max(0, abs(i2)-1)^2)
                push!(magic_path, PathNode(key, rr))
            end
        end
        # sort by rr
        # where rr is equal, sort by average distance
        sort!(magic_path, by = (node::PathNode -> node.key[1]^2 + node.key[2]^2))
        sort!(magic_path, by = (node::PathNode -> node.rr))
        return new(cells, locks, origin, h, magic_path)
    end
end

@inline function findkey(list::CellList, x::RealVector)::Key
	x = x - list.origin
    @inbounds i1 = floor(Int, x[1]/list.h) + 1
    @inbounds i2 = floor(Int, x[2]/list.h) + 1
	return Key(i1, i2)
end

@inbounds function insert!(list::CellList, x::RealVector, label::Int)::Bool
    key = findkey(list, x)
    if checkbounds(Bool, list.cells, key)
        lock(list.locks[key])
        try
            push!(list.cells[key], label)
        finally
            unlock(list.locks[key])
        end
        return true
    end
    return false
end
