"""
    FastVector{T} <: AbstractVector{T}

A vector with preallocated memory than can only increase in size.
Compared to normal vector, this improves the performance of mesh generation by about 50%.
If this is already implemented somewhere, please let me know.
"""
mutable struct FastVector{T} <: AbstractVector{T}
    data::Vector{T}
    last::Int
    FastVector{T}(sizehint::Int) where T = new(Vector{T}(undef, sizehint), 0)
end

function Base.push!(a::FastVector{T}, val::T) where T
    a.last += 1
    if a.last <= length(a.data)
        a.data[a.last] = val
    else
        push!(a.data, val)
    end
    return
end

function Base.isempty(a::FastVector{T})::Bool where T
    return (a.last == 0)
end

function Base.pop!(a::FastVector{T})::T where T
    if isempty(a)
        throw(ArgumentError("array must be non-empty"))
    end
    a.last -= 1
    return a.data[a.last + 1]
end

function Base.length(a::FastVector{T}) where T
    return a.last
end

function Base.size(a::FastVector{T}) where T
    return (a.last)
end

function Base.deleteat!(a::FastVector{T}, ind::Int) where T
    if a.last > ind
        a.data[ind] = a.data[a.last]
    end
    a.last -= 1
    return
end

function Base.empty!(a::FastVector{T}) where T
    a.last = 0
    return
end

function Base.iterate(a::FastVector{T}, state::Int = 0) where T
    state += 1
    if state > a.last
        return nothing
    end
    return (a.data[state], state)
end

function Base.getindex(a::FastVector{T}, i::Int)::T where T
    return a.data[i]
end

function Base.setindex!(a::FastVector{T}, val::T, i::Int) where T
    a.data[i] = val
    return
end

function Base.eachindex(a::FastVector{T}) where T
    return 1:length(a)    
end