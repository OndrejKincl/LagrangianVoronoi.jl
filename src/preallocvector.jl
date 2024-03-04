# implementation of a pre-allocated vector 
# compared to normal vector, this improves performance by ~50%
# if this is already implemented somewhere, pls let me know

mutable struct PreAllocVector{T} <: AbstractVector{T}
    data::Vector{T}
    last::Int
    PreAllocVector{T}(sizehint::Int) where T = new(Vector{T}(undef, sizehint), 0)
end

function Base.push!(a::PreAllocVector{T}, val::T) where T
    a.last += 1
    if a.last <= length(a.data)
        a.data[a.last] = val
    else
        push!(a.data, val)
    end
    return
end

function Base.isempty(a::PreAllocVector{T})::Bool where T
    return (a.last == 0)
end

function Base.pop!(a::PreAllocVector{T})::T where T
    if isempty(a)
        throw(ArgumentError("array must be non-empty"))
    end
    a.last -= 1
    return a.data[a.last + 1]
end

function Base.length(a::PreAllocVector{T}) where T
    return a.last
end

function Base.size(a::PreAllocVector{T}) where T
    return (a.last)
end

function Base.deleteat!(a::PreAllocVector{T}, ind::Int) where T
    if a.last > ind
        a.data[ind] = a.data[a.last]
    end
    a.last -= 1
    return
end

function Base.empty!(a::PreAllocVector{T}) where T
    a.last = 0
    return
end

function Base.iterate(a::PreAllocVector{T}, state::Int = 0) where T
    state += 1
    if state > a.last
        return nothing
    end
    return (a.data[state], state)
end

function Base.getindex(a::PreAllocVector{T}, i::Int)::T where T
    return a.data[i]
end

function Base.setindex!(a::PreAllocVector{T}, val::T, i::Int) where T
    a.data[i] = val
    return
end

function Base.eachindex(a::PreAllocVector{T}) where T
    return 1:length(a)    
end