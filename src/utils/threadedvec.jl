import Base: eltype, size
import Base: *
import Base: similar, copyto!, fill!, pointer
import LinearAlgebra: axpy!, rmul!, dot, norm

struct ThreadedVec{T} <: DenseVector{T}
    val::Vector{T}
    ThreadedVec(val::Vector{T}) where T = new{T}(val)
    ThreadedVec{T}(::UndefInitializer, n::Int64) where T = new{T}(Vector{T}(undef, n))
end

eltype(v::ThreadedVec) = eltype(v.val)
size(v::ThreadedVec, I...) = size(v.val, I...)


function Base.getindex(x::ThreadedVec, i::Int)
    return x.val[i]
end

function Base.setindex!(x::ThreadedVec{T}, val::T, i::Int) where T
    return x.val[i] = val
end

function similar(x::ThreadedVec, type::Type{T} = eltype(x.val), n::Integer = length(x.val)) where T
    return ThreadedVec(zeros(type, n))
end

function axpy!(a::Number, x::ThreadedVec{T}, y::ThreadedVec{T})::ThreadedVec{T} where T
    @batch for i in eachindex(x.val)
        @inbounds y[i] += a*x[i]
    end
    return y
end

function copyto!(dst::ThreadedVec{T}, src::ThreadedVec{T}) where T
    @batch for i in eachindex(dst.val)
        @inbounds dst[i] = src[i]
    end
end

function fill!(x::ThreadedVec{T}, val::T) where T
    @batch for i in eachindex(x.val)
        @inbounds x[i] = val
    end
end

function rmul!(x::ThreadedVec{T}, val::T) where T
    @batch for i in eachindex(x.val)
        @inbounds x[i] *= val
    end
end

function pointer(x::ThreadedVec)
    return pointer(x.val)
end