import Base: eltype, size
import LinearAlgebra: mul!
using SparseArrays
using Base.Threads
import Base: *
import Base: similar, copyto!, fill!, pointer
import LinearAlgebra: axpy!, rmul!, dot, norm

struct ThreadedMul{Tv,Ti}
    A::SparseMatrixCSC{Tv,Ti}
end

function mul!(y::AbstractVector, M::ThreadedMul, x::AbstractVector)
    @threads for i = 1 : M.A.n
        _threaded_mul!(y, M.A, x, i)
    end
    y
end

*(A::ThreadedMul, y::AbstractVector) = mul!(similar(y), A, y)

@inline function _threaded_mul!(y, A::SparseMatrixCSC{Tv}, x, i) where {Tv}
    s = zero(Tv)
    for j = A.colptr[i] : A.colptr[i + 1] - 1
        @inbounds s += A.nzval[j] * x[A.rowval[j]]
    end
    @inbounds y[i] = s
    y
end

eltype(M::ThreadedMul) = eltype(M.A)
size(M::ThreadedMul, I...) = size(M.A, I...)

#Base.similar(x) : return vec y similar to x
#LinearAlgebra.axpy!(a, x, y) : assign y = y + a*x and return y
#LinearAlgebra.rmul!(x, b) :  scale x by scalar b in place
#LinearAlgebra.dot(a, b) :  return the dot product of a and b
#Base.copyto!(dst, src) : overwrite dst[i] by src[i]

struct ThreadedVec{T} <: DenseVector{T}
    val::Vector{T}
    psums::Vector{T}
    ThreadedVec(val::Vector{T}) where T = new{T}(val, zeros(T, Threads.nthreads()))
    ThreadedVec{T}(::UndefInitializer, n::Int64) where T = new{T}(Vector{T}(undef, n), zeros(T, Threads.nthreads()))
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

function dot(x::ThreadedVec{T}, y::ThreadedVec{T})::T where T
    fill!(x.psums, zero(eltype(x)))
    @threads for i in eachindex(x.val)
        @inbounds x.psums[Threads.threadid()] += x.val[i]*y.val[i]
    end
    return sum(x.psums, init = zero(eltype(x)))
end


function norm(x::ThreadedVec{T})::T where T
    return sqrt(dot(x,x))
end

function axpy!(a::Number, x::ThreadedVec, y::ThreadedVec)::ThreadedVec
    @threads for i in eachindex(x.val)
        @inbounds y.val[i] += a*x.val[i]
    end
    return y
end

function copyto!(dst::ThreadedVec, src::ThreadedVec)
    @threads for i in eachindex(dst.val)
        @inbounds dst.val[i] = src.val[i]
    end
end

function fill!(x::ThreadedVec, val::Number)
    @threads for i in eachindex(x.val)
        @inbounds x.val[i] = val
    end
end

function rmul!(x::ThreadedVec, val::Number)
    @threads for i in eachindex(x.val)
        @inbounds x.val[i] *= val
    end
end

function pointer(x::ThreadedVec)
    return pointer(x.val)
end