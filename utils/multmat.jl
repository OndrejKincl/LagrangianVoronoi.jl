import Base: eltype, size
import LinearAlgebra: mul!
using SparseArrays
using Base.Threads
using Polyester
import Base: *
import Base: similar, copyto!, fill!, pointer
import LinearAlgebra: axpy!, rmul!, dot, norm

struct ThreadedMul{Tv,Ti}
    A::SparseMatrixCSC{Tv,Ti}
end

function mul!(y::AbstractVector, M::ThreadedMul, x::AbstractVector)
    @batch for i = 1 : M.A.n
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

function dot(u::ThreadedVec{Float64}, v::ThreadedVec{Float64})::Float64
 
    Nt = nthreads()
    Lt = length(u) ÷ Nt
    if length(u) != length(v)
        throw(ArgumentError("vectors must have same dims"))
    end
    # Calculate partial sums.
    s = Vector{Task}(undef, nthreads()-1)
    for j = 1:Nt-1  # last chunk is handled separately just in case length(v) is not multiple of Nt
        lo = (j-1) * Lt + 1
        hi = j * Lt
        s[j] = @spawn begin
            psum = 0.0
            for i in lo:hi
                @inbounds psum += u[i]*v[i]
            end
            return psum
        end
    end
    # Add partial sums
    lo_last = (Nt-1) * Lt + 1
    stot = 0.0
    for i in lo_last:length(u) # handle last chunk
        @inbounds stot += u[i]*v[i]
    end
    for j = 1:Nt-1
        stot += fetch(s[j])
    end
    return stot

    return dot(u.val,v.val)
end

function norm(x::ThreadedVec{Float64})::Float64
    return sqrt(dot(x,x))
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