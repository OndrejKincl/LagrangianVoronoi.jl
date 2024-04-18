import Base: eltype, size
using Base.Threads
using Polyester
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

# TOO MUCH OVERHEAD --- NOT WORTH IT
#=
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
end

function norm(x::ThreadedVec{Float64})::Float64
    return sqrt(dot(x,x))
end
=#

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