import Base: eltype, size
import LinearAlgebra: mul!
using SparseArrays
using Base.Threads
import Base: *

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
    @inbounds for j = A.colptr[i] : A.colptr[i + 1] - 1
        s += A.nzval[j] * x[A.rowval[j]]
    end
    @inbounds y[i] = s
    y
end

eltype(M::ThreadedMul) = eltype(M.A)
size(M::ThreadedMul, I...) = size(M.A, I...)