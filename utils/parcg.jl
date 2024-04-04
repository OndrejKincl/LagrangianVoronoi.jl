include("multmat.jl")

mutable struct ThreadedSum
    psums::Vector{Float64}
    ThreadedSum() = new(zeros(Threads.nthreads()))
end

struct ParCgSolver
    N::Int
    r::Vector{Float64}
    p::Vector{Float64}
    Ap::Vector{Float64}
    x::Vector{Float64}
    ParCgSolver(N::Int) = begin
        r = zeros(N)
        p = zeros(N)
        Ap = zeros(N)
        x = zeros(N)
        return new(N, r, p, Ap, x)
    end
end


function pardot(u::Vector{Float64}, v::Vector{Float64})
    Nt = nthreads()
    Lt = length(u) ÷ Nt
    if length(u) != length(v)
        throw(ArgumentError("vectors must have same dims"))
    end
    # Calculate partial sums.
    s = Vector{Task}(undef, Nt-1)
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


function parcg!(sol::ParCgSolver, A::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64};
    itmax::Int = 2000, rtol = sqrt(eps(Float64)), atol = sqrt(eps(Float64)),
    verbose = false)::Bool
    #initialization
    for i in 1:sol.N
        @inbounds sol.r[i] = b[i] # -A*x0
        @inbounds sol.p[i] = sol.r[i]
        @inbounds sol.x[i] = 0.0 # = x0
    end
    rri = pardot(sol.r, sol.r)
    rr1 = rri
    for k in 1:itmax
        if verbose
            println("Iteration $(k)")
        end
        rr0 = rr1
        @batch for i in 1:sol.N
            sol.Ap[i] = 0.0
            for j = A.colptr[i] : A.colptr[i + 1] - 1
                @inbounds sol.Ap[i] += A.nzval[j] * sol.p[A.rowval[j]]
            end
        end 
        pAp = pardot(sol.Ap, sol.p)
        alpha = rr0/pAp
        @batch for i in 1:sol.N
            @inbounds sol.r[i] -= alpha*sol.Ap[i]
            @inbounds sol.x[i] += alpha*sol.p[i]
        end
        rr1 = pardot(sol.r, sol.r)
        if (rr1 < atol^2) || (rr1/rri < rtol^2)
            if verbose
                @info "Solver converged at iteration $(k)."
            end
            return true
        end  
        beta = rr1/rr0
        @batch for i in 1:sol.N
            @inbounds sol.p[i] = beta*sol.p[i] + sol.r[i]
        end
        if verbose
            println("abs residue = $(sqrt(rr1)), tol = $(atol)")
            println("rel residue = $(sqrt(rr1/rri)), tol = $(rtol)")
            println()
        end
    end
    if verbose
        @warn "Solver did not converge."
    end
    return false
end