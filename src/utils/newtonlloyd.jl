
using SparseArrays

function __pushsubmatrix!(I::Vector{Int}, J::Vector{Int}, V::Vector{Float64}, i::Int, j::Int, a::RealMatrix)
    push!(I, 2i-1)
    push!(J, 2j-1)
    push!(V, a[1,1])

    push!(I, 2i-1)
    push!(J, 2j)
    push!(V, a[1,2])

    push!(I, 2i)
    push!(J, 2j-1)
    push!(V, a[2,1])

    push!(I, 2i)
    push!(J, 2j)
    push!(V, a[2,2])
end

function newtonlloyd_assemble(grid::VoronoiGrid)::Tuple{SparseMatrixCSC{Float64}, Vector{Float64}}
    N = length(grid.polygons)
    I = Int64[]
    J = Int64[]
    V = Float64[]
    b = zeros(2N)
    for (i,p) in enumerate(grid.polygons)
        c = centroid(p)
        A = area(p)
        Mii = A*MAT1
        b[2i-1] = A*(p.x[1] - c[1])
        b[2i]   = A*(p.x[2] - c[2])
        for (q,e) in neighbors(p, grid)
            j = e.label
            lrr = lr_ratio(p, q, e)
            m = 0.5*(e.v1 + e.v2)
            Mij = MAT0
            # Simpson rule - this should be exact
            for (node, weight) in ((e.v1, 1/6), (m, 4/6), (e.v2, 1/6))
                Mii -= weight*lrr*outer(node - p.x, node - p.x)
                Mij += weight*lrr*outer(node - p.x, node - q.x)
            end
            __pushsubmatrix!(I, J, V, i, j, Mij)
        end
        __pushsubmatrix!(I, J, V, i, i, Mii)
    end
    A = sparse(I, J, V, 2N, 2N)
    return A, b
end

function newtonlloyd!(grid::VoronoiGrid; maxit::Int = 10, preit::Int = 30, rtol::Float64 = 1e-8, atol::Float64 = 1e-8)
    for it in 1:preit
        for (i,p) in enumerate(grid.polygons)
            p.x = centroid(p)
        end
        remesh!(grid)
    end
    res = Inf
    it = 0
    while it < maxit
        A, b = newtonlloyd_assemble(grid)
        res = norm(b, 1)
        @show res
        if it == 0
            atol = min(atol, res*rtol)
        end
        if res < atol
            break
        end
        dx = -A\b
        for (i,p) in enumerate(grid.polygons)
            vec = dx[2i-1]*VECX + dx[2i]*VECY
            alpha = min(1.0, 0.1*grid.h/norm(vec))
            p.x += alpha*vec
        end
        remesh!(grid)
        it += 1
    end
    if (it == maxit)
        @warn "newton-lloyd solver not converged (residue = $res)"
    else
        @info "newton-lloyd solver converged in $it iterations (residue = $res)"
    end
end 