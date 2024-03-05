using SparseArrays
using Base.Threads

# assemble (sparse) matrix from functions on VoronoiPolygons
# todo : make paralel
function assemble_system(
    grid::VoronoiGrid, 
    diagonal_element::Function, edge_element::Function, vector_element::Function;
    filter::Function = (::VoronoiPolygon -> true),
    constrained_average::Bool = false
    )::Tuple{SparseMatrixCSC{Float64}, Vector{Float64}}

    if !hasmethod(diagonal_element, (VoronoiGrid, VoronoiPolygon))
        throw(ArgumentError("diagonal element must be a function (::VoronoiGrid, ::VoronoiPolygon)::Float64"))
    end
    if !hasmethod(edge_element, (VoronoiPolygon, VoronoiPolygon, Edge))
        throw(ArgumentError("edge element must be a function (::VoronoiPolygon, ::VoronoiPolygon, ::Edge)::Float64"))
    end
    if !hasmethod(vector_element, (VoronoiGrid, VoronoiPolygon))
        throw(ArgumentError("edge element must be a function (::VoronoiGrid, ::VoronoiPolygon)::Float64"))
    end
    N = 0
    # assign ids
    for p in grid.polygons
        if filter(p)
            N += 1
            p.id = N
        end
    end
    rhs = zeros(N)
    I = [Int[] for _ in 1:Threads.nthreads()]
    J = [Int[] for _ in 1:Threads.nthreads()]
    V = [Float64[] for _ in 1:Threads.nthreads()]
    @threads for p in grid.polygons
        if !filter(p)
            continue
        end
        i = p.id
        rhs[i] = vector_element(grid, p)
        v = diagonal_element(grid, p)
        push_matrix_element!(I, J, V, i, i, v)
        for e in p.edges
            if !checkbounds(Bool, grid.polygons, e.label)
                continue
            end
            q = grid.polygons[e.label]
            if !filter(q)
                continue
            end
            j = q.id
            v = edge_element(p, q, e)
            push_matrix_element!(I, J, V, i, j, v)
        end
        if constrained_average
            push_matrix_element!(I, J, V, i, N+1, 1.0)
        end
    end
    if constrained_average
        push!(rhs, 0.0)
        for j in 1:N
            push_matrix_element!(I, J, V, N+1, j, 1.0)
        end
        N += 1
    end
    for n in 2:Threads.nthreads()
        append!(I[1], I[n])
        append!(J[1], J[n])
        append!(V[1], V[n])
    end
    return (sparse(I[1], J[1], V[1], N, N), rhs)
end

function push_matrix_element!(I::Vector, J::Vector, V::Vector, i::Int, j::Int, v::Float64)
    _I = I[Threads.threadid()]
    _J = J[Threads.threadid()]
    _V = V[Threads.threadid()]
    if v != 0.0
        push!(_I, i)
        push!(_J, j)
        push!(_V, v)
    end
end