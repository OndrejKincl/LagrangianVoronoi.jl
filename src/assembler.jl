using SparseArrays

# assemble vector from a unary function
function assemble_vector(grid::VoronoiGrid, vector_element::Function; fix_avg::Bool = false)::Vector{Float64}
    #if !hasmethod(vector_element, (VoronoiPolygon, ))
    #    throw(ArgumentError("functional argument must be fun(::VoronoiPolygon)::Float64"))
    #end  
    N = length(grid.polygons)
	v = zeros(N)
	for i in 1:N
		v[i] = vector_element(grid, grid.polygons[i])
	end
    if fix_avg
        push!(v, 0.0)
    end
	return v
end

# assemble (sparse) matrix from a binary function
function assemble_matrix(grid::VoronoiGrid, diagonal_element::Function, edge_element::Function; fix_avg::Bool = false)::SparseMatrixCSC{Float64}
    if !hasmethod(diagonal_element, (VoronoiGrid, VoronoiPolygon))
        throw(ArgumentError("first functional argument must be fun(::VoronoiGrid, ::VoronoiPolygon)::Float64"))
    end
    if !hasmethod(edge_element, (VoronoiPolygon, VoronoiPolygon, Edge))
        throw(ArgumentError("second functional argument must be fun(::VoronoiPolygon, ::VoronoiPolygon, ::Edge)::Float64"))
    end
    N = length(grid.polygons)
    I = Int[]
    J = Int[]
    V = Float64[]
    for i in eachindex(grid.polygons)
        p = grid.polygons[i]
        v = diagonal_element(grid, p)
        if v != 0.0
            push!(I, i)
            push!(J, i)
            push!(V, v)
        end
        for e in p.edges
            j = e.label
            if !checkbounds(Bool, grid.polygons, j)
                continue
            end
            q = grid.polygons[j]
            v = edge_element(p, q, e)
            if v != 0.0
                push!(I, i)
                push!(J, j)
                push!(V, v)
            end 
        end
        if fix_avg 
            push!(I, i)
            push!(J, N+1)
            push!(V, -1.0/N)
        end
    end
    if fix_avg
        for j in eachindex(grid.polygons)
            push!(I, N+1)
            push!(J, j)
            push!(V, 1.0/N)
        end
        return sparse(I, J, V, N+1, N+1)
    end
    return sparse(I, J, V, N, N)
end