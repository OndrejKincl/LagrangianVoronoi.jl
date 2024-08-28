const Lrr_cache = Vector{Vector{Float64}}

function refresh!(cache::Lrr_cache, grid::VoronoiGrid)
    @batch for i in eachindex(grid.polygons)
        empty!(cache[i])
        p = grid.polygons[i]
        for (q,e) in neighbors(p, grid)
            push!(cache[i], lr_ratio(p,q,e))
        end
    end
end

function new_lrr_cache(grid::VoronoiGrid)::Lrr_cache
    n = length(grid.polygons)
    cache = [PreAllocVector(Float64, POLYGON_SIZEHINT) for _ in 1:n]
    refresh!(cache, grid)
    return cache
end