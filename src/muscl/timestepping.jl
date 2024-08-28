function save_state!(grid::VoronoiGrid)
    @batch for p in grid.polygons
        p.old_state = MUSCLstate(p.x, p.M, p.P, p.E)
    end
end

function load_state!(grid::VoronoiGrid)
    @batch for p in grid.polygons
        p.x = p.old_state.x
        p.M = p.old_state.M
        p.P = p.old_state.P
        p.E = p.old_state.E
    end
end

function RK2_step!(grid::VoronoiGrid, dt::Float64; gamma::Float64 = 1.4)
    save_state!(grid)
    get_intensives!(grid)
    check(grid)
    reconstruction!(grid)
    physical_step!(grid; gamma = gamma)
    update!(grid, 0.5*dt)
    get_intensives!(grid)
    reconstruction!(grid)
    physical_step!(grid)
    load_state!(grid)
    update!(grid, dt)
end


function check(grid::VoronoiGrid)
    for p in grid.polygons
        @assert (p.M > 0) "negative mass"
        @assert (p.rho > 0) "negative density"
        #@assert (p.e > 0) "negative energy"
        #pr, a = ideal_gas_law(p.rho, p.u, p.e)
        #@assert (pr > 0) "negative pressure"
    end
end