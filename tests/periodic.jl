module periodic

const dr = 1/50
const dt = 1/100
const t_end = 1.0

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi, WriteVTK

function set_v!(grid::VoronoiGrid)
    for p in grid.polygons
        v1 =  1.0
        v2 =  0.1
        p.v = v1*VECX + v2*VECY
    end
end


function highlight_cell!(grid::VoronoiGrid)
    for p in grid.polygons
        p.phase = 0
    end
    p = grid.polygons[1000]
    p.phase = 1
    for (q,e) in neighbors(p, grid)
        q.phase = 2
    end
end

function main()
    domain = UnitRectangle()
    grid = GridNSc(domain, dr; xperiodic = true, yperiodic = true)
    populate_vogel!(grid)
    remesh!(grid)
    t = 0.0
    pvd = paraview_collection("results/periodic/cells.pvd")
    k = 0
    while t < t_end
        @show t
        set_v!(grid)
        @time move!(grid, dt)
        highlight_cell!(grid)
        pvd[t] = export_grid(grid, "results/periodic/frame$k.vtp", :v, :phase)
        k += 1
        t += dt
    end
    vtk_save(pvd)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end