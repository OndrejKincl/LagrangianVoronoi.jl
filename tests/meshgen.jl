module meshtest

const dr = 1/200
const ncolors = 5

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi, WriteVTK


function main()
    domain = UnitRectangle()
    grid = GridNS(domain, dr)
    populate_rand!(grid)
    for p in grid.polygons
        p.P = Float64(abs(rand(Int)%ncolors))
    end
    @time remesh!(grid)
    @show length(grid.polygons)
    vtp = export_grid(grid, "results/mesh.vtp", :P)
    vtk_save(vtp)
    vtp = export_points(grid, "results/points.vtp")
    vtk_save(vtp)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end