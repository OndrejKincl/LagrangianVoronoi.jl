module gridtest

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using Plots
using WriteVTK

const check_voronoi = false
const plot_graph = false
const export_file = true

mutable struct PhysFields
    T::Float64 # temperature
    v::RealVector # velocity
    PhysFields(x::RealVector) = new(
        randn(Float64), 
        randn(RealVector)
    )
end
                

function main()
    dr = 1.0e-2
    rect = Rectangle(xlims = (-0.5, 0.5), ylims = (-0.5, 0.5))
    grid = VoronoiGrid{PhysFields}(2*dr, rect)
    
    # add some points
    N = pi/(dr*dr)
    for i in 1:N
        r = sqrt(i/N)
        theta = 2.39996322972865332*i
        x = RealVector(r*cos(theta), r*sin(theta))
        if isinside(rect, x)
            push!(grid.polygons, VoronoiPolygon{PhysFields}(x))
        end    
    end

    @show length(grid.polygons)
    @time remesh!(grid)
    @time remesh!(grid)
    @time remesh!(grid)
    @time remesh!(grid)
    if export_file
        @time vtp = export_grid(grid, "results/poisson/poisson_cells.vtp", :T)
    end
    vtk_save(vtp)
end

end