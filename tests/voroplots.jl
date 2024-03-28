module voroplots

using Plots, LaTeXStrings
include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi

mutable struct Fields
    select::Bool
    Fields(x) = new(false)
end

function draw_grid!(plt, grid)
    for p in grid.polygons
        verts = [(e.v1[1], e.v1[2]) for e in p.edges]
        push!(verts, verts[1])
        poly = Shape(verts)
        if p.var.select
            plot!(plt, poly, fillcolor = plot_color(:darkorange1, 0.5))
        else
            plot!(plt, poly, fillcolor = plot_color(:royalblue1, 0.5))
        end
    end
    for (i,p) in enumerate(grid.polygons)
        annotate!(plt, p.x[1], p.x[2]-0.05, latexstring("\\mathbf{x}_{", i, "}"), :black)
    end
    scatter!(plt, [p.x[1] for p in grid.polygons], [p.x[2] for p in grid.polygons], mc = :black)
end

# a Voronoi mesh of non-convex domain, where the derivative of Omega_1 wrt x_2 is undefined
function disco()
    corner = (0.5, 0.6)
    lims = (0.0, 1.0)
    grid = VoronoiGrid{Fields}(1.0, Rectangle(xlims = lims, ylims = lims))
    xs = [(0.45, 0.5), (0.55, 0.5), (0.2, 0.8), (0.1, 0.15), (0.65, 0.2), (0.8, 0.55)]
    for x in xs
        push!(grid.polygons, VoronoiPolygon{Fields}(x[1]*VECX + x[2]*VECY))
    end
    grid.polygons[1].var.select = true
    remesh!(grid)
    plt = plot(axis_ratio = 1.0, legend = :none, xlims = (0.0, 1.0), ylims = (0.0, 1.0), grid = :none)
    for p in grid.polygons
        if p.x[1] < corner[1]
            y = (2*corner[1] - p.x[1])*VECX + p.x[2]*VECY
            LagrangianVoronoi.voronoicut!(p, y, 0)
            LagrangianVoronoi.normalize!(p)
        elseif p.x[2] < corner[2]
            y = p.x[1]*VECX + (2*corner[2] - p.x[2])*VECY
            LagrangianVoronoi.voronoicut!(p, y, 0)
            LagrangianVoronoi.normalize!(p)
        end
    end
    draw_grid!(plt, grid)
    #mask = Shape([(0.5, 0.6), (0.5, 1.0), (1.0, 1.0), (1.0, 0.6)])
    #plot!(plt, mask, fillcolor = :white, linecolor = :white)
    #mask = Shape([(0.5, 0.6), (0.5, 2.0), (2.0, 2.0), (2.0, 0.6)])
    #plot!(plt, mask, fillcolor = :white) #, linecolor = :white)
    savefig(plt, "results/disco.pdf")
end

# a Voronoi mesh, where the area of cell Omega_1 increases under linear compression
function compress()
    lims = (-0.5, 0.5)
    grid = VoronoiGrid{Fields}(1.0, Rectangle(xlims = lims, ylims = lims))
    a = 0.1
    b = 0.2
    c = 0.35
    xs = [(0.0, 0.0), (a, b), (a, -b), (-a, -b), (-a, b), (c, c), (-c, c), (c, -c), (-c, -c), (0.0, -c), (0.0, c)]
    for x in xs
        push!(grid.polygons, VoronoiPolygon{Fields}(x[1]*VECX + x[2]*VECY))
    end
    grid.polygons[1].var.select = true
    remesh!(grid)
    plt = plot(axis_ratio = 1.0, legend = :none, xlims = lims, ylims = lims, grid = :none)
    draw_grid!(plt, grid)
    savefig(plt, "results/compress1.pdf")
    A = area(grid.polygons[1])
    @show A
    lambda = 0.5
    for p in grid.polygons
        p.x = 0.5*p.x[1]*VECX + p.x[2]*VECY 
    end
    remesh!(grid)
    plt = plot(axis_ratio = 1.0, legend = :none, xlims = lims, ylims = lims, grid = :none)
    draw_grid!(plt, grid)
    savefig(plt, "results/compress2.pdf")
    A = area(grid.polygons[1])
    @show A
end

end