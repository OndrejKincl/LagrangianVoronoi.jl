module hextest

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using Plots
using WriteVTK

const check_voronoi = false
const plot_graph = false
const export_file = true

mutable struct PhysFields
    n::Int64
    PhysFields(x::RealVector) = new(
        0
    )
end

function populate!(grid::VoronoiGrid, rect::Rectangle, dr::Float64)
    a = ((4/3)^0.25)*dr
    b = ((3/4)^0.25)*dr
    i_min = floor(Int, rect.xmin[1]/a) - 1
    j_min = floor(Int, rect.xmin[2]/b)
    i_max = ceil(Int, rect.xmax[1]/a)
    j_max = ceil(Int, rect.xmax[2]/b)
	for i in i_min:i_max, j in j_min:j_max
        x = RealVector((i + (j%2)/2)*a, j*b) #+ 1e-3*dr*rand(RealVector)
        if isinside(rect, x)
            push!(grid.polygons, VoronoiPolygon{PhysFields}(x))
        end
	end
    k = 1
    for p in grid.polygons
        p.var.n = k 
        k += 1
    end
	return
end

function populate_linear!(grid::VoronoiGrid, dr::Float64)
    N = round(Int, 1/(dr*dr))
    step = 
    for i in 1:N
        x = RealVector(-0.5 + (i-0.5)/N, 0.0)
        push!(grid.polygons, VoronoiPolygon{PhysFields}(x))
    end
    k = 1
    for p in grid.polygons
        p.var.n = k 
        k += 1
    end
	return
end

function populate_square!(grid::VoronoiGrid, dr::Float64)
    N = round(Int, 1/dr)
    for i in 1:N
        for j in 1:N
            x = RealVector(-0.5 + (i-0.5)/N, -0.5 + (j-0.5)/N)
            push!(grid.polygons, VoronoiPolygon{PhysFields}(x))
        end
    end
    k = 1
    for p in grid.polygons
        p.var.n = k 
        k += 1
    end
	return
end

function main()
    dr = 1.0/4
    rect = Rectangle(xlims = (-0.5, 0.5), ylims = (-0.5, 0.5))
    grid = VoronoiGrid{PhysFields}(2*dr, rect)
    populate_square!(grid, dr)

    #=
    id = 163
    poly = grid.polygons[id] 
    remesh!(grid)

    plt = plot(axis_ratio = 1.0, legend = :none)
    scatter!(plt, [p.x[1] for p in grid.polygons], [p.x[2] for p in grid.polygons])
    if LagrangianVoronoi.voronoicut!(poly, grid.polygons[142].x, 142)
        @warn "this is bad"
    end
    for e in poly.edges
        plot!(plt, [e.v1[1], e.v2[1]], [e.v1[2], e.v2[2]])
    end
    savefig(plt, "results/hexagon.pdf")
    =#
    #remesh!(grid)
    #=
    id = 19
    p = grid.polygons[id] 
    LagrangianVoronoi.reset!(p, grid.boundary_rect)
    anim = @animate for q in grid.polygons
        if (p == q)
            continue
        end
        plt = plot(axis_ratio = 1.0, legend = :none, xlims = (-0.5, 0.5), ylims = (-0.5, 0.5))
        scatter!(plt, [z.x[1] for z in grid.polygons], [z.x[2] for z in grid.polygons])
        for e in p.edges
            plot!(plt, [e.v1[1], e.v2[1]], [e.v1[2], e.v2[2]])
        end
        scatter!(plt, [p.x[1], q.x[1]], [p.x[2], q.x[2]], markershape = :star5, markersize = 5)
        LagrangianVoronoi.voronoicut!(p, q.x, 0)
    end
    gif(anim, "hex_anim.gif", fps=1)
    #@show length(grid.polygons)
    #@time remesh!(grid)
    #@time remesh!(grid)
    #@time remesh!(grid)
    #@time remesh!(grid)
    =#
    remesh!(grid)
    if export_file
        vtp1 = export_grid(grid, "results/hexgrid.vtp", :n)
        vtp2 = export_points(grid, "results/honey.vtp")
    end
    vtk_save(vtp1)
    vtk_save(vtp2)
end

end