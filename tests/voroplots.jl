module voroplots

using Plots; pgfplotsx()
using LaTeXStrings, Measures, Colors, FixedPointNumbers, SparseArrays
include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using LinearAlgebra:norm
using WriteVTK

const ORANGE = plot_color(:darkorange1, 0.5)
const BLUE = plot_color(:royalblue1, 0.5)

mutable struct Fields
    select::Bool
    id::Int
    Fields(x) = new(false, 0)
end

function draw_grid!(plt, grid)
    for p in grid.polygons
        draw_polygon!(plt, p)
    end
    for (i,p) in enumerate(grid.polygons)
        annotate!(plt, p.x[1], p.x[2]-0.05, latexstring("\\mathbf{x}_{", i, "}"), :black)
    end
    scatter!(plt, [p.x[1] for p in grid.polygons], [p.x[2] for p in grid.polygons], mc = :black)
end

function draw_polygon!(plt, p)
    verts = [(e.v1[1], e.v1[2]) for e in p.edges]
    push!(verts, verts[1])
    poly = Shape(verts)
    if p.var.select
        plot!(plt, poly, fillcolor = ORANGE)
    else
        plot!(plt, poly, fillcolor = BLUE)
    end
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
            LagrangianVoronoi.sort_edges!(p)
        elseif p.x[2] < corner[2]
            y = p.x[1]*VECX + (2*corner[2] - p.x[2])*VECY
            LagrangianVoronoi.voronoicut!(p, y, 0)
            LagrangianVoronoi.sort_edges!(p)
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

function draw_cell!(plt, cell_list, key, highlight = false)
    i = key[1]
    j = key[2]
    ox = cell_list.origin[1]
    oy = cell_list.origin[2]
    h = cell_list.h
    x0 = ox + h*(i-1)
    x1 = x0 + h
    y0 = oy + h*(j-1)
    y1 = y0 + h
    verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    push!(verts, verts[1])
    if highlight
        plot!(plt, Shape(verts), color = ORANGE, lw = 1)
    else
        plot!(plt, verts, color = plot_color(:black, 0.1), lw = 1)
    end
end

function draw_cell_list!(plt, cell_list)
    for key in CartesianIndices(cell_list.cells)
        draw_cell!(plt, cell_list, key)
    end
end


function mesh_construction()
    lims = (-0.5, 0.5)
    dr = 0.1
    grid = VoronoiGrid{Fields}(2*dr, Rectangle(xlims = lims, ylims = lims))
    
    #push!(grid.polygons, VoronoiPolygon{Fields}(VEC0))
    
    
    p0 = grid.polygons[1]
    LagrangianVoronoi.reset!(p0, grid.boundary_rect)
    x = p0.x
    prr = LagrangianVoronoi.influence_rr(p0)
    key0 = LagrangianVoronoi.findkey(grid.cell_list, x)
    frame = 0
    for node in grid.cell_list.magic_path

        rr = node.rr
        offset = node.key
        if (rr > prr)
            break
        end
        key = key0 + offset

        plt = plot(axis_ratio = 1.0, legend = :none, xlims = (-3*dr, 3*dr), ylims = (-3*dr, 3*dr), grid = :none, axis=([], false))
        LagrangianVoronoi.sort_edges!(p0)
        
        active_ps = []
        for p in grid.polygons
            if p != p0 && LagrangianVoronoi.findkey(grid.cell_list, p.x) == key
                push!(active_ps, p)
            end
        end


        draw_polygon!(plt, p0)
        draw_cell_list!(plt, grid.cell_list)
        if frame < 5
            draw_cell!(plt, grid.cell_list, key, true)
            scatter!(plt, [p.x[1] for p in active_ps], [p.x[2] for p in active_ps], mc = :red, ms = 4)
        end

        scatter!(plt, [p.x[1] for p in grid.polygons], [p.x[2] for p in grid.polygons], mc = :black, ms = 4)
        scatter!(plt, [p0.x[1]], [p0.x[2]], mc = :green, ms = 6, markershape = :hex)
        savefig(plt, "results/mesh_construction/frame$(frame).pdf")
        frame += 1

    

        if !(checkbounds(Bool, grid.cell_list.cells, key))
            continue
        end
        for i in grid.cell_list.cells[key]
            y = grid.polygons[i].x
            if (p0.x == y) || (norm_squared(p0.x-y) > prr)
                continue
            end
            if LagrangianVoronoi.voronoicut!(p0, y, i)
                prr = LagrangianVoronoi.influence_rr(p0)
            end
        end
    end
    
end

function make_colorbar(x_start, x_end, label, cmap)
    data = [(i-1)/9*x_start + (10-i)/9*x_end for i in 1:10, j in 1:10]
    plt = heatmap(1:10, 1:10, data, colorbar = :top, cmap = cmap, colorbar_title = label, colorbar_titlefontsize=40, colorbar_tickfontsize=12, right_margin = 50mm)
    savefig(plt, "colormap.pdf")
end


function sparsity_comparison()
    lims = (-0.5, 0.5)
    dr = 0.04
    h_SPH = 3.0*dr
    grid = VoronoiGrid{Fields}(2*dr, Rectangle(xlims = lims, ylims = lims))
    #populate_lloyd!(grid, dr, niterations = 10)
    populate_rand!(grid, dr)
    sort!(grid.polygons, by = (p -> LagrangianVoronoi.findkey(grid.cell_list, p.x)))
    remesh!(grid)
    #populate_rect!(grid, dr)
    #populate_rand!(grid, dr)
    #populate_vogel!(grid, dr)
    begin
        # construct SPH matrix
        N = length(grid.polygons)
        V = Float64[]
        I = Int[]
        J = Int[]
        for (i,p) in enumerate(grid.polygons)
            p.var.id = i
            for (j,q) in enumerate(grid.polygons)
                if norm(p.x-q.x) < h_SPH
                    push!(V, 1.0)
                    push!(I, i)
                    push!(J, j)
                end
            end
        end
        A_SPH = sparse(I, J, V, N, N)
        plt = plot(axis_ratio = 1.0)
        spy!(plt, A_SPH)
        savefig(plt, "spy_SPH.pdf")
        println("SPH:")
        println("nonzeros_per_node = $(length(V)/N)")
        println("sparsity = $(length(V)/(N^2))")
    end

    begin
        # construct LVM matrix
        N = length(grid.polygons)
        V = Float64[]
        I = Int[]
        J = Int[]
        for (i,p) in enumerate(grid.polygons)
            push!(V, 1.0)
            push!(I, i)
            push!(J, i)
            for e in p.edges
                if isboundary(e)
                    continue
                end
                j = e.label
                q = grid.polygons[j]
                push!(V, 1.0)
                push!(I, i)
                push!(J, j)
            end
        end
        A_LVM = sparse(I, J, V, N, N)
        plt = plot(axis_ratio = 1.0)
        spy!(plt, A_LVM)
        savefig(plt, "spy_LVM.pdf")
        println("LVM:")
        println("nonzeros_per_node = $(length(V)/N)")
        println("sparsity = $(length(V)/(N^2))")
    end

    vtk = export_grid(grid, "sparsity_grid.vtp", :id)
    vtk_save(vtk)


end

end