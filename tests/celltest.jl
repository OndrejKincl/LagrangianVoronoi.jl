module celltest

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using Plots

function main()
    N = 200
    xlims = (-1.0, 1.0)
    ylims = (-1.0, 1.0)
    
    rect = Rectangle(xlims = xlims, ylims = ylims)
    @time list = LagrangianVoronoi.CellList(3.0/sqrt(N), rect)
    points = RealVector[]
    for i in 1:N
        s = rand()
        t = rand()
        x = s*xlims[1] + (1-s)*xlims[2]
        y = t*ylims[1] + (1-t)*ylims[2]
        p = RealVector(x,y)
        push!(points, p)
        LagrangianVoronoi.insert!(list, p, i)
    end
    key0 = LagrangianVoronoi.findkey(list, RealVector(0.0, 0.0))
    p = points[list.cells[key0][1]]
    xs = [q[1] for q in points]
    ys = [q[2] for q in points]

    n = 1
    anim = @animate for node in list.magic_path
        key = key0 + node.key
        if !(checkbounds(Bool, list.cells, key))
            continue
        end 
        plt = plot(axis_ratio = 1.0, legend = :none, xlims = xlims, ylims = ylims)
        xs_cell = [points[i][1] for i in list.cells[key]]
        ys_cell = [points[i][2] for i in list.cells[key]]
        scatter!(plt, xs,  ys, color = :black, markersize = 1)
        scatter!(plt, xs_cell,  ys_cell, markershape = :star5, color = :blue)
        scatter!(plt, [p[1]],  [p[2]], markershape = :star5, color = :red)
        A = list.origin + RealVector(list.h*(key[1]-1), list.h*(key[2]-1))
        B = A + list.h*VECX
        C = B + list.h*VECY
        D = C - list.h*VECX
        plot!(plt, [A.x, B.x, C.x, D.x, A.x], [A.y, B.y, C.y, D.y, A.y])
        n += 1
        if (n > 200)
            break
        end
    end
    gif(anim, "celltest.gif", fps = 2)

end

end