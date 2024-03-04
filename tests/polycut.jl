module polycut

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using Plots

function main()
    p = RealVector(0.2, 0.1)
    qs = [RealVector(rand() - 0.5, rand() - 0.5) for _ in 1:40]
    poly = VoronoiPolygon(p)
    rect = Rectangle(xlims = (-0.5, 0.5), ylims = (-0.5, 0.5))
    t_end = 1.0
    dt = t_end/100
    speed = 2.0
    anim = @animate for t in 0:dt:t_end
        for i in eachindex(qs)
            qs[i] += dt*speed*RealVector(randn(), randn())
        end
        LagrangianVoronoi.reset!(poly, rect)
        for q in qs
            LagrangianVoronoi.voronoicut!(poly, q, 0)
        end
        plt = plot(
            axis_ratio = 1.0, legend = :none, 
            xlims = (-1.0, 1.0), ylims = (-1.0, 1.0)
        )
        scatter!(plt, [p.x], [p.y], marker = :star5)
        scatter!(plt, [q[1] for q in qs], [q[2] for q in qs])
        for e in poly.edges
            plot!(plt, [e.v1[1], e.v2[1]], [e.v1[2], e.v2[2]], color = :blue)
        end
        theta = 0.0:0.2:2.2*pi
        r = sqrt(poly.rr)
        plot!(plt, [p[1] .+ r.*cos.(theta)], [p[2] .+ r.*sin.(theta)], color = :blue)
    end
    gif(anim, "vorotest.gif", fps = 15)
end

end