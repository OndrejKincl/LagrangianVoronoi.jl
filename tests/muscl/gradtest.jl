module gradtest

include("../../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi

using WriteVTK, Match, Plots, LinearAlgebra

function scalar_fun(x::RealVector)::Float64
    return x[1]^2 + x[2]^2
    #return x[1] + x[2]
end

function scalar_grad(x::RealVector)::RealVector
    return 2.0*x[1]*VECX + 2.0*x[2]*VECY
    #return RealVector(1.0, 1.0)
end

function vector_fun(x::RealVector)::RealVector
    return 1.0/pi*cos(pi*x[2])*VECX + 1.0/pi*sin(pi*x[1])*VECY
    #return x[2]*VECX
end

function vector_grad(x::RealVector)::RealMatrix
    return RealMatrix(0.0, cos(pi*x[1]), -sin(pi*x[2]), 0.0)
    #return RealMatrix(0.0, 0.0, 1.0, 0.0)
end

function main(dr::Float64)
    diag = 0.5*(VECX + VECY)
    domain = Rectangle(-diag, diag)
    grid = GridMUSCL(domain, dr)
    #populate_hex!(grid)
    populate_circ!(grid)
    for p in grid.polygons
        p.c = centroid(p)
        p.rho = scalar_fun(p.c)
        p.u = vector_fun(p.c)
        p.e = scalar_fun(p.c)
    end
    @time reconstruction!(grid, slopelimiter = true)
    err = 0.0

    rhos = Float64[]
    xs   = -0.5:(0.01*dr):0.5
    rho_min = Inf
    for x in xs
        a = RealVector(x, 0.0)
        p = nearest_polygon(grid, a)
        rho = p.rho + dot(p.Grho, a - p.c)
        rho_min = min(rho, rho_min)
        push!(rhos, rho)
    end
    @show rho_min


    plt = plot(xs, rhos)
    savefig(plt, "results/linedata.pdf")
    for p in grid.polygons
        p.Grho -= scalar_grad(p.c)
        p.Gu -= vector_grad(p.c)
        p.Ge -= scalar_grad(p.c)
        p.e = sqrt(norm(p.Grho)^2 + norm(p.Ge)^2 + norm(p.Gu)^2)
        if !isboundary(p)
            err += p.e^2*area(p)
        end
    end
    err  = sqrt(err)
    @show err

    vtk_file = export_grid(grid, "results/grad.vtp", :e)
    vtk_save(vtk_file)
    
end

end