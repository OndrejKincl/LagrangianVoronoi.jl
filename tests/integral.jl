module integraltest

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using Plots
using WriteVTK
using Match
using Plots, LaTeXStrings, LinearAlgebra
using SmoothedParticles:rDwendland2

const TaylorExpansion = QuadraticExpansion
const hr_ratio = 3
const test_no = 1

function testfun(n::Int, x::RealVector)
    return @match n begin
        1 => cos(pi*x[1])*cos(pi*x[2]) + cos(4*pi*x[1])*cos(4*pi*x[2])
        2 => exp(-x[1])*exp(-x[2]) + (x[1]^4)*(x[2]^2)
        _ => throw("invalid test_no")
    end
end

function exact_integral(n::Int)
    return @match n begin
        1 => 4.0/(pi^2)
        2 => Float64(1.0872029362971542236224779081807900318697248913983940761414710960)
        _ => throw("invalid test_no")
    end
end


mutable struct PhysFields
    u::Float64
    u_taylor::TaylorExpansion
    PhysFields(x::RealVector) = new(
        testfun(test_no, x),
        zero(TaylorExpansion),
    )
end

function integrate_u(grid::VoronoiGrid)::Float64
    I = 0.0
    for p in grid.polygons
        p.var.u_taylor = ls_reconstruction(TaylorExpansion, grid, p, p -> p.var.u, kernel = rDwendland2)
        I += integral(p, p.var.u, p.var.u_taylor)
    end
    return I
end 


function get_error(N::Int)::Float64
    dr = 1.0/N
    rect = Rectangle(xlims = (-0.5, 0.5), ylims = (-0.5, 0.5))
    grid = VoronoiGrid{PhysFields}(hr_ratio*dr, rect)
    
    @show N

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
    remesh!(grid)
    @time I = integrate_u(grid)
    err = abs(I/exact_integral(test_no) - 1.0)
    #err = I
    @show err
    return err
end

function main()
    N = [32, 48, 72, 108, 162, 243]
    e = [get_error(_N) for _N in N]
    logN = log10.(N)
    loge = log10.(e)
    plt = plot(
        logN, loge, 
        axis_ratio = 1, 
        xlabel = L"\log \, N", ylabel = L"\log \, \epsilon", 
        markershape = :hex, 
        label = ""
    )
    # linear regression
    A = [logN ones(length(N))]
    b = A\loge
    loge_reg = [b[1]*logN[i] + b[2] for i in 1:length(N)]
    plot!(plt, logN, loge_reg, linestyle = :dot, label = string("slope = ", round(b[1], sigdigits=3)))
    savefig(plt, "integratio_convergence.pdf")
    print("slope = ", b[1])
end

end