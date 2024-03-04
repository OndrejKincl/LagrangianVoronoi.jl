module poisson

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using WriteVTK
using Plots, LaTeXStrings, LinearAlgebra

const DOMAIN = Rectangle(xlims = (0, 1), ylims = (0, 1))

mutable struct PhysFields
    u::Float64
    u0::Float64
    err::Float64
    PhysFields(x::RealVector) = new(0.0, u_exact(x), 0.0)
end


function u_exact(x::RealVector)::Float64
    return 0.5*x[1]*(1.0 - x[1])*x[2]*(1.0 - x[2])
end

function rhs_fun(x::RealVector)::Float64
    return x[1]*(1.0 - x[1]) + x[2]*(1.0 - x[2])
end

# tools to assemble linear system

function lr_ratio(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)::Float64
    l2 = norm_squared(e.v1 - e.v2)
    r2 = norm_squared(p.x - q.x)
    return sqrt(l2/r2)
end

function edge_element(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)::Float64
    return isboundary(p) ? 0.0 : -lr_ratio(p,q,e)
end

function diagonal_element(grid::VoronoiGrid, p::VoronoiPolygon)::Float64
    return isboundary(p) ? 1.0 : sum(e -> lr_ratio(p, grid.polygons[e.label], e), p.edges)
end

function vector_element(p::VoronoiPolygon)::Float64
    return isboundary(p) ? u_exact(p.x) : area(p)*rhs_fun(p.x)
end

# add points to grid

function populate!(grid::VoronoiGrid, dr::Float64)
    #=
    N = pi/(dr*dr)
    for i in 1:N
        r = sqrt(i/N)
        theta = 2.39996322972865332*i
        x = RealVector(0.5 + r*cos(theta), 0.5 + r*sin(theta))
        if isinside(DOMAIN, x)
            push!(grid.polygons, VoronoiPolygon{PhysFields}(x))
        end 
    end
    =#
    a = ((4/3)^0.25)*dr
    b = ((3/4)^0.25)*dr
    i_min = floor(Int, DOMAIN.xmin[1]/a) - 1
    j_min = floor(Int, DOMAIN.xmin[2]/b)
    i_max = ceil(Int, DOMAIN.xmax[1]/a)
    j_max = ceil(Int, DOMAIN.xmax[2]/b)
	for i in i_min:i_max, j in j_min:j_max
        x = RealVector((i + (j%2)/2)*a, j*b)
        if isinside(DOMAIN, x)
            push!(grid.polygons, VoronoiPolygon{PhysFields}(x))
        end
	end
    @show length(grid.polygons)
end

function solve(dr::Float64)
    grid = VoronoiGrid{PhysFields}(2*dr, DOMAIN)
    populate!(grid, dr)

    @info "mesh generation"
    remesh!(grid)

    @info "calling assembler"
    @time begin
        A = assemble_matrix(grid, diagonal_element, edge_element)
        b = assemble_vector(grid, vector_element)
    end

    @info "solving linear system"
    @time u = A\b
    l2_error = 0.0

    @info "post processing"
    @time for i in eachindex(grid.polygons)
        p = grid.polygons[i]
        p.var.u = u[i]
        p.var.err = p.var.u - p.var.u0
        l2_error += area(p)*p.var.err^2
    end 
    l2_error = sqrt(l2_error)
    @show l2_error

    #@info "exporting the result to a vtp file"
    #vtp_file = export_grid(grid, "results/poisson/poisson_cells.vtp", :u, :u0, :err)
    #vtk_save(vtp_file)
    return l2_error
end

function main()
    N = [32, 48, 72, 108, 162, 243]
    e = [solve(1.0/_N) for _N in N]
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
    savefig(plt, "poisson_convergence.pdf")
    print("slope = ", b[1])
end

end