module divergence

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using WriteVTK, LinearAlgebra, Random

const DOMAIN = Rectangle(xlims = (0, 1), ylims = (0, 1))
const dr = 1e-2

mutable struct PhysFields
    v::RealVector
    div::Float64
    div0::Float64
    err::Float64
    invA::Float64
    PhysFields(x::RealVector) = new(init_vel(x), 0.0, div_exact(x), Inf)
end



function init_vel(x::RealVector)::RealVector
    return x[1]*x[2]*(1.0 - x[1])*(1.0 - x[2])*RealVector(-x[2], x[1])
end

function div_exact(x::RealVector)::Float64
    return x[1]*x[1]*(1.0 - x[1])*(1.0 - 2*x[2]) - x[2]*x[2]*(1.0 - x[2])*(1.0 - 2*x[1])
end

# add points to grid
function populate_vogel!(grid::VoronoiGrid)
    N = pi/(dr*dr)
    for i in 1:N
        r = sqrt(i/N)
        theta = 2.39996322972865332*i
        x = RealVector(0.5 + r*cos(theta), 0.5 + r*sin(theta))
        if isinside(DOMAIN, x)
            push!(grid.polygons, VoronoiPolygon{PhysFields}(x))
        end 
    end
    @show length(grid.polygons)
end


function populate_rand!(grid::VoronoiGrid)
    N = 1.0/(dr*dr)
    Random.seed!(12345)
    for _ in 1:N
        s = rand()
        t = rand()
        x = s*DOMAIN.xmin[1] + (1-s)*DOMAIN.xmax[1]
        y = t*DOMAIN.xmin[2] + (1-t)*DOMAIN.xmax[2]
        x = RealVector(x,y)
        if isinside(DOMAIN, x)
            push!(grid.polygons, VoronoiPolygon{PhysFields}(x))
        end 
    end
    @show length(grid.polygons)
end

function lr_ratio(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)::Float64
    l2 = norm_squared(e.v1 - e.v2)
    r2 = norm_squared(p.x - q.x)
    return sqrt(l2/r2)
end


# estimate divergence
function get_invA!(p::VoronoiPolygon)
    p.var.invA = 1.0/area(p)    
end

function get_div!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    z = 0.5*(p.x + q.x)
    p.var.div += p.var.invA*lr_ratio(p,q,e)*dot(m - q.x, p.var.v - q.var.v)
    #p.var.div += p.var.invA*lr_ratio(p,q,e)*dot(z - q.x, p.var.v - q.var.v)
    #p.var.div += p.var.invA*lr_ratio(p,q,e)*(dot(m - z, p.var.v - q.var.v) - 0.5*dot(p.x - q.x, p.var.v + q.var.v))
end

function get_err!(p::VoronoiPolygon)
    #p.var.err = isboundary(p) ? 0.0 : p.var.div - p.var.div0
    p.var.err = p.var.div - p.var.div0
end

function main()
    grid = VoronoiGrid{PhysFields}(2*dr, DOMAIN)
    populate_vogel!(grid)

    @info "mesh generation"
    remesh!(grid)

    @info "estimating divergence"
    apply_unary!(grid, get_invA!)
    apply_binary!(grid, get_div!)
    apply_unary!(grid, get_err!)

    @info "post processing"
    l2_error = 0.0
    @time for i in eachindex(grid.polygons)
        p = grid.polygons[i]
        l2_error += area(p)*p.var.err^2
    end 
    l2_error = sqrt(l2_error)
    @show l2_error

    @info "exporting the result to a vtp file"
    vtp_file = export_grid(grid, "results/divergence/divergence.vtp", :v, :div, :div0, :err)
    vtk_save(vtp_file)
end


end