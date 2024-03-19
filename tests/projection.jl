module projection

include("../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using LinearAlgebra
using WriteVTK, StaticArrays, Random
using Plots, LaTeXStrings
using SmoothedParticles:rDwendland2
using Parameters
using Base.Threads

const DOMAIN = Rectangle(xlims = (0,1), ylims = (0,1))
const dt = 1.0
const rho = 1.0
const v0 = 1.0
const TaylorExpansion = LinearExpansion
const export_path = "results/projection"

@with_kw mutable struct PhysFields
    v::RealVector = VEC0
    P::Float64 = 0.0
    invA::Float64 = 0.0
    div::Float64 = 0.0
    div_exact::Float64 = 0.0
    v_exact::RealVector = VEC0
    P_exact::Float64 = 0.0
    mass::Float64 = 0.0
    fixpoint::Bool = false
    isboundary::Bool = false
    L::RealMatrix = zero(RealMatrix)
end

function PhysFields(x::RealVector)::PhysFields
    pf = PhysFields()
    pf.v = v_init(x)
    pf.v_exact = v_exact(x)
    pf.P_exact = P_exact(x)
    pf.div_exact = div_exact(x)
    return pf
end

function v_init(x::RealVector)::RealVector
    vx = -x[1]*(x[1] - 1.0)*(x[2] - 0.5) + v0*cos(x[1])*sin(x[2]) #- pi*cos(pi*x[1])*sin(pi*x[2])
    vy = (x[1] - 0.5)*x[2]*(x[2] - 1.0) + v0*sin(x[1])*cos(x[2]) #- pi*sin(pi*x[1])*cos(pi*x[2])
    return RealVector(vx, vy)
end

function v_exact(x::RealVector)::RealVector
    vx = -x[1]*(x[1] - 1.0)*(x[2] - 0.5)
    vy = (x[1] - 0.5)*x[2]*(x[2] - 1.0)
    return RealVector(vx, vy)
end

function div_exact(x::RealVector)::Float64
    vx_x = -(2*x[1] - 1.0)*(x[2] - 0.5) - v0*sin(x[1])*sin(x[2])
    vy_y =  (x[1] - 0.5)*(2*x[2] - 1.0) - v0*sin(x[1])*sin(x[2])
    return vx_x + vy_y
end

function P_exact(x::RealVector)::Float64
    return v0*rho/dt*(sin(x[1])*sin(x[2])) #-rho/dt*sin(pi*x[1])*sin(pi*x[2])
end

# add points to grid
function populate_vogel!(grid::VoronoiGrid, dr::Float64)
    N = pi/(dr*dr)
    for i in 1:N
        r = sqrt(i/N)
        theta = 2.39996322972865332*i
        x = RealVector(0.5 + r*cos(theta), 0.5 + r*sin(theta))
        if isinside(DOMAIN, x)
            push!(grid.polygons, VoronoiPolygon{PhysFields}(x))
        end 
    end
    #grid.polygons[1].var.fixpoint = true
    @show length(grid.polygons)
end


function populate_rand!(grid::VoronoiGrid, dr::Float64)
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
    grid.polygons[1].var.fixpoint = true
    @show length(grid.polygons)
end

function populate_square!(grid::VoronoiGrid, dr::Float64)
    N = round(Int, 1.0/dr)
    epsilon = dr/100
    step = (1.0 - 2*epsilon)/N
    for i in 0:N
        for j in 0:N
            x = RealVector(epsilon + step*i, epsilon + step*j)
            if isinside(DOMAIN, x)
                push!(grid.polygons, VoronoiPolygon{PhysFields}(x))
                if i == div(N, 2) && j == div(N, 2)
                    grid.polygons[end].var.fixpoint = true
                end
            end
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
    A = area(p)
    p.var.invA = 1.0/A
    p.var.mass = rho*A    
    p.var.div = 0.0
end

function get_div!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    z = 0.5*(p.x + q.x)
    #p.var.div -= lr_ratio(p,q,e)*dot(0.5*(p.x - q.x) + (m-z), p.var.v - q.var.v)
    p.var.div += p.var.invA*lr_ratio(p,q,e)*dot(p.x - m, p.var.v - q.var.v)
    #p.var.div += p.var.invA*lr_ratio(p,q,e)*dot(m - q.x, p.var.v - q.var.v)
    #p.var.div += 0.5*p.var.invA*lr_ratio(p,q,e)*dot(p.x - q.x, p.var.v - q.var.v)
    #p.var.div += p.var.invA*lr_ratio(p,q,e)*dot(m - q.x, p.var.v - q.var.v)
    #p.var.div += lr_ratio(p,q,e)*dot(z - q.x, p.var.v - q.var.v)
    #p.var.div += lr_ratio(p,q,e)*(dot(m - z, p.var.v - q.var.v) - 0.5*dot(p.x - q.x, p.var.v + q.var.v))
end

# tools to assemble linear system

function edge_element(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)::Float64
    return p.var.fixpoint ? 0.0 : -lr_ratio(p,q,e)
end

function diagonal_element(grid::VoronoiGrid, p::VoronoiPolygon)::Float64
    return p.var.fixpoint ? 1.0 : sum(e -> e.label == 0 ? 0.0 : lr_ratio(p, grid.polygons[e.label], e), p.edges)
end

@inline function v_interpolant(p::VoronoiPolygon, x::RealVector)::RealVector
    return p.var.v + p.var.L*(x - p.x)
end

function vector_element(grid::VoronoiGrid, p::VoronoiPolygon)::Float64
    if p.var.fixpoint
        return P_exact(p.x)
    end
    #=
    div = 0.0
    for e in p.edges
        n = normal_vector(e)
        v1 = v_interpolant(p, e.v1)
        v12 = v_interpolant(p, 0.5*e.v1 + 0.5*e.v2)
        v2 = v_interpolant(p, e.v2)
        div += p.var.invA*len(e)/6*dot(n, v1 + 4*v12 + v2)
    end
    =#
    return -(rho/dt)*p.var.div*area(p)
end

# pressure force
function internal_force!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    z = 0.5*(p.x + q.x)
    
    #=
    p.var.v += -(dt/p.var.mass)*lr_ratio(p,q,e)*(
        (q.var.P - p.var.P)*(m - z) 
        - 0.5*(p.var.P + q.var.P)*(p.x - q.x)
    )
    =#
    p.var.v += (dt/p.var.mass)*lr_ratio(p,q,e)*(
        +(p.var.P - q.var.P)*(m - z) 
        + 0.5*(p.var.P + q.var.P)*(p.x - q.x)
    )
    
    #mP = 0.5*poly_eval(p.var.P, p.var.P_taylor, m - p.x) + 0.5*(poly_eval(q.var.P, q.var.P_taylor, m - q.x))
    #p.var.v += (dt/p.var.mass)*lr_ratio(p,q,e)*mP*(p.x - q.x)
end

function wall_force!(p::VoronoiPolygon)
    #=
    for e in p.edges
        if isboundary(e)
            n = normal_vector(e)
            p.var.v += dot(p.var.v, n)*n
        end
    end
    =#
    if isboundary(p)
        p.var.v = p.var.v_exact
    end
end

function filter_fun(p::VoronoiPolygon)::Bool
    return norm(p.x-RealVector(0.5, 0.5)) < 0.4
end

function identify_bdary!(p::VoronoiPolygon)
    p.var.fixpoint = isboundary(p)
end

@inbounds function outer(x::RealVector, y::RealVector)::RealMatrix
    return RealMatrix(x[1]*y[1], x[2]*y[1], x[1]*y[2], x[2]*y[2])
end

function moving_ls!(grid::VoronoiGrid)
    @threads for p in grid.polygons
        p.var.L = zero(RealMatrix)
        R = zero(RealMatrix)
        h = grid.h
        key0 = LagrangianVoronoi.findkey(grid.cell_list, p.x)
        for node in grid.cell_list.magic_path
            if (node.rr > 0.0)
                break
            end
            key = key0 + node.key
            if !(checkbounds(Bool, grid.cell_list.cells, key))
                continue
            end
            for i in grid.cell_list.cells[key]
                q = grid.polygons[i]
                if (p.x == q.x)
                    continue
                end
                dx = q.x - p.x
                rr = dot(dx, dx)
                if rr < h^2
                    r = sqrt(rr)
                    ker = rDwendland2(h, r)
                    dx = q.x - p.x
                    p.var.L += ker*outer(q.var.v - p.var.v, dx)
                    R += ker*outer(dx, dx)
                end
            end
        end
        p.var.L = inv(R)*p.var.L
    end
    return
end

function solve(dr::Float64)
    grid = VoronoiGrid{PhysFields}(2*dr, DOMAIN)
    #populate_rand!(grid, dr)
    #populate_square!(grid, dr)
    populate_vogel!(grid, dr)

    @show dr
    @info "meshing"
    @time remesh!(grid)
    apply_unary!(grid, identify_bdary!)

    #@info "exporting the initial state to a vtp file"
    #vtp_file = export_grid(grid, "results/projection/before_projection.vtp", :v, :P, :div)
    #vtk_save(vtp_file)
    
    @info "moving ls"
    @time moving_ls!(grid)
    
    #apply_binary!(grid, get_div!)
    #l2_div = sqrt(sum(p -> Float64(filter_fun(p))*area(p)*p.var.div^2, grid.polygons))

    @info "assembly"
    @time begin
        apply_unary!(grid, get_invA!)
        #apply_unary!(grid, wall_force!)
        apply_binary!(grid, get_div!)
        A, b = assemble_system(grid, diagonal_element, edge_element, vector_element)
    end

    @info "lin sys solving"
    @time pressure_vector = A\b
    
    @info "pressure push"
    @time begin
        for i in eachindex(grid.polygons)
            p = grid.polygons[i]
            p.var.P = pressure_vector[p.id]
            if isboundary(p)
                p.var.P = P_exact(p.x)
                p.var.div = div_exact(p.x)
            end
        end
        apply_binary!(grid, internal_force!)
        #apply_unary!(grid, wall_force!)
    end

    p_err = 0.0
    v_err = 0.0
    for p in grid.polygons
        p_err += Float64(filter_fun(p))*area(p)*abs(p.var.P - p.var.P_exact)^2
        v_err += Float64(filter_fun(p))*area(p)*norm(p.var.v - p.var.v_exact)^2
    end
    p_err = sqrt(p_err)
    v_err = sqrt(v_err)
    @show p_err
    @show v_err
    return (grid, p_err, v_err)



    #=
    @info "estimating divergence"
    
    apply_binary!(grid, get_div!)
    l2_div = sqrt(sum(p -> Float64(filter_fun(p))*area(p)*p.var.div^2, grid.polygons))
    @show l2_div

    @info "exporting the result to a vtp file"
    vtp_file = export_grid(grid, "results/projection/after_projection.vtp", :v, :P, :div, :P_exact, :v_exact, :init_vel, :isboundary)
    vtk_save(vtp_file)
    =#
end

function main()
    if !ispath(export_path)
        mkpath(export_path)
        @info "created a new path \""*export_path*"\""
    end 
    pvd = paraview_collection(export_path*"/cells.pvd")
    Ns = [32, 48, 72, 108, 162, 243]
    P_errs = []
    v_errs = []
    for N in Ns
        grid, P_err, v_err = solve(1.0/N)
        push!(P_errs, P_err)
        push!(v_errs, v_err)
        pvd[N] = export_grid(grid, string(export_path, "/frame", N, ".vtp"), :v, :P, :v_exact, :P_exact, :div_exact, :div)
    end
    vtk_save(pvd)
    logNs = log10.(Ns)
    logP_errs = log10.(P_errs)
    logv_errs = log10.(v_errs)
    plt = plot(
        logNs, [logP_errs logv_errs], 
        axis_ratio = 1, 
        xlabel = L"\log \, N", ylabel = L"\log \, \epsilon", 
        markershape = :hex, 
        label = ["pressure" "velocity"]
    )
    # linear regression
    A = [logNs ones(length(Ns))]
    b_P = A\logP_errs
    logP_errs_reg = [b_P[1]*logNs[i] + b_P[2] for i in 1:length(Ns)]
    b_v = A\logv_errs
    logv_errs_reg = [b_v[1]*logNs[i] + b_v[2] for i in 1:length(Ns)]
    plot!(plt, logNs, logP_errs_reg, linestyle = :dot, label = string("p slope = ", round(b_P[1], sigdigits=3)))
    plot!(plt, logNs, logv_errs_reg, linestyle = :dot, label = string("v slope = ", round(b_v[1], sigdigits=3)))
    savefig(plt, "helmholtz_convergence.pdf")
    println("P slope = ", b_P[1])
    println("v slope = ", b_v[1])
    
end

end