# requires these constants:
#
# rho, dt


# requires these fields:
# 
# mass::Float64
# v::RealVector
# p::Float64
# invA::Float65
# isdirichlet::Bool
# L::RealMatrix
# 

function get_invA!(p::VoronoiPolygon)
    A = area(p)
    p.var.invA = 1/A
    p.var.div = 0.0
    p.var.a = VEC0
end

function lr_ratio(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)::Float64
    l2 = norm_squared(e.v1 - e.v2)
    r2 = norm_squared(p.x - q.x)
    return sqrt(l2/r2)
end

function edge_element(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)::Float64
    return p.var.isdirichlet ? 0.0 : -p.var.invA*lr_ratio(p,q,e)
end

function diagonal_element(grid::VoronoiGrid, p::VoronoiPolygon)::Float64
    return p.var.isdirichlet ? 1.0 : sum(e -> e.label == 0 ? 0.0 : p.var.invA*lr_ratio(p, grid.polygons[e.label], e), p.edges, init = 0.0)
end

@inline function v_interpolant(p::VoronoiPolygon, x::RealVector)::RealVector
    return p.var.v + p.var.L*(x - p.x)
end


function vector_element(grid::VoronoiGrid, p::VoronoiPolygon)::Float64
    div = 0.0
    for e in p.edges
        q = p
        if !isboundary(e)
            q = grid.polygons[e.label]
        end
        n = normal_vector(e)
        v1 = v_interpolant(p, e.v1)
        v12 = v_interpolant(p, 0.5*e.v1 + 0.5*e.v2)
        v2 = v_interpolant(p, e.v2)
        v_n = dot(n, v1 + 4*v12 + v2)
        v_n = min(v_n, max(dot(n, p.var.v), dot(n, q.var.v)))
        v_n = max(v_n, min(dot(n, p.var.v), dot(n, q.var.v)))
        div += p.var.invA*len(e)/6*v_n
    end
    p.var.div = div
    if p.var.isdirichlet
        return 0.0 #p.var.P
    end
    return -(rho/dt)*div
end

# the limiter decreases variation of L so that the min/max dv inside the cell
# cannot exceed the min/max dv counted from all neighbors
# this prevents the generation of new maxima
function limiter!(grid)
    for p in grid.polygons
        # determine min and max value of vx and vy
        dv_min =  VEC0
        dv_max =  VEC0
        for e in p.edges
            if !isboundary(e)
                q = grid.polygons[e.label]
                dv = q.var.v - p.var.v
                dv_min = RealVector(min(dv_min[1], dv[1]), min(dv_min[2], dv[2]))
                dv_max = RealVector(max(dv_max[1], dv[1]), max(dv_max[2], dv[2]))
            end
        end
        # find 0 <= eps_i <= 1 such that eps_i*dv[i] inside the cell is always between dv_min and dv_max
        eps_1 = 1.0
        eps_2 = 1.0
        for e in p.edges
            dv = v_interpolant(p, e.v1) - p.var.v
            if dv[1] > dv_max[1] 
                eps_1 = min(eps_1, dv_max[1]/dv[1])
            end
            if dv[1] < dv_min[1]
                eps_1 = min(eps_1, dv_min[1]/dv[1])
            end
            if dv[2] > dv_max[2] 
                eps_2 = min(eps_2, dv_max[2]/dv[2])
            end
            if dv[2] < dv_min[2]
                eps_2 = min(eps_2, dv_min[2]/dv[2])
            end
        end
        # finally, downscale L by left mult
        @assert (0.0 <= eps_1 <= 1.0)
        @assert (0.0 <= eps_2 <= 1.0)
        p.var.L = RealMatrix(eps_1, 0, eps_2, 0)*p.var.L 
    end
end

# e1     A11 A12
#    e2  A21 A22

#=
function get_div!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    p.var.div += (p.var.invA)*lr_ratio(p,q,e)*dot(m - q.x, p.var.v - q.var.v)
end

function vector_element(p::VoronoiPolygon)::Float64
    return p.var.isdirichlet ? 0.0 : -(rho/dt)*p.var.div
end
=#



@inbounds function outer(x::RealVector, y::RealVector)::RealMatrix
    return RealMatrix(x[1]*y[1], x[2]*y[1], x[1]*y[2], x[2]*y[2])
end

function get_L!(grid::VoronoiGrid)
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

# pressure force
function get_a!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    z = 0.5*(p.x + q.x)
    p.var.a += -(1.0/p.var.mass)*lr_ratio(p,q,e)*(
        (q.var.P - p.var.P)*(m - z) 
        - 0.5*(p.var.P + q.var.P)*(p.x - q.x)
    )
end

function get_pressure_acc!(grid::VoronoiGrid)
    apply_unary!(grid, get_invA!)
    get_L!(grid)
    #limiter!(grid)
    #apply_binary!(grid, get_div!)
    A = assemble_matrix(grid, diagonal_element, edge_element; fix_avg = true)
    b = assemble_vector(grid, vector_element; fix_avg = true)
    pressure_vector = A\b
    @threads for i in eachindex(grid.polygons)
        p = grid.polygons[i]
        p.var.P = pressure_vector[i]
    end
    apply_binary!(grid, get_a!)
end