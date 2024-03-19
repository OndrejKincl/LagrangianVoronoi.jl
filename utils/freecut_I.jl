const LIMITER_EDGES = 6
const INTEGRAL_SUBDIVISIONS = 4
const SUPPORT_MULTIPLIER = 1

function char_function(p::VoronoiPolygon, x::RealVector)::Float64
    return norm_squared(x - p.x) < p.var.crop_R^2 ? 1.0 : 0.0
end

function integral_line(e::Edge, fun::Function, n::Int = INTEGRAL_SUBDIVISIONS)
    m = 0.5*(e.v1 + e.v2)
    if n == 0
        l = len(e)
        return l/6*(fun(e.v1) + 4*fun(m) + fun(e.v2))
    else
        e1 = Edge(e.v1, m)
        e2 = Edge(m, e.v2)
        return integral_line(e1, fun, n-1) + integral_line(e2, fun, n-1)
    end
end

function integral_poly(p::VoronoiPolygon, fun::Function)::Float64
    int = 0.0
    for e in p.edges
        int += integral_tri(p.x, e.v1, e.v2, fun)
    end
    return int
end

function integral_tri(a::RealVector, b::RealVector, c::RealVector, fun::Function, n::Int = INTEGRAL_SUBDIVISIONS)::Float64
    if n == 0
        A = tri_area(a, b, c)
        return (A/3)*(fun(2/3*a + 1/6*b + 1/6*c) + fun(1/6*a + 2/3*b + 1/6*c) + fun(1/6*a + 1/6*b + 2/3*c))
    else
        M = (a + b + c)/3
        return integral_tri(a, b, M, fun, n-1) + integral_tri(b, c, M, fun, n-1) + integral_tri(c, a, M, fun, n-1)
    end
end

function tri_area(a::RealVector, b::RealVector, c::RealVector)::Float64
    return 0.5*abs(LagrangianVoronoi.cross2(b - a, c - a))
end

function free_length(p::VoronoiPolygon, e::Edge)::Float64
    if p.var.isfree
        return integral_line(e, x::RealVector -> charfun(p,x))
    end
    return len(e)
end

function free_area(p::VoronoiPolygon)::Float64
    if is_free_polygon(p)
        return integral_poly(p, x::RealVector -> charfun(p,x))
    end
    return area(p)
end

function is_free_polygon(p::VoronoiPolygon)::Bool
    rad = maximum(e -> max(norm(e.v1 - p.x)), p.edges)
    return (rad > p.var.crop_R)
end


function limit_free_polygons!(grid::VoronoiGrid)
    @threads for p in grid.polygons
        p.var.isfree = false
        if is_free_polygon(p)
            p.var.isfree = true
            for k in 1:LIMITER_EDGES
                theta = 2.0*pi*k/LIMITER_EDGES
                y = p.x + 2*SUPPORT_MULTIPLIER*p.var.crop_R*RealVector(cos(theta), sin(theta))
                LagrangianVoronoi.voronoicut!(p, y, 0)
            end
        end
        p.isbroken = false
    end
end