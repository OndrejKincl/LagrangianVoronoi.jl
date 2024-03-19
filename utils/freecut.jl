const CROP_TOL = 1e-12
const LIMITER_EDGES = 6

function tri_area(a::RealVector, b::RealVector, c::RealVector)::Float64
    return 0.5*abs(LagrangianVoronoi.cross2(b - a, c - a))
end

function arc_len(a::RealVector, b::RealVector, c::RealVector, r::Float64)::Float64
    sin_alpha = 2.0*tri_area(a, b, c)/(norm(b - a)*norm(c - a))
    sin_alpha = (isnan(sin_alpha) ? 0.0 : sin_alpha)
    sin_alpha = max(sin_alpha, -1.0)
    sin_alpha = min(sin_alpha,  1.0)
    alpha = asin(sin_alpha)
    return r*alpha
end

function arc_area(a::RealVector, b::RealVector, c::RealVector, r::Float64)::Float64
    return 0.5*r*arc_len(a, b, c, r)
end

function circ_crop(p::VoronoiPolygon, e::Edge)::Edge
    l = len(e)
    if l < CROP_TOL
        return e
    end
    t = (e.v2 - e.v1)/l # tangent vector
    n = -t[2]*VECX + t[1]*VECY # normal vector
    m = 0.5*e.v1 + 0.5*e.v2
    z = p.x + dot(m - p.x, n)*n # nearest point on e (extended)
    d2 = p.var.crop_R^2 - norm_squared(z - p.x)
    if d2 > 0
        d = sqrt(d2)
        a = max(dot(e.v1 - z, t), -d)
        b = min(dot(e.v2 - z, t),  d)
        if (a < b)
            return Edge(z + a*t, z + b*t, label = e.label)
        end
    end
    return Edge(m, m, label = e.label)
end

function free_length(p::VoronoiPolygon, e::Edge)::Float64
    if p.var.isfree
        return len(circ_crop(p, e))
    end
    return len(e)
end

function free_arclength(p::VoronoiPolygon, e::Edge)::Float64
    if p.var.isfree
        f = circ_crop(p, e)
        return arc_len(p.x, e.v1, f.v1, p.var.crop_R) + arc_len(p.x, f.v2, e.v2, p.var.crop_R)
    end
    return 0.0
end

function free_area(p::VoronoiPolygon)::Float64
    if is_free_polygon(p)
        A = 0.0
        for e in p.edges
            f = circ_crop(p, e)
            A += arc_area(p.x, e.v1, f.v1, p.var.crop_R)
            A += tri_area(p.x, f.v1, f.v2)
            A += arc_area(p.x, f.v2, e.v2, p.var.crop_R)
        end
        return A
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
                y = p.x + 2*p.var.crop_R*RealVector(cos(theta), sin(theta))
                LagrangianVoronoi.voronoicut!(p, y, -1)
            end
        end
        p.isbroken = false
    end
end