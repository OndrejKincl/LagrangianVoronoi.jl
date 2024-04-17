include("geometry.jl")

const POLYGON_SIZEHINT = 10
const SIGNUM_EPS = 2*eps(Float64)

@inline function signum(x::Float64)::Int
    if (x < -SIGNUM_EPS)
        return -1
    elseif (x > SIGNUM_EPS)
        return 1
    end
    return 0
end

mutable struct VoronoiPolygon{T}
    x::RealVector # position
    v::RealVector # velocity
    a::RealVector # acceleration
    P::Float64    # pressure
    rho::Float64  # density
    mass::Float64 # mass (duh)
    var::T        # user-defined variables
    edges::PreAllocVector{Edge}  # sides of the polygon (in no particular order)
    VoronoiPolygon{T}(x::RealVector) where T = new{T}(
        x,
        VEC0,
        VEC0,
        0.0,
        0.0,
        0.0,
        T(),
        PreAllocVector{Edge}(POLYGON_SIZEHINT),
    )
end

# when you have no user-defined variables
const VanillaPolygon = VoronoiPolygon{Nothing}


@inbounds function reset!(p::VoronoiPolygon, boundary_rect::Rectangle)
    empty!(p.edges)
    A = boundary_rect.xmin
    C = boundary_rect.xmax
    B = RealVector(C[1], A[2])
    D = RealVector(A[1], C[2])
    push!(p.edges, Edge(B, A))
    push!(p.edges, Edge(A, D))
    push!(p.edges, Edge(D, C))
    push!(p.edges, Edge(C, B))
end

# intersects a VoronoiPolygon with the halfplane of all points closer to y than to p.x
# use label for newly created edge
# this code has very sharp edges
@inbounds function voronoicut!(p::VoronoiPolygon, y::RealVector, label::Int)::Bool
    diff = y - p.x
    mid = 0.5*(y + p.x)
    c = dot(diff, mid)
    i = 1
    # points of the new edge to be created
    X = VECNULL
    Y = VECNULL
    while (i <= length(p.edges))
        # intersect edge with a halfplane H = {x: f(x) <= 0}
        e = p.edges[i]
        f1 = dot(diff, e.v1) - c
        f2 = dot(diff, e.v2) - c
        s1 = signum(f1)
        s2 = signum(f2)
        s12 = s1 + s2
        # at least one vertex is out:
        if (0 <= s12 <= 1) && (s1|s2 != 0)
            Y = X
            # the intersector of edge end the boundary of H
            X = 1.0/(f1 - f2)*(f1*e.v2 - f2*e.v1)
            # preferably, return one of vertices exactly
            X = (s1 == 0 ? e.v1 : X)
            X = (s2 == 0 ? e.v2 : X)
            # cut the edge
            e_cut = (s1 == 1 ? Edge(X, e.v2, label = e.label) : Edge(e.v1, X, label = e.label))
            p.edges[i] = e_cut
        end
        # delete the edge if it is almost entirely in complement of H 
        if (1 <= s12)
            deleteat!(p.edges, i)
        else
            i += 1
        end
    end
    # if intersected, add the new edge and return true
    if !isnullvector(Y) && (X != Y)
        e = Edge(X, Y, label = label)
        # reorient the edge so it goes counter-clockwise
        if cross2(Y - X, p.x - X) > 0.0
            e = invert(e)
        end 
        push!(p.edges, e)
        return true
    end
    return false
end

function influence_rr(p::VoronoiPolygon)::Float64
    rr = 0.0
    for e in p.edges
        rr = max(rr, 4.0*norm_squared(e.v1 - p.x))
    end
    return rr
end

function area(p::VoronoiPolygon)::Float64
    A = 0.0
    for e in p.edges
        # we need to use abs because the edges may not 
        # have the right orientation
        A += 0.5*abs(cross2(e.v1 - p.x, e.v2 - p.x))
    end
    return A
end

function isboundary(e::Edge)
    return e.label <= 0
end

function isboundary(p::VoronoiPolygon)
    for e in p.edges
        if isboundary(e)
            return true
        end
    end
    return false
end

@inbounds function normal_vector(e::Edge)
    n = RealVector(e.v1[2]-e.v2[2], e.v2[1]-e.v1[1])
    return n/norm(n)
end

@inbounds function surface_element(p::VoronoiPolygon)
    dS = VEC0
    for e in p.edges
        if isboundary(e)
            dS += RealVector(e.v1[2]-e.v2[2], e.v2[1]-e.v1[1])
        end
    end
    return dS
end


@inbounds function is_inside(p::VoronoiPolygon, x::RealVector)::Bool
    # Sunday algorithm
    wn = 0
    for e in p.edges
        isleft = (
              (e.v2[1] - e.v1[1])*(x[2] - e.v1[2])
            - (x[1] - e.v1[1])*(e.v2[2] - e.v1[2])
        )
        if (e.v1[2] <= x2 < e.v2[2]) && (isleft > 0.)
            wn += 1
        end
        if (e.v1[2] > x2 >= e.v2[2]) && (isleft < 0.)
            wn -= 1
        end
    end
    return wn != 0
end

function tri_area(a::RealVector, b::RealVector, c::RealVector)::Float64
    return 0.5*abs(LagrangianVoronoi.cross2(b - a, c - a))
end

function centroid(p::VoronoiPolygon)::RealVector
    A = 0.0
    c = VEC0
    for e in p.edges
        dA = tri_area(p.x, e.v1, e.v2)
        A += dA
        c += dA*(p.x + e.v1 + e.v2)/3
    end
    return c/A
end

function lr_ratio(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)::Float64
    l2 = norm_squared(e.v1 - e.v2)
    r2 = norm_squared(p.x - q.x)
    return sqrt(l2/r2)
end