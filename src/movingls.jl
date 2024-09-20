"""
    LinearExpansion

Static vector to store coefficients of a linear Taylor expansion.
"""
const LinearExpansion = SVector{2, Float64}

"""
    QuadraticExpansion

Static vector to store coefficients of a quadratic Taylor expansion.
"""
const QuadraticExpansion = SVector{5, Float64}

"""
    CubicExpansion

Static vector to store coefficients of a cubic Taylor expansion.
"""
const CubicExpansion = SVector{9, Float64}

function gaussian_kernel(h::Float64, r::Float64)
    return exp(-0.5*(r/h)^2)/sqrt(2*pi*h^2)
end
 
function power_vector(::Type{LinearExpansion}, x::RealVector)::LinearExpansion
    return LinearExpansion(x[1], x[2])
end

function power_vector(::Type{QuadraticExpansion}, x::RealVector)::QuadraticExpansion
    return QuadraticExpansion(x[1], x[2], x[1]*x[1], x[1]*x[2], x[2]*x[2])
end

function power_vector(::Type{CubicExpansion}, x::RealVector)::CubicExpansion
    return CubicExpansion(x[1], x[2], x[1]*x[1], x[1]*x[2], x[2]*x[2], x[1]*x[1]*x[1], x[1]*x[1]*x[2], x[1]*x[2]*x[2], x[2]*x[2]*x[2])
end

"""
    movingls(::Type{T}, grid::VoronoiGrid, p::VoronoiPolygon, fun; h,  kernel)

Finds the Taylor expansion of a given function using moving least squares. 
The first argument `T` should be one of following:

* LinearExpansion
* QuadraticExpansion
* CubicExpansion

Polygon `p` is where the Taylor expansion of a function `fun` is computed. A keyword argument `h` is the moving radius.
This needs to be sufficiently big, otherwise the Taylor expension may be undefined. Kernel is the weighting function
used for defining the (weighted) least squared problem.
"""
function movingls(
        ::Type{T},
        grid::VoronoiGrid, p::VoronoiPolygon, fun::Function; 
        h::Number = grid.h,  kernel::Function = gaussian_kernel
    )::T where {T <: SVector}
    fp = fun(p)
    a = zero(T)
    R = a*a'
    key0 = findkey(grid.cell_list, p.x)
    for node in grid.cell_list.magic_path
        if (node.rr > h)
            break
        end
        key = key0 + node.key
        if !(checkbounds(Bool, grid.cell_list.cells, key))
            continue
        end
        for i in grid.cell_list.cells[key]
            q = grid.polygons[i]
            y = p.x + get_arrow(q.x, p.x, grid)
            if (p.x == y)
                continue
            end
            r = norm(p.x-y)
            if r < h
                ker = kernel(h, r)
                fq = fun(q)
                dx = power_vector(T, y - p.x)
                a += ker*(fq - fp)*dx
                R += ker*dx*dx'
            end
        end
    end
    a = R\a
    return a
end

"""
    poly_eval(val::Float64, taylor::T, dx::RealVector)::Float64 where {T <: SVector}

Use this function to interpolate a value from the known Taylor expansion.
"""
function poly_eval(val::Float64, taylor::T, dx::RealVector)::Float64 where {T <: SVector}
    return val + dot(taylor, power_vector(T, dx))
end

"""
    integral(p::VoronoiPolygon, val::Float64, taylor::SVector)::Float64

Compute an integral of a polynomial over polygon `p` defined by value `val` at `p.x` and the 
Taylor expansion `taylor`.
"""
function integral(p::VoronoiPolygon, val::Float64, taylor::LinearExpansion)::Float64
    int = 0.0
    for e in p.edges
        A = 0.5*abs(cross2(e.v1 - p.x, e.v2 - p.x))
        int += A*poly_eval(val, taylor, -2/3*p.x + 1/3*e.v1 + 1/3*e.v2)
    end
    return int
end

function integral(p::VoronoiPolygon, val::Float64, taylor::QuadraticExpansion)::Float64
    int = 0.0
    for e in p.edges
        A = 0.5*abs(cross2(e.v1 - p.x, e.v2 - p.x))
        int += (A/3)*poly_eval(val, taylor, -1/3*p.x + 1/6*e.v1 + 1/6*e.v2)
        int += (A/3)*poly_eval(val, taylor, -5/6*p.x + 2/3*e.v1 + 1/6*e.v2)
        int += (A/3)*poly_eval(val, taylor, -5/6*p.x + 1/6*e.v1 + 2/3*e.v2)
    end
    return int
end

function integral(p::VoronoiPolygon, val::Float64, taylor::CubicExpansion)::Float64
    int = 0.0
    for e in p.edges
        A = 0.5*abs(cross2(e.v1 - p.x, e.v2 - p.x))
        int -=  (9*A/16)*poly_eval(val, taylor, -2/3*p.x + 1/3*e.v1 + 1/3*e.v2)
        int += (25*A/48)*poly_eval(val, taylor, -2/5*p.x + 1/5*e.v1 + 1/5*e.v2)
        int += (25*A/48)*poly_eval(val, taylor, -4/5*p.x + 3/5*e.v1 + 1/5*e.v2)
        int += (25*A/48)*poly_eval(val, taylor, -4/5*p.x + 1/5*e.v1 + 3/5*e.v2)
    end
    return int
end

"""
    point_value(grid::VoronoiGrid, x::RealVector, fun::Function)

Obtain the interpolation of a function `fun` defined on polygons at an arbitrary point `x`.
This is a slow code and should never be used in perfomance-critical areas.
"""
function point_value(grid::VoronoiGrid, x::RealVector, fun::Function)
    p = nearest_polygon(grid, x)
    L = movingls(LinearExpansion, grid, p, fun) 
    return fun(p) + dot(L, x - p.x)
end