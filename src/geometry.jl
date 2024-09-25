"""
	RealVector(x1::Float64, x2::Float64)

Static Float64 vector with 2 elements.
"""
const RealVector = SVector{2, Float64}

"""
	RealMatrix(x11::Float64, x21::Float64, x12::Float64, x22::Float64)

Static Float64 matrix with 2x2 elements. Be warned that Julia has column-major ordering of elements!
"""
const RealMatrix = SMatrix{2, 2, Float64, 4}

"""
	VEC0

Static zero vector. Equivalent to `zero(RealVector)`.
"""
const VEC0 = RealVector(0.0, 0.0)

"""
	VECX

Static cartesian basis vector in the X direction. Equivalent to `RealVector(1,0)`
"""
const VECX = RealVector(1.0, 0.0)

"""
	VECY

Static cartesian basis vector in the Y direction. Equivalent to `RealVector(0,1)`.
"""
const VECY = RealVector(0.0, 1.0)

"""
    VECNULL

Undefined vector.
"""
const VECNULL = RealVector(NaN, NaN)

"""
    MAT0

A static 2x2 zero matrix.
"""
const MAT0 = RealMatrix(0.0, 0.0, 0.0, 0.0)

"""
    MAT1

A static 2x2 identity matrix.
"""
const MAT1 = RealMatrix(1.0, 0.0, 0.0, 1.0)

"""
    isnullvector(x::RealVector)::Bool

Returns true iff the vector is VECNULL.
"""
function isnullvector(x::RealVector)::Bool
    return isnan(x[1]) && isnan(x[2])
end

"""
    function cross2(a::RealVector, b::RealVector)::Float64

A cross product in 2d.
Returns a scalar equal to the z component of the corresponding 3d cross product
"""
function cross2(a::RealVector, b::RealVector)::Float64
    @inbounds return (a[1]*b[2] - a[2]*b[1])
end

"""
    Edge(v1::RealVector, v2::RealVector; label::Int = 0)

Edge defined by two endpoints and a label which indicates the neighbor's index. Zero or negative indices are used for domain boundaries. 
Edges are stack-allocated and immutable. 
"""
struct Edge
    v1::RealVector        # vertex
    v2::RealVector        # vertex
    label::Int            # use label to indicate a neighbor
    Edge(v1::RealVector, v2::RealVector; label::Int = 0) = new(v1, v2, label)
end

"""
    Rectangle(xmin::RealVector, xmax::RealVector)

A rectangle aligned with the coordinate system defined by bottomleft and topright corner. 
"""
struct Rectangle
    # corners of the rectangle
    xmin::RealVector
    xmax::RealVector
end

"""
    UnitRectangle()

The unit rectange (0,1)x(0,1).
"""
UnitRectangle() = Rectangle(VEC0, VECX + VECY)

Rectangle(; xlims::Tuple{Number, Number}, ylims::Tuple{Number, Number}) = begin
    xmin = RealVector(xlims[1], ylims[1])
    xmax = RealVector(xlims[2], ylims[2])
    return Rectangle(xmin, xmax)
end

"""
    area(r::Rectangle)::Float64

Unoriented area of a rectangle.
"""
function area(r::Rectangle)::Float64
    return abs(r.xmax[1] - r.xmin[1])*abs(r.xmax[2] - r.xmin[2])
end

"""
    isinside(r::Rectangle, x::RealVector)::Bool

Return `true` iff `x` lies inside `r`.
"""
function isinside(r::Rectangle, x::RealVector)::Bool
    return (r.xmin[1] <= x[1] <= r.xmax[1]) && (r.xmin[2] <= x[2] <= r.xmax[2])
end

"""
    len(e::Edge)::Float64

Return the length of an edge.
"""
function len(e::Edge)::Float64
    return norm(e.v1 - e.v2)
end

"""
    midpoint(e::Edge)::RealVector

The edge midpoint.
"""
function midpoint(e::Edge)::RealVector
    return 0.5*(e.v1 + e.v2)
end

"""
    midpoint(x::RealVector, y::RealVector)::RealVector

Midpoint inbetween two points.
"""
function midpoint(x::RealVector, y::RealVector)::RealVector
    return 0.5*(x + y)
end

"""
    norm_squared(x::RealVector)::Float64

The squared norm of a vector. This is much faster than `norm(x)^2`.
"""
function norm_squared(x::RealVector)::Float64
    return dot(x, x)
end

"""
    verts(r::Rectangle)::NTuple{4, RealVector}

Return the four vertices of a Rectangle in counter-clockwise order, starting with the bottomleft corner.
"""
function verts(r::Rectangle)::NTuple{4, RealVector}
    return (
        RealVector(r.xmin[1], r.xmin[2]),
        RealVector(r.xmin[1], r.xmax[2]),
        RealVector(r.xmax[1], r.xmax[2]),
        RealVector(r.xmax[1], r.xmin[2])
    )
end

"""
    outer(x::RealVector, y::RealVector)::RealMatrix

The outer product of two vectors.
"""
function outer(x::RealVector, y::RealVector)::RealMatrix
    return RealMatrix(x[1]*y[1], x[2]*y[1], x[1]*y[2], x[2]*y[2])
end