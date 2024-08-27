const RealVector = SVector{2, Float64}
const RealMatrix = SMatrix{2, 2, Float64, 4}

const VEC0 = RealVector(0.0, 0.0)
const VECX = RealVector(1.0, 0.0)
const VECY = RealVector(0.0, 1.0)
const VECNULL = RealVector(NaN, NaN)
const MAT0 = RealMatrix(0.0, 0.0, 0.0, 0.0)
const MAT1 = RealMatrix(1.0, 0.0, 0.0, 1.0)

function isnullvector(x::RealVector)::Bool
    return isnan(x[1]) && isnan(x[2])
end

# cross product in 2d
# returns a scalar equal to the z component of the corresponding 3d cross product
@inbounds function cross2(a::RealVector, b::RealVector)
    return (a[1]*b[2] - a[2]*b[1])
end

struct Edge
    v1::RealVector        # vertex
    v2::RealVector        # vertex
    label::Int            # use label to indicate a neighbor
    Edge(v1::RealVector, v2::RealVector; label::Int = 0) = new(v1, v2, label)
end

struct Rectangle
    # corners of the rectangle
    xmin::RealVector
    xmax::RealVector
end

UnitRectangle() = Rectangle(VEC0, VECX + VECY)

Rectangle(; xlims::Tuple{Number, Number}, ylims::Tuple{Number, Number}) = begin
    xmin = RealVector(xlims[1], ylims[1])
    xmax = RealVector(xlims[2], ylims[2])
    return Rectangle(xmin, xmax)
end

function area(r::Rectangle)::Float64
    return abs(r.xmax[1] - r.xmin[1])*abs(r.xmax[2] - r.xmin[2])
end

function isinside(r::Rectangle, x::RealVector)::Bool
    return (r.xmin[1] <= x[1] <= r.xmax[1]) && (r.xmin[2] <= x[2] <= r.xmax[2])
end

function len(e::Edge)::Float64
    return norm(e.v1 - e.v2)
end

function norm_squared(x::RealVector)::Float64
    return dot(x, x)
end

function verts(r::Rectangle)::NTuple{4, RealVector}
    return (
        RealVector(r.xmin[1], r.xmin[2]),
        RealVector(r.xmin[1], r.xmax[2]),
        RealVector(r.xmax[1], r.xmax[2]),
        RealVector(r.xmax[1], r.xmin[2])
    )
end

function outer(x::RealVector, y::RealVector)::RealMatrix
    return RealMatrix(x[1]*y[1], x[2]*y[1], x[1]*y[2], x[2]*y[2])
end