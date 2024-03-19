const RealVector = SVector{2, Float64}
const RealMatrix = SMatrix{2, 2, Float64, 4}

const VEC0 = RealVector(0.0, 0.0)
const VECX = RealVector(1.0, 0.0)
const VECY = RealVector(0.0, 1.0)
const VECNULL = RealVector(NaN, NaN)

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
    lr_ratio::Float64
    Edge(v1::RealVector, v2::RealVector; label::Int = 0, lr_ratio::Float64 = 0.0) = new(v1, v2, label, lr_ratio)
end

struct Rectangle
    # corners of the rectangle
    xmin::RealVector
    xmax::RealVector
    Rectangle(; xlims::Tuple{Number, Number}, ylims::Tuple{Number, Number}) = begin
        xmin = RealVector(xlims[1], ylims[1])
        xmax = RealVector(xlims[2], ylims[2])
        if (xmin[1] >= xmax[1]) || (xmin[2] >= xmax[2])
            throw(DomainError("Rectangle with inverted bounds."))
        end
        return new(xmin, xmax)
    end
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