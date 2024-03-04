const LinearExpansion = SVector{2, Float64}
const QuadraticExpansion = SVector{5, Float64}
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

function ls_reconstruction(
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
            if (p.x == q.x)
                continue
            end
            r = norm(p.x - q.x)
            if r < h
                ker = kernel(h, r)
                fq = fun(q)
                dx = power_vector(T, q.x - p.x)
                a += ker*(fq - fp)*dx
                R += ker*dx*dx'
            end
        end
    end
    a = R\a
    return a
end

function poly_eval(val::Float64, taylor::T, dx::RealVector)::Float64 where {T <: SVector}
    return val + dot(taylor, power_vector(T, dx))
end

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