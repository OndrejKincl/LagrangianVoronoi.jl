# Voronoi Polygon for compressible Navier-Stokes equations

@kwdef mutable struct PolygonNSc <: VoronoiPolygon
    x::RealVector        # position

    rho::Float64  = 0.0  # density
    v::RealVector = VEC0 # velocity
    e::Float64    = 0.0  # specific energy
    
    P::Float64    = 0.0  # pressure
    c2::Float64   = 0.0  # speed of sound squared
    D::RealMatrix = MAT0 # velocity deformation tensor
    S::RealMatrix = MAT0 # viscous stress
    mu::Float64 = 0.0    # dynamic viscosity
    dv::RealVector = VEC0

    # extensive vars
    mass::Float64 = 0.0
    momentum::RealVector = VEC0
    energy::Float64 = 0.0
    phase::Int = 0

    # sides of the polygon (in no particular order)
    edges::FastVector{Edge} = emptypolygon()
end

PolygonNSc(x::RealVector)::PolygonNSc = PolygonNSc(x=x)
const GridNSc = VoronoiGrid{PolygonNSc}


# Voronoi Polygon for compressible Navier-Stokes-Fourier equations

@kwdef mutable struct PolygonNSFc <: VoronoiPolygon
    x::RealVector        # position

    rho::Float64  = 0.0  # density
    v::RealVector = VEC0 # velocity
    e::Float64    = 0.0  # specific energy
    
    P::Float64    = 0.0  # pressure
    T::Float64    = 0.0  # temperature
    k::Float64 = 0.0     # heat condictivity
    cV::Float64   = 0.0  # specific heat capacity at constant volume
    c2::Float64   = 0.0  # speed of sound squared
    D::RealMatrix = MAT0 # velocity deformation tensor
    S::RealMatrix = MAT0 # viscous stress
    mu::Float64 = 0.0    # dynamic viscosity
    dv::RealVector = VEC0

    # extensive vars
    mass::Float64 = 0.0
    momentum::RealVector = VEC0
    energy::Float64 = 0.0
    phase::Int = 0

    # sides of the polygon (in no particular order)
    edges::FastVector{Edge} = emptypolygon()
end

PolygonNSFc(x::RealVector)::PolygonNSFc = PolygonNSFc(x=x)
const GridNSFc = VoronoiGrid{PolygonNSFc}

function lr_ratio(dx::RealVector, e::Edge)::Float64
    l2 = norm_squared(e.v1 - e.v2)
    r2 = norm_squared(dx)
    return sqrt(l2/r2)
end

# get vector from x to y
# is equal to x - y on non-periodic domains
function get_arrow(x::RealVector, y::RealVector, grid::VoronoiGrid)
    v = x - y
    if grid.xperiodic && (abs(v[1]) > 0.5*grid.xperiod)
        v -= sign(v[1])*grid.xperiod*VECX
    end
    if grid.yperiodic && (abs(v[2]) > 0.5*grid.yperiod)
        v -= sign(v[2])*grid.yperiod*VECY
    end
    return v
end