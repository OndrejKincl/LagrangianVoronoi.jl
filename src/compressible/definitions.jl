# Voronoi Polygon for compressible Navier-Stokes equations

@kwdef mutable struct PolygonNSc <: VoronoiPolygon
    x::RealVector        # position
    v::RealVector = VEC0 # velocity
    a::RealVector = VEC0 # acceleration
    P::Float64    = 0.0  # pressure
    rho::Float64  = 0.0  # density
    mass::Float64 = 0.0
    c::Float64    = 0.0  # sound speed
    #s::Float64    = 0.0  # entropy
    e::Float64    = 0.0
    mu::Float64   = 0.0  # dynamic viscosity
    D::RealMatrix = MAT0 # velocity deformation tensor
    # sides of the polygon (in no particular order)
    edges::FastVector{Edge} = emptypolygon()
end

PolygonNSc(x::RealVector)::PolygonNSc = PolygonNSc(x=x)
const GridNSc = VoronoiGrid{PolygonNSc}