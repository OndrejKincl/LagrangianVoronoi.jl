# Voronoi Polygon for compressible Navier-Stokes equations

@kwdef mutable struct PolygonNSc <: VoronoiPolygon
    x::RealVector        # position
    v::RealVector = VEC0 # velocity
    a::RealVector = VEC0 # acceleration
    P::Float64    = 0.0  # pressure
    rho::Float64  = 0.0  # density
    e::Float64    = 0.0  # specific energy
    c2::Float64   = 0.0  # speed of sound squared

    # extensive vars
    mass::Float64 = 0.0
    momentum::RealVector = VEC0
    energy::Float64 = 0.0
    area::Float64 = 0.0

    D::RealMatrix = MAT0 # velocity deformation tensor
    mu::Float64 = 0.0    # dynamic viscosity

    # sides of the polygon (in no particular order)
    edges::FastVector{Edge} = emptypolygon()
end

PolygonNSc(x::RealVector)::PolygonNSc = PolygonNSc(x=x)
const GridNSc = VoronoiGrid{PolygonNSc}