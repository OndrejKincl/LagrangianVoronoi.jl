# Voronoi Polygon for Navier-Stokes equations

@kwdef mutable struct PolygonNS <: VoronoiPolygon
    x::RealVector        # position
    v::RealVector = VEC0 # velocity
    a::RealVector = VEC0 # acceleration
    P::Float64    = 0.0  # pressure
    rho::Float64  = 0.0  # density
    mass::Float64 = 0.0
    # sides of the polygon (in no particular order)
    edges::Vector{Edge} = emptypolygon()
end

PolygonNS(x::RealVector)::PolygonNS = PolygonNS(x=x)
const GridNS = VoronoiGrid{PolygonNS}