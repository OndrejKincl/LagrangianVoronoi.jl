# Voronoi Polygon for compressible Navier-Stokes equations

@kwdef mutable struct PolygonNSc <: VoronoiPolygon
    x::RealVector        # position
    v::RealVector = VEC0 # velocity
    s::Float64 = 0.0     # entropy
    h::Float64 = 0.0     # enthalpy
    P::Float64 = 0.0
    mass::Float64 = 0.0
    area::Float64 = 0.0
    rho::Float64 = 0.0
    c2::Float64 = 0.0    # sound speed squared
    # sides of the polygon (in no particular order)
    edges::FastVector{Edge} = emptypolygon()
end

PolygonNSc(x::RealVector)::PolygonNSc = PolygonNSc(x=x)
const GridNSc = VoronoiGrid{PolygonNSc}