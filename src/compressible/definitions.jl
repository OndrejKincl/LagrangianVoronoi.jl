# Voronoi Polygon for compressible Navier-Stokes equations

@kwdef mutable struct PolygonNSc <: VoronoiPolygon
    x::RealVector        # position
    v::RealVector = VEC0 # velocity
    a::RealVector = VEC0
    e::Float64 = 0.0     # energy
    P::Float64 = 0.0     # pressure
    S::RealMatrix = MAT0     # viscous stress
    mass::Float64 = 0.0
    area::Float64 = 0.0
    rho::Float64 = 0.0
    tau::Float64 = 0.0
    c2::Float64 = 0.0    # sound speed squared
    # sides of the polygon (in no particular order)
    edges::FastVector{Edge} = emptypolygon()
end

PolygonNSc(x::RealVector)::PolygonNSc = PolygonNSc(x=x)
const GridNSc = VoronoiGrid{PolygonNSc}