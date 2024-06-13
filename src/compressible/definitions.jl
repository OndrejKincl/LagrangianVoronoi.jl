# Voronoi Polygon for compressible Navier-Stokes equations

@kwdef mutable struct PolygonNSc <: VoronoiPolygon
    x::RealVector         # position
    v::RealVector = VEC0  # velocity
    dv::RealVector = VEC0 # nodal velocity
    e::Float64 = 0.0      # specific energy
    
    M::Float64 = 0.0     # mass
    U::RealVector = VEC0 # momentum
    E::Float64 = 0.0     # energy
    
    P::Float64 = 0.0     # pressure
    S::RealMatrix = MAT0 # viscous stress
    area::Float64 = 0.0
    rho::Float64 = 0.0
    c2::Float64 = 0.0    # sound speed squared

    # sides of the polygon
    edges::FastVector{Edge} = emptypolygon()
end

PolygonNSc(x::RealVector)::PolygonNSc = PolygonNSc(x=x)
const GridNSc = VoronoiGrid{PolygonNSc}