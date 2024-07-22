# Voronoi Polygon for compressible Navier-Stokes equations

@kwdef mutable struct PolygonNSc <: VoronoiPolygon
    x::RealVector         # position
    v::RealVector = VEC0  # velocity
    e::Float64 = 0.0      # specific energy
    
    P::Float64 = 0.0     # pressure
    D::RealMatrix = MAT0 # velocity deformation tensor
    rho::Float64 = 0.0
    c2::Float64 = 0.0    # sound speed squared
    mu::Float64 = 0.0    # dynamic viscosity
    
    dv::RealVector = VEC0 # repair velocity 

    # extensive variables 
    area::Float64 = 0.0
    mass::Float64 = 0.0
    momentum::RealVector = VEC0
    energy::Float64 = 0.0
    
    # sides of the polygon
    edges::FastVector{Edge} = emptypolygon()
end

PolygonNSc(x::RealVector)::PolygonNSc = PolygonNSc(x=x)
const GridNSc = VoronoiGrid{PolygonNSc}