# basic Voronoi Polygon for Navier-Stokes equations

@kwdef mutable struct PolygonNS <: VoronoiPolygon
    x::RealVector         # position

    rho::Float64  = 0.0   # density
    v::RealVector = VEC0  # velocity
    e::Float64    = 0.0   # specific energy
    
    P::Float64    = 0.0   # pressure
    c2::Float64   = 0.0   # speed of sound squared
    D::RealMatrix = MAT0  # velocity deformation tensor
    mu::Float64 = 0.0     # dynamic viscosity
    dv::RealVector = VEC0 # repair velocity (used for mesh relaxation step)

    # extensive vars
    mass::Float64 = 0.0
    momentum::RealVector = VEC0
    energy::Float64 = 0.0
    phase::Int = 0

    # sides of the polygon (in no particular order)
    edges::FastVector{Edge} = emptypolygon()
    quality::Float64 = 0.0
end

PolygonNS(x::RealVector)::PolygonNS = PolygonNS(x=x)
const GridNS = VoronoiGrid{PolygonNS}


# Voronoi Polygon for Navier-Stokes-Fourier equations

@kwdef mutable struct PolygonNSF <: VoronoiPolygon
    x::RealVector        

    rho::Float64  = 0.0
    v::RealVector = VEC0
    e::Float64    = 0.0
    
    P::Float64    = 0.0
    T::Float64    = 0.0  # temperature
    k::Float64    = 0.0  # heat conductivity
    cV::Float64   = 0.0  # specific heat capacity at constant volume
    c2::Float64   = 0.0  
    D::RealMatrix = MAT0 
    mu::Float64 = 0.0    
    dv::RealVector = VEC0

    # extensive vars
    mass::Float64 = 0.0
    momentum::RealVector = VEC0
    energy::Float64 = 0.0
    phase::Int = 0

    # sides of the polygon (in no particular order)
    edges::FastVector{Edge} = emptypolygon()
    quality::Float64 = 0.0
end

PolygonNSF(x::RealVector)::PolygonNSF = PolygonNSF(x=x)
const GridNSF = VoronoiGrid{PolygonNSF}

