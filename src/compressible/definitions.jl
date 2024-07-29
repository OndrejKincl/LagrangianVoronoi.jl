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

    # limiter parameters
    phi_rho::Float64 = 0.0

    # sides of the polygon (in no particular order)
    edges::FastVector{Edge} = emptypolygon()
end

PolygonNSc(x::RealVector)::PolygonNSc = PolygonNSc(x=x)
const GridNSc = VoronoiGrid{PolygonNSc}