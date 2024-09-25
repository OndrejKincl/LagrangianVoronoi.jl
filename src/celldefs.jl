"""
    fluid_variables()

A convenience macro for enriching the definition of PolygonNS by user-defined variables.
See `examples/doubleshear.jl` for a tutorial how to use it.
"""
macro fluid_variables()
    return esc(quote
        x::RealVector         # generating seed
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

        # thermodynamic variables
        T::Float64    = 0.0  # temperature
        k::Float64    = 0.0  # heat conductivity
        cV::Float64   = 0.0  # specific heat capacity at constant volume

        # sides of the polygon (in no particular order)
        edges::FastVector{Edge} = emptypolygon()
        quality::Float64 = 0.0
    end)
end


"""
    PolygonNS(; x::RealVector, kwargs...)
    
Predefined Voronoi Polygon for Navier-Stokes equations.
Mandatory keyword variable `x` is the position of the generating seed.
""" 
@kwdef mutable struct PolygonNS <: VoronoiPolygon
    @fluid_variables
end

const GridNS = VoronoiGrid{PolygonNS}