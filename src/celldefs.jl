"""
    Euler_vars()

A convenience macro which defines the Eulerian fluid variables.
See `examples/doubleshear.jl` for an example of usage.
"""
macro Euler_vars()
    return esc(quote
        x::RealVector         # generating seed
        rho::Float64  = 0.0   # density
        v::RealVector = VEC0  # velocity
        e::Float64    = 0.0   # specific energy
        P::Float64    = 0.0   # pressure
        c2::Float64   = 0.0   # speed of sound squared
        dv::RealVector = VEC0 # repair velocity (used for mesh relaxation step)

        # extensive vars
        mass::Float64 = 0.0
        momentum::RealVector = VEC0
        energy::Float64 = 0.0
        phase::Int = 0

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
    @Euler_vars
    D::RealMatrix = MAT0  # velocity deformation tensor
    mu::Float64 = 0.0     # dynamic viscosity
end

const GridNS = VoronoiGrid{PolygonNS}

"""
    PolygonNSF(; x::RealVector, kwargs...)
    
Predefined Voronoi Polygon for Navier-Stokes-Fourier equations.
Mandatory keyword variable `x` is the position of the generating seed.
""" 
@kwdef mutable struct PolygonNSF <: VoronoiPolygon
    @Euler_vars
    D::RealMatrix = MAT0  # velocity deformation tensor
    mu::Float64   = 0.0   # dynamic viscosity
    T::Float64    = 0.0   # temperature
    k::Float64    = 0.0   # heat conductivity
    cV::Float64   = 0.0   # specific heat capacity at constant volume
end

const GridNSF = VoronoiGrid{PolygonNSF}

"""
    PolygonMulti(; x::RealVector, kwargs...)
    
Predefined Voronoi Polygon for Navier-Stokes equations with implicit viscosity,
optimized for multiphase problems with high density ratios, including surface tension.
Mandatory keyword variable `x` is the position of the generating seed.
"""
@kwdef mutable struct PolygonMulti <: VoronoiPolygon
    @Euler_vars
    mu::Float64    = 0.0      # dynamic viscosity
    cgrad::RealVector = VEC0  # coloring function gradient
    st::Float64    = 0.0      # surface tension coefficient
end

const GridMulti = VoronoiGrid{PolygonMulti}