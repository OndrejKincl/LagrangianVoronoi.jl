struct MUSCLstate
    x::RealVector
    M::Float64
    P::RealVector
    E::Float64
end

const MUSCLstate0 = MUSCLstate(VEC0, 0.0, VEC0, 0.0)

# Voronoi Polygon for MUSCL compressible Navier-Stokes equations

@kwdef mutable struct PolygonMUSCL <: VoronoiPolygon
    x::RealVector         # seed
    c::RealVector = VEC0  # centroid
    A::Float64 = 0.0      # area
    
    M::Float64 = 0.0      # mass
    P::RealVector = VEC0  # momentum
    E::Float64 = 0.0      # energy

    w::RealVector = VEC0  # mesh velocity
    
    # interpolating gradients
    Grho::RealVector = VEC0
    Gu::RealMatrix = MAT0
    Ge::RealVector = VEC0
    D::RealMatrix = MAT0
    
    rho::Float64 = 0.0
    u::RealVector = VEC0
    e::Float64 = 0.0

    Mdot::Float64 = 0.0
    Pdot::RealVector = VEC0
    Edot::Float64 = 0.0

    old_state::MUSCLstate = MUSCLstate0

    # sides of the polygon
    edges::FastVector{Edge} = emptypolygon()
end

PolygonMUSCL(x::RealVector)::PolygonMUSCL = PolygonMUSCL(x=x)
const GridMUSCL = VoronoiGrid{PolygonMUSCL}




