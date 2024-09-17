module LagrangianVoronoi

using LinearAlgebra
using StaticArrays
using WriteVTK
using Base.Threads
using Polyester
using CSV
using DataFrames
using Krylov

include("geometry.jl")
export RealVector, RealMatrix, VEC0, VECX, VECY, MAT0, MAT1, VECNULL, Edge, Rectangle, UnitRectangle, len, isinside, norm_squared, verts, midpoint

include("fastvector.jl")
export FastVector

include("polygon.jl")
export VoronoiPolygon, area, isboundary, surface_element, normal_vector, centroid, lr_ratio, POLYGON_SIZEHINT, PreAllocVector
export BDARY_UP, BDARY_RIGHT, BDARY_DOWN, BDARY_LEFT, emptypolygon

include("neighborlist.jl")

include("voronoigrid.jl")
export VoronoiGrid, remesh!, nearest_polygon, get_arrow

include("IO.jl")
export export_grid, export_points

include("apply.jl")
export apply_binary!, apply_unary!, apply_local!

include("movingls.jl")
export LinearExpansion, QuadraticExpansion, CubicExpansion
export ls_reconstruction, power_vector, ls_reconstruction, poly_eval, integral, point_value

include("populate.jl")
export populate_circ!, populate_rand!, populate_vogel!, populate_rect!, populate_lloyd!, populate_hex!, get_mass!

include("threadedvec.jl")
export ThreadedVec

include("iterators.jl")
export neighbors, boundaries

include("simulation.jl")
export SimulationWorkspace, run!, movingavg

include("celldefs.jl")
export GridNS, PolygonNS, GridNSF, PolygonNSF

include("move.jl")
export move!

include("pressure.jl")
export pressure_step!, ideal_eos!, stiffened_eos!, gravity_step!, PressureOperator, PressureSolver, find_pressure!

include("diffusion.jl")
export find_D!, viscous_step!

include("relaxation.jl")
export find_dv!, relaxation_step!, MultiphaseSolver, multiphase_projection!

include("fourier.jl")
export ideal_temperature!, fourier_step!, fourier_dirichlet_bc!



end