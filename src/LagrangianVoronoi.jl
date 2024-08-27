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
export RealVector, RealMatrix, VEC0, VECX, VECY, MAT0, MAT1, VECNULL, Edge, Rectangle, UnitRectangle, len, isinside, norm_squared, verts

include("utils/fastvector.jl")
export FastVector

include("polygon.jl")
export VoronoiPolygon, area, isboundary, surface_element, normal_vector, centroid, lr_ratio, POLYGON_SIZEHINT, PreAllocVector
export BDARY_UP, BDARY_RIGHT, BDARY_DOWN, BDARY_LEFT, emptypolygon

include("cell_list.jl")

include("voronoigrid.jl")
export VoronoiGrid, remesh!, nearest_polygon

include("IO.jl")
export export_grid, export_points

include("apply.jl")
export apply_binary!, apply_unary!, apply_local!

include("movingls.jl")
export LinearExpansion, QuadraticExpansion, CubicExpansion
export ls_reconstruction, power_vector, ls_reconstruction, poly_eval, integral, point_value

include("populate.jl")
export populate_circ!, populate_rand!, populate_vogel!, populate_rect!, populate_lloyd!, populate_hex!, get_mass!

include("utils/threadedvec.jl")
export ThreadedVec

include("utils/iterators.jl")
export neighbors, boundaries

include("utils/newtonlloyd.jl")
export newtonlloyd!

include("utils/run.jl")
export SimulationWorkspace, run!

include("utils/lrrcache.jl")
export Lrr_cache, new_lrr_cache, refresh!

include("NavierStokes/definitions.jl")
export GridNS, PolygonNS

include("NavierStokes/physics.jl")
export move!, pressure_force!, viscous_force!

include("NavierStokes/psolver.jl")
export PressureSolver, find_pressure!

include("compressible/definitions.jl")
export GridNSc, PolygonNSc, GridNSFc, PolygonNSFc

include("compressible/psolver.jl")
export CompressibleOperator, CompressibleSolver, find_pressure!

include("compressible/pressure.jl")
export pressure_step!, ideal_eos!, stiffened_eos!, gravity_step!

include("compressible/diffusion.jl")
export find_D!, viscous_step!

include("compressible/relaxation.jl")
export Relaxator, relaxation_step!

include("compressible/fourier.jl")
export ideal_temperature!, fourier_step!, fourier_dirichlet_bc!

include("muscl/definitions.jl")
export PolygonMUSCL, GridMUSCL

include("muscl/reconstruction.jl")
export reconstruction!, get_intensives!

include("muscl/physical_step.jl")
export ideal_gas_law, physical_step!, update!

include("muscl/timestepping.jl")
export RK2_step!




end