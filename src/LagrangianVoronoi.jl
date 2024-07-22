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
export RealVector, RealMatrix, VEC0, VECX, VECY, VECNULL, Edge, Rectangle, UnitRectangle, len, isinside, norm_squared, verts

include("utils/fastvector.jl")
export FastVector

include("polygon.jl")
export VoronoiPolygon, area, isboundary, surface_element, normal_vector, centroid, lr_ratio, POLYGON_SIZEHINT, PreAllocVector

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

include("NavierStokes/definitions.jl")
export GridNS, PolygonNS

include("NavierStokes/physics.jl")
export move!, pressure_force!, viscous_force!

include("NavierStokes/pressuresolver.jl")
export PressureSolver, find_pressure!

include("compressible/definitions.jl")
export GridNSc, PolygonNSc

include("compressible/solver.jl")
export CompressibleSolver, find_pressure!

include("compressible/physics.jl")
export find_D!, find_mu!, viscous_step!, ideal_eos!, pressure_step!, find_rho!, update_energy!, stiffened_eos!

include("compressible/relaxation.jl")
export relaxation_step!

include("muscl/definitions.jl")
export PolygonMUSCL, GridMUSCL

include("muscl/reconstruction.jl")
export reconstruction!, get_intensives!

include("muscl/physical_step.jl")
export ideal_gas_law, physical_step!, update!

include("muscl/timestepping.jl")
export RK2_step!

end