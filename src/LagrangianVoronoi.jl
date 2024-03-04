module LagrangianVoronoi

using LinearAlgebra
using StaticArrays
using WriteVTK
using Base.Threads

include("preallocvector.jl")

include("geometry.jl")
export RealVector, RealMatrix, VEC0, VECX, VECY, VECNULL, Edge, Rectangle, len, isinside, norm_squared

include("polygon.jl")
export VoronoiPolygon, area, isboundary, surface_element, normal_vector

include("cell_list.jl")

include("voronoigrid.jl")
export VoronoiGrid, remesh!, limit_cell_diameter!

include("IO.jl")
export export_grid, export_points

include("apply.jl")
export apply_binary!, apply_unary!, apply_local!

include("assembler.jl")
export assemble_vector, assemble_matrix

include("ls_reconstruction.jl")
export LinearExpansion, QuadraticExpansion, CubicExpansion
export ls_reconstruction, power_vector, ls_reconstruction, poly_eval, integral

end