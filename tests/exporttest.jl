module simpletest

using WriteVTK
using StaticArrays


function test1()
    points = [rand(SVector{3, Float64}) for _ in 1:4]
    cell = MeshCell(PolyData.Polys(), (1, 2, 3, 4))
    polys = MeshCell[]
    push!(polys, cell)
    vtk_grid("my_vtp_file", points, polys)
end

function test2()
    points = [rand(SVector{3, Float64}) for _ in 1:4]
    cell = MeshCell(PolyData.Polys(), (1, 2, 3, 4))
    polys = [cell]
    vtk_grid("my_vtp_file", points, polys)
end

end