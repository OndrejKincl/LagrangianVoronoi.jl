function Vec3(x::RealVector)::SVector{3, Float64}
    return SVector{3, Float64}(x[1], x[2], 0.0)
end

function invert(e::Edge)::Edge
    return Edge(e.v2, e.v1, label = e.label)
end

"""
    export_grid(grid::VoronoiGrid, filename::String, vars::Symbol...)

Export grid to a vtk file. Append datasets specified by `vars`. Currently only numbers and vectors can be exported.
"""
function export_grid(grid::VoronoiGrid, filename::String, vars::Symbol...)
    verts = Vector{SVector{3, Float64}}()
    polys = Vector{MeshCell{PolyData.Polys}}()
    for poly in grid.polygons
        sort_edges!(poly)
        # construct the meshcell
        n0 = length(verts)
        for i in 1:length(poly.edges)
            push!(verts, Vec3(poly.edges[i].v1))
        end 
        meshcell = (n0+1):(n0+length(poly.edges))
        push!(polys, MeshCell(PolyData.Polys(), meshcell))
    end
    if isempty(verts)
        @warn "Exporting a Voronoi grid with 0 edges."
    end
    vtk = vtk_grid(filename, verts, polys)
    append_datasets!(vtk, grid, vars...)
    return vtk
end

function sort_edges!(poly::VoronoiPolygon)
    # reorder the edges to connect them
    for i in 1:length(poly.edges)
        last_vert = poly.edges[i].v2
        for j in (i+1):length(poly.edges)
            if last_vert == poly.edges[j].v1
                tmp = poly.edges[i+1]
                poly.edges[i+1] = poly.edges[j]
                poly.edges[j] = tmp
                break
            end
        end
    end
end

"""
    export_points(grid::VoronoiGrid, filename::String, vars::Symbol...)

Export points to a vtk file as a cloud of particles. Append datasets specified by `vars`.
"""
function export_points(grid::VoronoiGrid, filename::String, vars::Symbol...)
    points = [Vec3(grid.polygons[i].x) for i in eachindex(grid.polygons)]
    cells = [MeshCell(PolyData.Verts(), (i,)) for i in eachindex(grid.polygons)]
    vtk = vtk_grid(filename, points, cells)
    append_datasets!(vtk, grid, vars...)     
    return vtk
end

function append_datasets!(vtk, grid::VoronoiGrid, vars::Symbol...)
    Tp = eltype(grid.polygons)
    for fieldname in vars
        if !(fieldname in fieldnames(Tp))
            throw(ArgumentError(string("Cannot export variable ", fieldname, " because it does not exist.")))
        end
        exportType = (fieldtype(Tp, fieldname) <: Number) ? Float64 : Vec3
        vtk[string(fieldname)] = [exportType(getproperty(grid.polygons[i], fieldname)) for i in eachindex(grid.polygons)]
    end
end