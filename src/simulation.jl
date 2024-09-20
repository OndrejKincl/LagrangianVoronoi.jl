"""
    SimulationWorkspace

Abstract type for containing the computational grid, solvers and global variables.
This methods should be defined for an instance `sim` of SimulationWorkspace:

* step!(sim, time::Float64)
* postproc!(sim, time::Float64) [optional] 
"""
abstract type SimulationWorkspace end

function donothing(sim::SimulationWorkspace, t::Float64)
    return
end

"""
    run!(sim::SimulationWorkspace, dt::Float64, t_end::Float64, step!::Function; kwargs...)

A utility function for time-marching and exporting the results. 
Specify the time step `dt`, final time `t_end` and a function 
``step!(sim::SimulationWorkspace, time::Float64).``
Keyword arguments:
* `nframes::Bool` (how many times should I export the data?)
* `path::String` (where to export the data?)
* `save_points::Bool` (should I export the point data?)
* `save_grid:::Bool` (should I export the grid data?)
* `save_csv::Bool` (should I export global variables as a csv data?)
* `vtp_vars` (which local variables should I export?)
* `csv_vars` (which global variables should I export?)
* `postproc!::Function` (If provided, `postproc!(sim, dt)` is called each time just before the data is saved. 
Use this for some (expensive) post proccessing that affects the output files, not the simulation itself.)

Note this assumes constant time step. See `examples/sedov.jl` for an example with adaptive time step.
"""
function run!(sim::SimulationWorkspace, dt::Float64, t_end::Float64, step!::Function;
    nframes::Int = 100,
    path::String = "results",
    save_points::Bool = true,
    save_grid::Bool = true,
    save_csv::Bool = true,
    vtp_vars = Symbol[],
    csv_vars = Symbol[],
    postproc!::Function = donothing
    )
    if !ispath(path)
        mkpath(path)
        @info "created a new path: $(path)"
    end 
    pvd_c = paraview_collection(joinpath(path, "cells.pvd"))
    pvd_p = paraview_collection(joinpath(path, "points.pvd"))


    csv_data = Dict(:time => Float64[])
    for var in csv_vars
        isa(var, Symbol) || error("csv_vars must be Symbols")
        (var == :time) && error("csv_vars cannot be :time")
        push!(csv_data, var => Float64[])
    end
    k = 0
    nframe = 0
    
    k_frame = typemax(Int)
    if nframes > 0
        k_frame = round(Int, t_end/(nframes*dt))
    end
    k_frame = max(k_frame, 1)
    t = 0.0

    while t < t_end
        k += 1
        step!(sim, t)
        if (k % k_frame == 0)
            @show t
            postproc!(sim, t)
            push!(csv_data[:time], t)
            for var in csv_vars
                push!(csv_data[var], getproperty(sim, var))
            end
            if save_grid
                filename= joinpath(path, "cframe$(nframe).vtp")
                pvd_c[t] = export_grid(sim.grid, filename, vtp_vars...)
            end
            if save_points
                filename= joinpath(path, "pframe$(nframe).vtp")
                pvd_p[t] = export_points(sim.grid, filename, vtp_vars...)
            end
            nframe += 1
        end
        t += dt
    end
    save_grid && vtk_save(pvd_c)
    save_points && vtk_save(pvd_p)
    save_csv && CSV.write(string(path, "/simdata.csv"), DataFrame(csv_data)) 
    return   
end

function movingavg(x::AbstractVector, radius::Integer=1)
    y = similar(x)
    len = length(x)
    for i in eachindex(x)
        lo = max(i-radius, 1)
        hi = min(i+radius, len)
        nsummands = 0
        y[i] = zero(eltype(x))
        for j in lo:hi
            y[i] += x[j]
            nsummands += 1
        end
        y[i] /= nsummands
    end
    return y
end