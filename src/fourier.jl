"""
    ideal_temperature!(grid::VoronoiGrid)

Assign temperature to every Voronoi polygon based on ideal gas law.
"""
function ideal_temperature!(grid::VoronoiGrid)
    @batch for p in grid.polygons
        p.T = eint(p)/p.cV 
    end
end

"""
    fourier_step!(grid::VoronoiGrid, dt::Float64)

Update the energy of Voronoi polygon by Fourier heat conduction. 
This assumes that every polygon `p` has its value of heat conductivity `p.k` assigned by the initial condition
or otherwise.
"""
function fourier_step!(grid::VoronoiGrid, dt::Float64)
    @batch for p in grid.polygons
        for (q,e,y) in neighbors(p,grid)
            lrr = lr_ratio(p.x-y,e)
            k = 0.5*(p.k + q.k)
            p.e -= dt*k/p.mass*lrr*(p.T - q.T)
        end
    end
end

"""
    heat_from_bdary!(grid::VoronoiGrid, dt::Float64, T_bc::Function)

Update the internal energy of a Voronoi cell by considering heat flux from boundary.
Function `T_bc` specifies the temperature at that boundary and should return `NaN` 
to indicate adiabatic walls.
This currently works only for ideal gas.
"""
function heat_from_bdary!(grid::VoronoiGrid, dt::Float64, T_bc::Function, gamma::Float64)
    @batch for p in grid.polygons
        if !isboundary(p)
            continue
        end
        implicit_factor = 1.0
        @assert(p.cV > 0.0)
        thermal_D = p.k/(p.rho*p.cV)
        e_kinetic = 0.5*norm_squared(p.v)
        p.T = (p.e - e_kinetic)/p.cV
        A = area(p)
        for e in p.edges
            if !isboundary(e)
                continue
            end
            m = 0.5*(e.v1 + e.v2)
            n = normal_vector(e)
            lrr = len(e)/abs(dot(m - p.x, n)) + 0.01*grid.dr
            T = T_bc(m, e.label)
            if !isnan(T)
                tmp = dt*thermal_D*lrr/A
                p.T += tmp*T
                implicit_factor += tmp
            end
        end
        p.T /= implicit_factor
        p.e = e_kinetic + p.cV*p.T
    end
end

