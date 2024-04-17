include("isolver.jl")
using Parameters

mutable struct CNAB_scheme
    grid::VoronoiGrid
    dt::Float64
    k::Int
    solver::PressureSolver
    VC_stab::Bool
    SPH_stab::Bool
    nu::Float64
    CNAB_scheme(
        grid::VoronoiGrid,
        dt::Float64; 
        VC_stab::Bool = true,
        SPH_stab::Bool = false,
        nu::Float64 = 0.0
    ) = new(grid, dt, 0, PressureSolver(grid), VC_stab, SPH_stab, nu)
end

function step!(scheme::CNAB_scheme)
    grid = scheme.grid
    k = scheme.k
    dt = scheme.dt
    # new positions
    @batch for p in grid.polygons
        if k > 1
            dx = 1.5*dt*p.var.v - 0.5*dt*p.var.v_old
        else
            dx = dt*p.var.v
        end
        dx = admissible_step(grid, p, dx)
        p.x += dx
    end
    remesh!(grid)
    # intermediate velocity
    @batch for p in grid.polygons
        p.var.v_old = p.var.v
        if k > 1
            p.var.v += 0.5*dt*p.var.a
        end
        p.var.a = VEC0
    end
    # the implicit step
    if scheme.SPH_stab
        h_stab = grid.h
        apply_local!(grid, (p,q,r) -> stabilizer!(p,q,r,h_stab,dt), h_stab)
    end
    implicit_dt = (k > 1 ? 0.5*dt : dt)
    find_pressure!(scheme.solver, 0.5*dt)
    apply_binary!(grid, internal_force!)
    if scheme.VC_stab
        stabilize!(grid)
    end
    @batch for p in grid.polygons
        p.var.v += implicit_dt*p.var.a
    end
    scheme.k += 1
end

function stabilizer!(p::VoronoiPolygon, q::VoronoiPolygon, r::Float64, h_stab::Float64, dt::Float64)
	p.var.v += -dt*q.var.mass*rDwendland2(h_stab,r)*(2*P_stab/rho0^2)*(p.x - q.x)
end

function admissible_step(grid::VoronoiGrid, p::VoronoiPolygon, dx::RealVector)::RealVector
    if isinside(grid.boundary_rect, p.x + dx)
        return dx
    else # try to project v to tangent space
        for e in p.edges
            if isboundary(e)
                n = normal_vector(e)
                dx -= dot(dx, n)*n
            end
        end
        if isinside(grid.boundary_rect, p.x + dx)
            return dx
        else # give up
            return VEC0
        end
    end
end