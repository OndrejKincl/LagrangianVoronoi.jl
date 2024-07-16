function move!(grid::VoronoiGrid, dt::Float64)
    @batch for p in grid.polygons
        if any(isnan, p.v)
            throw("Velocity field invalidated.") 
        end
        new_x = p.x + dt*p.v
        if isinside(grid.boundary_rect, new_x)
            p.x = new_x
        else # try to project v to tangent space
            for e in boundaries(p)
                n = normal_vector(e)
                p.v -= dot(p.v, n)*n
            end
            new_x = p.x + dt*p.v
            if isinside(grid.boundary_rect, new_x)
                p.x = new_x
            else # give up and halt the particle
                p.v = VEC0
            end
        end
    end
    remesh!(grid)
end


function accelerate!(grid::VoronoiGrid, dt::Float64)
    @batch for p in grid.polygons
        p.v += dt*p.a
        p.a = VEC0
    end
end

function pressure_interaction!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    p.a += (1.0/p.mass)*lr_ratio(p,q,e)*(p.P - q.P)*(m - p.x)
end

function pressure_force!(grid::VoronoiGrid, dt::Float64; stabilize = true)
    apply_binary!(grid, pressure_interaction!)
    if stabilize
        @batch for p in grid.polygons
            LapP = 0.0 #laplacian of pressure
            for (q,e) in neighbors(p, grid)
                q = grid.polygons[e.label]
                LapP -= lr_ratio(p,q,e)*(p.P - q.P)
            end
            if LapP > 0.0
                c = centroid(p)
                acc = 1.5*LapP/(p.mass)*(c - p.x)
                if dot(acc, p.v + dt*p.a) < 0.0
                    p.a += acc
                end 
            end
        end
    end
    accelerate!(grid, dt)
end

homoDirichlet(_::RealVector) = VEC0

function viscous_force!(grid::VoronoiGrid, mu::Float64, dt::Float64; noslip = false, vDirichlet::Function = homoDirichlet)
    @batch for p in grid.polygons
        for (q,e) in neighbors(p, grid)
            q = grid.polygons[e.label]
            p.a += mu*lr_ratio(p, q, e)/p.mass*(q.v - p.v)
        end
    end
    accelerate!(grid, dt)
    if noslip
        @batch for p in grid.polygons
            gamma = 1.0
            for e in boundaries(p)
                m = 0.5*(e.v1 + e.v2)
                n = normal_vector(e)
                lrr = len(e)/abs(dot(m - p.x, n))
                lambda = mu*lrr/p.mass
                p.v += dt*lambda*vDirichlet(m)
                gamma += dt*lambda
            end
            p.v = p.v/gamma
        end
    end
end