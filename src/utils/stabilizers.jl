struct LloydStabilizer{T}
    grid::VoronoiGrid{T}
    max_centroid_dist::Float64
    dx::Vector{RealVector}
    de::Vector{Float64}
    dv::Vector{RealVector}
    LloydStabilizer(grid::VoronoiGrid{T}; max_centroid_dist = 0.1) where T = begin
        N = length(grid.polygons)
        dx = zeros(RealVector, N)
        de = zeros(N)
        dv = zeros(RealVector, N)
        return new{T}(grid, max_centroid_dist, dx, de, dv)
    end
end

function stabilize!(ls::LloydStabilizer)
    for (i,p) in enumerate(ls.grid.polygons)
        if isboundary(p)
            continue
        end
        ls.dx[i] = VEC0
        ls.dv[i] = VEC0
        ls.de[i] = 0.0
        L_char = sqrt(p.mass/p.rho)
        centr = centroid(p)
        centroid_dist = norm(centr - p.x)
        #=
        particle_dist = Inf
        for (q, e) in neighbors(p, ls.grid)
            dist2 = norm_squared(p.x-q.x)
            particle_dist = min(dist2, particle_dist)
        end
        particle_dist = sqrt(particle_dist)
        =#
        dr = centroid_dist - ls.max_centroid_dist*L_char
        if dr > 0
            ls.dx[i] = dr*(centr - p.x)/centroid_dist
        end
    end
    for (i,p) in enumerate(ls.grid.polygons)
        for (q,e) in neighbors(p, ls.grid)
            j = e.label
            x_pq = p.x - q.x
            dotp = p.rho*dot(ls.dx[i], x_pq)
            dotq = q.rho*dot(ls.dx[j], x_pq)
            lrr = lr_ratio(p, q, e)/p.mass
            ls.dv[i] = 0.5*lrr*(dotp*p.v + dotq*q.v)
            ls.de[i] = 0.5*lrr*(dotp*p.e + dotq*q.e)
        end
    end
    for (i,p) in enumerate(ls.grid.polygons)
        p.v += ls.dv[i]
        p.e += ls.de[i]
        p.x += ls.dx[i]
    end
    remesh!(ls.grid)
end