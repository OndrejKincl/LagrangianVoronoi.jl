using Base.Threads

function lr_ratio(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)::Float64
    l2 = norm_squared(e.v1 - e.v2)
    r2 = norm_squared(p.x - q.x)
    return sqrt(l2/r2)
end


function internal_force!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    z = 0.5*(p.x + q.x)
    p.var.a += (1.0/p.var.mass)*lr_ratio(p,q,e)*(
        (p.var.P - q.var.P)*(m - z) 
        + 0.5*(p.var.P + q.var.P)*(p.x - q.x) 
    ) 
end

#=
function compute_laplace!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    p.var.P += lr_ratio(p,q,e)*dot(p.var.a - q.var.a, m - q.x)
end

function assign_laplace!(y::Vector{Float64}, x::Vector{Float64}, grid::VoronoiGrid)
    @threads for i in eachindex(y)
        p = grid.polygons[i]
        p.var.P = x[i]
    end
    apply_binary!(grid, internal_force!)
    apply_binary!(grid, compute_laplace!)
    @threads for i in eachindex(x)
        p = grid.polygons[i]
        y[i] = p.var.P
        p.var.a = VEC0
    end
end
=#

function compute_laplace!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    p.var.div += lr_ratio(p,q,e)*(p.var.P - q.var.P)
end

function assign_laplace!(y::Vector{Float64}, x::Vector{Float64}, grid::VoronoiGrid)
    @threads for i in eachindex(y)
        p = grid.polygons[i]
        p.var.P = x[i]
        p.var.div = 0.0
    end
    apply_binary!(grid, compute_laplace!)
    @threads for i in eachindex(x)
        p = grid.polygons[i]
        y[i] = p.var.div
        p.var.div = 0.0
    end
end


function get_div!(p::VoronoiPolygon, q::VoronoiPolygon, e::Edge)
    m = 0.5*(e.v1 + e.v2)
    p.var.div += lr_ratio(p,q,e)*dot(p.var.v - q.var.v, m - q.x)
end

function assign_rhs!(b::Vector{Float64}, grid::VoronoiGrid, dt::Float64)
    apply_binary!(grid, get_div!)
    @threads for i in eachindex(b)
        p = grid.polygons[i]
        b[i] = -p.var.div*(p.var.rho/dt)
        p.var.div = 0.0
    end
end

function move!(p::VoronoiPolygon)
    p.x += dt*p.var.v
end

function accelerate!(p::VoronoiPolygon)
    p.var.v += dt*p.var.a
    p.var.a = VEC0
end

function no_slip!(p::VoronoiPolygon)
    if isboundary(p)
        p.var.v = VEC0
    end
end

struct MinresSolver
    grid::VoronoiGrid
    n::Int
    x::Vector{Float64}
    r::Vector{Float64}
    p0::Vector{Float64}
    p1::Vector{Float64}
    p2::Vector{Float64}
    s0::Vector{Float64}
    s1::Vector{Float64}
    s2::Vector{Float64}
    b::Vector{Float64}
    atol::Float64
    rtol::Float64
    maxit::Int
    MinresSolver(grid::VoronoiGrid) = begin
        n = length(grid.polygons)
        return new(
            grid,
            n, 
            zeros(n), 
            zeros(n), 
            zeros(n), 
            zeros(n), 
            zeros(n), 
            zeros(n), 
            zeros(n),  
            zeros(n),  
            zeros(n),  
            1e-2,
            1e-2,
            30
        )
    end
end

function reset!(solver::MinresSolver)
    @threads for i in 1:solver.n
        solver.x[i] = 0.0
        solver.r[i] = 0.0
        solver.p0[i] = 0.0
        solver.p1[i] = 0.0
        solver.p2[i] = 0.0
        solver.s0[i] = 0.0
        solver.s1[i] = 0.0
        solver.s2[i] = 0.0
        solver.b[i] = 0.0
    end
end


function solve!(solver::MinresSolver, dt::Float64)
    #reset!(solver)
    assign_rhs!(solver.b, solver.grid, dt)
    assign_laplace!(solver.r, solver.x, solver.grid)
    @threads for i in 1:solver.n
        solver.r[i] = solver.b[i] - solver.r[i]
        solver.p0[i] = solver.r[i]
    end
    res0 = sqrt(threaded_dot(solver.r, solver.r))
    assign_laplace!(solver.s0, solver.p0, solver.grid)
    @threads for i in 1:solver.n
        solver.p1[i] = solver.p0[i]
        solver.s1[i] = solver.s0[i]
    end
    iter = 0
    while iter < solver.maxit
        @threads for i in 1:solver.n
            solver.p2[i] = solver.p1[i]
            solver.p1[i] = solver.p0[i]
            solver.s2[i] = solver.s1[i]
            solver.s1[i] = solver.s0[i]
        end
        alpha = threaded_dot(solver.r, solver.s1)/threaded_dot(solver.s1, solver.s1);
        @threads for i in 1:solver.n
            solver.x[i] += alpha*solver.p1[i]
            solver.r[i] -= alpha*solver.s1[i] 
            solver.p0[i] = solver.s1[i]
        end
        res = sqrt(threaded_dot(solver.r, solver.r))
        if (res < solver.atol) || (res < solver.rtol)
            break
        end
        assign_laplace!(solver.s0, solver.s1, solver.grid)
        beta1 = threaded_dot(solver.s0, solver.s1)/threaded_dot(solver.s1, solver.s1);
        @threads for i in 1:solver.n
            solver.p0[i] -= beta1*solver.p1[i];
            solver.s0[i] -= beta1*solver.s1[i];
        end
        if iter > 1
            beta2 = threaded_dot(solver.s0, solver.s2)/threaded_dot(solver.s2, solver.s2);
            @threads for i in 1:solver.n
                solver.p0[i] -= beta2*solver.p2[i];
                solver.s0[i] -= beta2*solver.s2[i];
            end
        end
        iter += 1
    end
    @threads for i in 1:solver.n
        solver.grid.polygons[i].var.P = solver.x[i]
        #solver.x[i] = 0.0
    end
    apply_binary!(solver.grid, internal_force!)
    if iter == solver.maxit
        @warn "Minres solver did not converge!"
        res = sqrt(threaded_dot(solver.r, solver.r))
        println("absolute res = ", res)
        println("relative res = ", res/res0)
    end
end

function threaded_dot(x::Vector{Float64}, y::Vector{Float64})::Float64
    psums = zeros(Threads.nthreads())
    @threads for i in eachindex(x)
        psums[Threads.threadid()] += x[i]*y[i]
    end
    return sum(psums; init = 0.0)
end