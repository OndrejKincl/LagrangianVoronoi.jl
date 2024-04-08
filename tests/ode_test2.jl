# solve equation
# dx/dt = v
# dv/dt = -x - 2*mu*v
# x(0) = 0
# v(0) = 1
# with implicit-explicit midpoint rule



module ode_test
using Plots, LaTeXStrings, LinearAlgebra

const t_end = 2.0
const mu = 0.1
const alpha = sqrt(1 - mu^2)

function x_exact(t::Float64)
    return exp(-mu*t)*sin(alpha*t)/alpha
end

function v_exact(t::Float64)
    return exp(-mu*t)*cos(alpha*t) - mu*x_exact(t)
end

function a_exact(t::Float64)
    return -x_exact(t) - 2*mu*v_exact(t)
end

mutable struct PhasePoint
    x::Float64
    v::Float64
    a::Float64
    x_old::Float64
    v_old::Float64
    a_old::Float64
    PhasePoint(x,v,a) = new(x,v,a,0,0,0)
end

function midpoint_step!(p::PhasePoint, dt::Float64)
    # explicit step
    _x = p.x_old + 2*dt*p.v
    p.x_old = p.x
    p.x = _x

    # implicit step
    _v = p.v_old + dt*p.a_old
    p.v_old = p.v
    p.v = _v
    p.a_old = p.a
    get_a!(p, dt)
    p.v += dt*p.a

    return
end

function cnab_step!(p::PhasePoint, dt::Float64)
    # explicit step
    p.x += 1.5*dt*p.v - 0.5*dt*p.v_old

    # implicit step
    p.v_old = p.v
    p.v += 0.5*dt*p.a
    get_a!(p, 0.5*dt)
    p.v += 0.5*dt*p.a

    return
end

function init!(p::PhasePoint, dt::Float64)
    p.x_old = p.x
    p.v_old = p.v
    p.a_old = p.a

    p.x += dt*p.v
    get_a!(p, dt)
    p.v += dt*p.a
end


function get_a!(p::PhasePoint, dt::Float64)
    p.a = -(p.x + 2*mu*p.v)/(1 + 2*mu*dt)
    return 
end

function solve(N::Int)
    dt = t_end/N
    p = PhasePoint(
        x_exact(0.0), 
        v_exact(0.0),
        a_exact(0.0)
    )
    xs = Float64[]
    push!(xs, p.x)
    for it in 1:N
        if it < 3
            init!(p, dt)
        else
            cnab_step!(p, dt)
        end
        push!(xs, p.x)
    end
    err = abs(p.x - x_exact(t_end))
    @show err
    return xs, err
end

function plot_solution(N)
    (xs, _) = solve(N)
    t = range(0, t_end, length = length(xs))
    xs_exact = x_exact.(t)
    plt = plot(t, [xs xs_exact ], label = ["x" "x_exact"])
    savefig(plt, "ode_graph.pdf")
end

function main()
    N = [32, 48, 72, 108, 162]
    e = [solve(_N)[2] for _N in N]
    logN = log10.(N)
    loge = log10.(e)
    plt = plot(
        logN, loge, 
        axis_ratio = 1, 
        xlabel = L"\log \, N", ylabel = L"\log \, \epsilon", 
        markershape = :hex, 
        label = ""
    )
    # linear regression
    A = [logN ones(length(N))]
    b = A\loge
    loge_reg = [b[1]*logN[i] + b[2] for i in 1:length(N)]
    plot!(plt, logN, loge_reg, linestyle = :dot, label = string("slope = ", round(b[1], sigdigits=3)))
    savefig(plt, "ode_convergence.pdf")
    print("slope = ", b[1])
end

end