# solve equation
# du/dt = f(u) + g(u)
# with implicit-explicit midpoint rule
# f(u) = u
# g(u) = -u*u
# u(0) = 10
# u_exact = 10/(10 - 9*exp(-t))
module ode_test
using Plots, LaTeXStrings, LinearAlgebra

const t_end = 2.0

function imex_step(u0, u1, dt)::Float64
    v = u0 - dt*u0^2 + 2*dt*u1
    u2 = 2*v/(1 + sqrt(1 + 4*dt*v)) 
    #u2 = u1 + dt*(u1 - u1^2)
    return u2
end

function u_exact(t)::Float64
    return 10/(10 - 9*exp(-t))
end

function solve(N::Int)
    dt = t_end/N
    u = Float64[]
    push!(u, u_exact(0.0))
    push!(u, u_exact(dt))
    for k in 3:(N+1)
        #t = (k-1)*dt
        push!(u, imex_step(u[k-2], u[k-1], dt))
    end
    err = abs(u[end] - u_exact(t_end))
    @show err
    return (u, err)
end

function plot_solution(N)
    (u, _) = solve(N)
    t = range(0, t_end, length=(N+1))
    plt = plot(t, [u u_exact.(t)], label = ["u" "u_exact"])
    savefig(plt, "ode_graph.pdf")
end

function main()
    N = [32, 48, 72, 108, 162, 243]
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