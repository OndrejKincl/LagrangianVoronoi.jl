module sedov

using StaticArrays, DifferentialEquations, Plots, LaTeXStrings, Match, CSV, DataFrames
import SpecialFunctions:gamma
const VEC = SVector{3, Float64}
const MAT = SMatrix{3, 3, Float64, 9}

const d = 2 # dimension 
const kappa = 7/5 # adiabatic index
const E_yield = 0.3 # yield energy
const rho1 = 1.0
const rho_min = 1e-4
const eta_min = 1e-4

const dt = 1e-4
const t_plot = 1.0
const x0_plot = 0.0
const x1_plot = 1.0
const unit_sphere = 2.0*pi^(d/2)/gamma(d/2) # the volume of a unit sphere in d dimensions

function odefun(u, p, eta)
    if u[1] < rho_min
        return zero(VEC)
    end
    mu = 2.0/(d+2)
    A = MAT(
        mu-u[2],       -u[1],       0.0,
            0.0,     mu-u[2], -1.0/u[1],
            0.0, -kappa*u[3],   mu-u[2]
    )
    b = (1.0/eta)*VEC(
        d*u[1]*u[2],
        2.0*u[3]/u[1] - (1.0 - u[2])*u[2],
        d*kappa*u[2]*u[3] - 2.0*(1.0 - u[2])*u[3]
    )
    return transpose(A)\b
end

function integrand(u, eta)
    return unit_sphere*(0.5*u[1]*u[2]^2 + u[3]/(kappa - 1.0))*eta^(d+1)
end

function find_u(eta_max)
    u0 = VEC( 
        (kappa+1.0)/(kappa-1.0),
        4.0/((d+2)*(kappa+1.0)),
        8.0/((d+2)^2*(kappa+1.0))
    )
    prob = ODEProblem(odefun, u0, (eta_max, eta_min), dt = -dt)
    @time sol = solve(prob, RKO65());
    N = length(sol)
    eta = collect(range(eta_min, eta_max, length = N))
    sol = [sol[N + 1 - i] for i in 1:N]
    return sol, eta
end

function energy_integral(sol, eta) # this should equal 1.0
    I = 0.0
    N = length(sol)
    for i in 1:2:N-2
        h = eta[i+2]-eta[i]
        I += h*integrand(sol[i], eta[i])/6
        I += 4*h*integrand(sol[i+1], eta[i+1])/6
        I += h*integrand(sol[i+2], eta[i+2])/6
    end
    if iseven(N)
        h = eta[N]-eta[N-1]
        I += 0.5*h*integrand(sol[N-1], eta[N-1])
        I += 0.5*h*integrand(sol[N], eta[N])
    end
    return I
end

function find_eta_max(tol = 1e-4, maxit = 100)
    a = eta_min # lower bound
    b = Inf     # upper bound
    I = 0.0     # energy integral
    it = 0
    m = 1.0
    while (abs(I - 1.0) > tol) && (it < maxit)
        m = (isinf(b) ? max(1.0, 2*a) : 0.5*(a+b))  # midpoint
        sol, eta = find_u(m)
        I = energy_integral(sol, eta)
        @show m
        @show it
        @show I
        if I > 1.0
            b = m
        else
            a = m
        end
        it += 1
    end
    return m
end

function extract_solution(sol, eta, t; from = 0.0, to = 1.0)
    r = eta.*(t^2*E_yield/rho1)^(1.0/(d+2))
    N = length(eta)
    rho = [rho1*sol[i][1] for i in 1:N]
    v = [sol[i][2]*r[i]/t for i in 1:N]
    P = [rho1*sol[i][3]*(r[i]/t)^2 for i in 1:N] 
    for _r in r[end] : 0.1 : to
        push!(r, _r)
        push!(rho, rho1)
        push!(v, 0.0)
        push!(P, 0.0)
    end
    start = 1
    while (start < N) && ((r[start] < from) || (rho[start] < rho_min))
        start += 1
    end
    r = r[start:end]
    rho = rho[start:end]
    v = v[start:end]
    P = P[start:end]
    return r, rho, v, P
end

function main()
    eta_max = find_eta_max()
    @show eta_max
    sol, eta = find_u(eta_max)
    r, rho, v, P = extract_solution(sol, eta, t_plot; from = x0_plot, to = x1_plot)
    plt = plot(r, [rho v P], label = [L"\rho" L"v" L"p"], xlabel = "r", ylabel = "")
    @show energy_integral(sol, eta)
    savefig(plt, "sedov.pdf")
    df = DataFrame(r = r, rho = rho, v = v, P = P)
    CSV.write("sedov.csv", df)
end


end