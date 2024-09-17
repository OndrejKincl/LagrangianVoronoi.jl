module rabe #Rayleigh-Bénard instability
include("../../src/LagrangianVoronoi.jl")
using .LagrangianVoronoi
using Match, Polyester, LinearAlgebra, WriteVTK


const g = 9.8 #gravitation acceleration
const mu = 1e-4 #dynamic viscosity
const H = 0.1 #height
const W = 0.1
const gamma = 1.4 #adiabatic index
const rho0 = 10.0

const cV = 10.0  #sp. heat capacity  
const R = (gamma - 1.0)*cV
const thermal_k = mu*cV*gamma/0.71 #thermal conductivity

const P0 = 1e4
const c0 = sqrt(gamma*P0/rho0)
const Tu = P0/(rho0*R) #temperature of upper boundary (cooler)
const Td = 1000.0 #30.0 #Tu*(1.0 + contrast) #temperature of lower boundary (heater)

const export_path = "results/rabe_test"
const dr = H/150
const v_char = 2.0
const dt = 0.1*dr/v_char
const nframes = 400
const t_end = 10*dt #2.0

function print_info()
    @show Tu
    @show Td
    thermal_D = thermal_k/(rho0*gamma*cV) #thermal diffusivity
    thermal_a = 1.0/Tu
    nu = mu/rho0
    @show Pr = nu/thermal_D #Prandtl of air is about 0.71
    dT = Td - Tu
    contrast = dT/Tu
    @show Ra = thermal_a*g*dT*(H^3)/(nu*thermal_D)
    m = (g*H)/(R*Tu*contrast) 
    stability_index = (gamma - 1.0)*m
    @show stability_index
    @show R
    @match stability_index begin
        x, if x < 1.0 end => println("Stable")
        x, if x > 1.0 end => println("Unstable")
    end
    if (Tu >= Td)
        throw("Td should be bigger than Tu")
    end
end

function linear_atmo!(p::VoronoiPolygon)
    p.P = P0 + rho0*g*(H - p.x[2])
    p.rho = rho0
    p.e = p.P/(p.rho*(gamma-1.0))
    p.mass = p.rho*area(p)
    p.k = thermal_k
    p.cV = cV
    p.T = p.e/cV
end

function exp_atmo!(p::VoronoiPolygon)
    p.T = Tu
    p.P = P0*exp(-g*p.x[2]/(R*Tu))
    p.e = cV*p.T
    p.rho = p.P/(R*p.T)
    p.mass = p.rho*area(p)
    p.k = thermal_k
    p.cV = cV
end

function const_atmo!(p::VoronoiPolygon)
    p.T = Tu
    p.P = P0
    p.e = cV*p.T
    p.rho = p.P/((gamma-1.0)*p.e)
    p.mass = p.rho*area(p)
    p.k = thermal_k
    p.cV = cV
end

# boundary condition for temperature
function T_bc(::RealVector, bdary::Int)::Float64
    if (bdary == BDARY_DOWN)
        return Td 
    elseif (bdary == BDARY_UP)
        return Tu
    end
    return NaN
end

function enforce_bc!(grid::VoronoiGrid, dt::Float64, T_bc::Function)
    @batch for p in grid.polygons
        if !isboundary(p)
            continue
        end
        implicit_factor = 1.0
        thermal_D = thermal_k/(p.rho*gamma*cV)
        e_kinetic = 0.5*norm_squared(p.v)
        p.T = (p.e - e_kinetic)/cV
        A = area(p)
        for e in p.edges
            if !isboundary(e)
                continue
            end
            m = 0.5*(e.v1 + e.v2)
            n = normal_vector(e)
            lrr = len(e)/abs(dot(m - p.x, n)) + 0.01*dr
            T = T_bc(m, e.label)
            if !isnan(T)
                tmp = dt*thermal_D*lrr/A
                p.T += tmp*T
                implicit_factor += tmp
            end
        end
        p.T /= implicit_factor
        p.e = e_kinetic + cV*p.T
    end
end

mutable struct Simulation <: SimulationWorkspace
    grid::GridNSF
    solver::PressureSolver{PolygonNSF}
    E::Float64
    S::Float64
    E_kinetic::Float64
    Simulation() = begin
        xlims = (0.0, W)
        ylims = (0.0, H)
        domain = Rectangle(xlims = xlims, ylims = ylims)
        grid = GridNSF(domain, dr)
        populate_lloyd!(grid, ic! = exp_atmo!)
        return new(grid, PressureSolver(grid), 0.0, 0.0, 0.0)
    end
end

function step!(sim::Simulation, t::Float64)
    move!(sim.grid, dt)
    gravity_step!(sim.grid, -g*VECY, dt)
    ideal_eos!(sim.grid, gamma)
    find_pressure!(sim.solver, dt)
    pressure_step!(sim.grid, dt)
    ideal_temperature!(sim.grid)
    fourier_step!(sim.grid, dt)
    enforce_bc!(sim.grid, dt, T_bc)
    find_D!(sim.grid)
    viscous_step!(sim.grid, dt)
    find_dv!(sim.grid, dt)
    relaxation_step!(sim.grid, dt)
    return
end

function postproc!(sim::Simulation, t::Float64) 
    sim.S = 0.0
    sim.E = 0.0
    sim.E_kinetic = 0.0
    P0 = rho0*c0^2/gamma
    for p in sim.grid.polygons
        sim.S += p.mass*p.cV*(log(abs(p.P/P0)) - gamma*log(abs(p.rho/rho0)))
        sim.E += p.mass*p.e
        sim.E_kinetic += 0.5*p.mass*norm_squared(p.v)
    end
    @show sim.E
    @show sim.S
    @show sim.E_kinetic
end

function main()
    print_info()
    sim = Simulation()
    run!(sim, dt, t_end, step!; path = export_path, 
        vtp_vars = (:v, :P, :T, :rho),
        postproc! = postproc!,
        nframes = nframes
    )
end

end