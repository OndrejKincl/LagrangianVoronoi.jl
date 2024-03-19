module gresho

using Printf
using SmoothedParticles
using Plots, CSV, DataFrames, Printf, LaTeXStrings, Parameters, Match

#=
### Declare const parameters (dimensionless problem)
=#

const v_char = 1.0
const c_sound = 10.0*v_char
const rho0 = 1.0
const xlims = (-1.0, 1.0)
const ylims = (-1.0, 1.0)
const N = 100 #resolution
const dr = 1.0/N
const m = rho0*dr^2
const h = 2.4*dr
const wwall = h

const dt = 0.2*dr/v_char
const t_end =  0.5
const nframes = 100
const dt_frame = max(dt, t_end/nframes)
const P_max = 3.0 + 4.0*log(2.0)

##particle types
const FLUID = 0.
const WALL = 1.

##path to store results
const path = "results/gresho_sph"

#=
### Declare variables to be stored in a Particle
=#

function v_exact(x::RealVector)::RealVector
    omega = @match norm(x) begin
        r, if r < 0.2 end => 5.0
        r, if r < 0.4 end => 2.0/r - 5.0
        _ => 0.0
    end
    return omega*RealVector(-x[2], x[1], 0.0)
end

function P_exact(x::RealVector)::Float64
    P = @match norm(x) begin
            r, if r < 0.2 end => 5.0 + 12.5*r^2
            r, if r < 0.4 end => 9.0 + 12.5*r^2 - 20.0*r + 4.0*log(5.0*r)
            _ => 3.0 + 4.0*log(2.0)
        end
    return P
end

@with_kw mutable struct Particle <: AbstractParticle
	x::RealVector=VEC0	#position
	v::RealVector=VEC0  #velocity
	Dv::RealVector=VEC0 #acceleratation
	rho::Float64=0.0  #density
	rho_C::Float64=0.0   #density constant
	P::Float64=0.0      #pressure
    P0::Float64=0.0     #initial pressure
	type::Float64       #particle type
    E_wall::Float64=0.0      #wall energy
end

#=
### Define geometry and create particles
=#

function make_system()
	grid = Grid(dr, :hexagonal)
	box = Rectangle(xlims[1], ylims[1], xlims[2], ylims[2])
	wall = BoundaryLayer(box, grid, wwall)
	sys = ParticleSystem(Particle, box + wall, h)
	generate_particles!(sys, grid, box, x -> Particle(x=x, type=FLUID))
	generate_particles!(sys, grid, wall, x -> Particle(x=x, type=WALL))
	create_cell_list!(sys)
    apply!(sys, find_rho!, self=true)
    for p in sys.particles
        p.v = v_exact(p.x)
        p.P0 = P_exact(p.x)
        p.rho_C = rho0 - p.rho
        p.rho = rho0
    end
	apply!(sys, find_pressure!)
	apply!(sys, internal_force!) 
	return sys
end

#=
### Define interactions between particles
=#

function reset_rho!(p::Particle)
    p.rho = p.rho_C
end

function find_rho!(p::Particle, ::Particle, r::Float64)
	p.rho += m*wendland2(h,r)
end

function find_pressure!(p::Particle)
	p.P = p.P0 + c_sound^2*(p.rho - rho0)
end

function internal_force!(p::Particle, q::Particle, r::Float64)
	rDk = rDwendland2(h,r)
	x_pq = p.x - q.x
	#v_pq = p.v - q.v
	p.Dv += -m*rDk*(p.P/p.rho^2 + q.P/q.rho^2)*x_pq
	#p.Dv += 8/(Re*p.rho*q.rho)*m*rDk*dot(v_pq, x_pq)/(r^2 + 0.01*h^2)*x_pq #Monaghan's type of viscosity to conserve angular momentum
end


function move!(p::Particle)
	p.Dv = VEC0
	if p.type == FLUID
		p.x += 0.5*dt*p.v
	end
end

function accelerate!(p::Particle)
	p.v += 0.5*dt*p.Dv
	if p.type == WALL
        p.E_wall += 0.25*m*dot(p.v, p.v)
        p.v = VEC0
    end
end

#=
### Time iteration
=#

function get_energy(sys::ParticleSystem)::Float64
    E = 0.0
    for p in sys.particles
        E += 0.5*m*dot(p.v, p.v)
        E += m*c_sound^2*(log(abs(p.rho/rho0)) + (rho0 + (P_max - p.P0)/c_sound^2)/p.rho - 1.0)
        E += p.E_wall
    end
    return E
end

function main()
	sys = make_system()
	out = new_pvd_file(path)
    E = Float64[]
    println("energy = ", get_energy(sys))
	@time for k = 0 : Int64(round(t_end/dt))
		apply!(sys, accelerate!)
		apply!(sys, move!)
		create_cell_list!(sys)
		apply!(sys, reset_rho!)
		apply!(sys, find_rho!, self=true)
        apply!(sys, find_pressure!)
        apply!(sys, internal_force!)
		if (k % Int64(round(dt_frame/dt)) == 0) #save the frame
			@printf("t = %.6e s ", k*dt)
            push!(E, get_energy(sys))
            println("energy error = ", E[end]/E[1] - 1.0)
			println("(",round(100*k*dt/t_end),"% complete)")
			save_frame!(out, sys, :P, :v, :type, :rho, :rho_C)
		end
		apply!(sys, accelerate!)
	end
	save_pvd_file(out)
end

end