#=

# Lid-driven cavity

Standard CFD benchmark, which allows comparison with very accurate mesh-based methods.
Description of the problem can be found on many webpages, like [here](https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node14.html).
In the image below, you can see streamlines for Re = 400 and N = 320. This was computed on cluster and took some time.
To correctly resolve corner vortices is much more demanding in SPH than in FEM.

```@raw html
	<img src='../assets/cavity.png' alt='missing' width="50%" height="50%" /><br>
```

Result is compared to the referential solution by [Ghia et al 1980](https://www.sciencedirect.com/science/article/pii/0021999182900584).
=#

module cavity_flow

using Printf
using SmoothedParticles
using CSV, DataFrames, Printf, Parameters, Match

#=
### Declare const parameters (dimensionless problem)
=#

##geometrical/physical parameters
const N  =   80                #number of sample points
const Re =   100                #Reynolds number
const llid = 0.8               #length of the lid
const mu =   1.0/Re             #viscosity
const rho0 = 1.0                #density
const vlid = 1.0                #flow speed of the lid
const dr = 1.0/N 		        #interparticle distance
const h = 3.0*dr		        #size of kernel support
const m = rho0*dr^2             #particle mass
const c = 20*vlid		    	#numerical speed of sound
const P0 = 5.0                  #background pressure (good to prevent tensile instability)
const wwall = h

##temporal parameters
const dt = 0.1*h/c                       #numerical time-step
const t_end = 0.4*pi                   #end of simulation
const dt_frame = max(dt, t_end/50)      #how often save data

##particle types
const FLUID = 0.
const WALL = 1.
const LID = 2.

##path to store results
const path = "results/gresho_SPH"

#=
### Declare variables to be stored in a Particle
=#

@with_kw mutable struct Particle <: AbstractParticle
	x::RealVector=VEC0	#position
	v::RealVector=VEC0  #velocity
	Dv::RealVector=VEC0 #acceleratation
	rho::Float64=rho0   #density
	Drho::Float64=0.0   #rate of density
	P::Float64=0.0      #pressure
	type::Float64       #particle type
end

function initial_state!(p::Particle)
	omega, P = @match norm(p.x) begin
		r, if r < 0.2 end => (5.0, 5.0 + 12.5*r^2)
		r, if r < 0.4 end => (2.0/r - 5.0, 9.0 + 12.5*r^2 - 20.0*r + 4.0*log(r/0.2))
		_ => (0.0, 3.0 + 4.0*log(2.0))
	end
	p.v = omega*RealVector(-p.x[2], p.x[1], 0.0)
	p.rho = rho0 + P/c^2
end

#=
### Define geometry and create particles
=#

function make_system()
	grid = Grid(dr, :square)
	box = Rectangle(-llid, -llid, llid, llid)
	wall = BoundaryLayer(box, grid, wwall)
	sys = ParticleSystem(Particle, box + wall, h)
	#generate_particles!(sys, grid, box, x -> Particle(x=x, type=FLUID))
	#generate_particles!(sys, grid, wall, x -> Particle(x=x, type=WALL))
	for r in (0.5*dr):dr:sqrt(2)
        k_max = round(Int, 2.0*pi*r*N)
        for k in 1:k_max
            theta = 2.0*pi*k/k_max
            x = RealVector(r*cos(theta), r*sin(theta), 0.0)
            if is_inside(x, box)
                push!(sys.particles, Particle(x=x, type=FLUID))
            elseif is_inside(x, wall)
				push!(sys.particles, Particle(x=x, type=WALL))
			end
        end
    end
	create_cell_list!(sys)
	apply!(sys, initial_state!)
	apply!(sys, find_pressure!)
	apply!(sys, internal_force!) 
	return sys
end

#=
### Define interactions between particles
=#

function balance_of_mass!(p::Particle, q::Particle, r::Float64)
	p.Drho += m*rDwendland2(h,r)*(dot(p.x-q.x, p.v-q.v))
end

function find_pressure!(p::Particle)
	p.rho += p.Drho*dt
	p.Drho = 0.0
	p.P = P0 + c^2*(p.rho-rho0)
end

function internal_force!(p::Particle, q::Particle, r::Float64)
	rDk = rDwendland2(h,r)
	x_pq = p.x - q.x
	p.Dv += -m*rDk*(p.P/p.rho^2 + q.P/q.rho^2)*x_pq
end


function move!(p::Particle)
	p.Dv = VEC0
	if p.type == FLUID
		p.x += 0.5*dt*p.v
	end
end

function accelerate!(p::Particle)
	if p.type == FLUID
		p.v += 0.5*dt*p.Dv
	end
end

#=
### Time iteration
=#

function main()
	sys = make_system()
	out = new_pvd_file(path)
	for k = 0 : Int64(round(t_end/dt))
		begin
			apply!(sys, accelerate!)
			apply!(sys, move!)
			create_cell_list!(sys)
			apply!(sys, balance_of_mass!)
			apply!(sys, find_pressure!)
			apply!(sys, move!)
			create_cell_list!(sys)
			apply!(sys, internal_force!)
			apply!(sys, accelerate!)
		end
		if (k % Int64(round(dt_frame/dt)) == 0) #save the frame
			@printf("t = %.6e s ", k*dt)
			println("(",round(100*k*dt/t_end),"% complete)")
			save_frame!(out, sys, :P, :v, :type)
		end
	end
	save_pvd_file(out)
end

end
