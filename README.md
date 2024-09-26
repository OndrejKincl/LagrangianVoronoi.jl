# LagrangainVoronoi.jl

<img src="docs/src/assets/voronoimesh.png" alt="" style="height: 400px"/>

[Documentation](https://github.com/OndrejKincl/LagrangianVoronoi.jl/blob/gh-pages/index.html)

A numerical library for hydrodynamic simulations using Lagrangian Voronoi method in [Julia](https://julialang.org/). The idea is to enhance [SPH](https://en.wikipedia.org/wiki/Smoothed-particle_hydrodynamics) using a moving [Voronoi mesh](https://en.wikipedia.org/wiki/Voronoi_diagram). This allows for consistent gradient approximations and simplifies some boundary conditions. Also, the Voronoi mesh makes a smaller stencil, which means that the interactions between particles are in some sense *more* local, leading to sparser matrices and faster computations. The code runs reasonable well in parallel (shared memory) and uses semi-implicit time-marching scheme. It can handle both compressible and incompressible fluids, shocks, multi-phase flows and fluids with heat conduction. Periodic, no-slip and free-slip boundary conditions are supported.

The main limitation is the lack of surface tension, free surface conditions and difficulties with mutli-phase problems featuring high density ratios (for example in water and air interface, the density ratio is 800:1, and that is challenging for the numerics). Also, the code is currently only 2D and we cannot prescribe inflow and outflow conditions. We hope to adress these problems in near future. The computational domain must be a rectangle, but complex geometries can be implemented indirectly with the help of dummy particles (TODO).

If you are interested in the beautiful math behind this method, you can have a look at [our paper](https://arxiv.org/abs/2405.04116).

If you have any suggestions or comments, feel free to raise an issue via the GitHub interface or contact me at ondrej.kincl@unitn.it. 


## Install prerequisites
Run the following command in Julia terminal:
```
using Pkg
deps = [
  "Polyester",
  "WriteVTK",
  "CSV",
  "StaticArrays",
  "SparseArrays",
  "Match",
  "Parameters",
  "DataFrames",
  "Plots",
  "LaTeXStrings",
  "Krylov",
  "Measures",
  "SmoothedParticles"
]
Pkg.add(deps)
```
## Quick start
Clone this repository. Open the `examples` folder and run this from your terminal:
```
julia -t 1 gresho.jl
```
You may replace `1` with the number of threads available on your PC. Wait for the simulation to finish. To see the results, download [paraview](https://www.paraview.org/download/). Use it to open the `result.pvd` file. You may try any other example but some require long time to compute. Also note pictures in documentation are sometimes made with higher than default resolution.

## Citing
If you use this code in your thesis or research paper, please cite: 
```
@article{kincl2024semi,
  title={Semi-implicit Lagrangian Voronoi Approximation for the incompressible Navier-Stokes equations},
  author={Kincl, Ond{\v{r}}ej and Peshkov, Ilya and Boscheri, Walter},
  journal={arXiv preprint arXiv:2405.04116},
  year={2024}
}
```
## Other libraries
You can also check my SPH library [SmoothedParticles.jl](https://github.com/OndrejKincl/SmoothedParticles.jl).
