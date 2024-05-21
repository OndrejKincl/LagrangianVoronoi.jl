# LagrangainVoronoi.jl
A numerical library for Lagrangian Voronoi method in Julia. 

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
Clone this repository. Open the test folder and run from terminal/console:
```
julia -t 1 gresho.jl
```
You may replace 1 with the number of threads available on your computer. Wait for the simulation to finish. To see the results, download [paraview](https://www.paraview.org/download/). Use it to open the `result.pvd` file. 

## Citing
If you use our code in your thesis or research paper, please cite: 
```
@article{kincl2024semi,
  title={Semi-implicit Lagrangian Voronoi Approximation for the incompressible Navier-Stokes equations},
  author={Kincl, Ond{\v{r}}ej and Peshkov, Ilya and Boscheri, Walter},
  journal={arXiv preprint arXiv:2405.04116},
  year={2024}
}
```
