using LinearAlgebra, SparseArrays, GraphPlot, LightGraphs
using Revise
using CartesianMesh






x0 = [0.0,0.0]
widths = [1.0,2.0]
nelements = [1,2]
mesh = UniformMesh(x0,widths,nelements)
