using SparseArrays, LinearAlgebra, StaticArrays, Plots
using ImplicitDomainQuadrature
using Revise
using CartesianMesh
using HDGElasticity

function affine_map(xi::T,xL::T,xR::T) where {T<:Real}
    return xL + (xR-xL)*(xi+1.0)*0.5
end

function affine_map(xi::T,xL::AbstractVector,xR::AbstractVector) where {T<:Real}
    return xL + (xR-xL)*(xi+1.0)*0.5
end

function hybrid_element_points(surface_basis,x0,widths)
    dx = [widths[1],0.0]
    dy = [0.0,widths[2]]
    p1 = affine_map.(surface_basis.points,x0,x0+dx)
    p2 = affine_map.(surface_basis.points,x0+dx,x0+widths)
    p3 = affine_map.(surface_basis.points,x0+dy,x0+widths)
    p4 = affine_map.(surface_basis.points,x0,x0+dy)
    return hcat(p1,p2,p3,p4)
end

function required_quadrature_order(polyorder)
    quadorder = ceil(Int,polyorder/2)+1
    quadorder = isodd(quadorder) ? quadorder + 1 : quadorder
    return quadorder
end

lambda = 1.0
mu = 2.0
x0 = [0.0,0.0]
widths = [2.0,1.0]
nelements = [1,1]
mesh = UniformMesh(x0,widths,nelements)

hooke_matrix = HDGElasticity.plane_strain_voigt_hooke_matrix(lambda,mu,2)
Dhalf = hooke_matrix^0.5

dim = 2
sdim = HDGElasticity.symmetric_tensor_dim(dim)
polyorder = 1
quadorder = required_quadrature_order(2polyorder)
stabilization = 3.0

basis = TensorProductBasis(2,polyorder)
surface_basis = TensorProductBasis(1,polyorder)
quad = TensorProductQuadratureRule(2,quadorder)
surface_quad = TensorProductQuadratureRule(1,quadorder)

NF = HDGElasticity.number_of_basis_functions(basis)
NHF = HDGElasticity.number_of_basis_functions(surface_basis)

jac = HDGElasticity.AffineMapJacobian(mesh,quad)

lop = HDGElasticity.LocalOperator(basis,quad,surface_quad,
    Dhalf,jac,stabilization)
lhop = HDGElasticity.LocalHybridCoupling(basis,surface_basis,surface_quad,
    Dhalf,jac,stabilization)

local_hybrid_operator = [[lhop.LH[i];lhop.UH[i]] for i = 1:4]
Ahat = hcat([v for v in local_hybrid_operator]...)

hpoints = hybrid_element_points(surface_basis,x0,widths)
uhat = similar(hpoints)
uhat[1,:] = 1e-2*hpoints[1,:]
uhat[2,:] .= 0.0
uhat = vec(uhat)

rhs = Ahat*uhat
sol = lop.local_operator\rhs

stressdofs = 1:(3*NF)
dispdofs = (stressdofs[end]+1):(stressdofs[end]+2*NF)
stress = -Dhalf*reshape(sol[stressdofs],3,:)
disp = reshape(sol[dispdofs],2,:)
