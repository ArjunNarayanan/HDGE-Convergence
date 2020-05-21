using SparseArrays, LinearAlgebra, StaticArrays, Plots
using ImplicitDomainQuadrature
using Revise
using CartesianMesh
using HDGElasticity
include("apply_boundary_conditions.jl")


lambda = 1.0
mu = 2.0
x0 = [0.0,0.0]
widths = [1.0,1.0]
nelements = [1,1]
mesh = UniformMesh(x0,widths,nelements)

hooke_matrix = HDGElasticity.plane_strain_voigt_hooke_matrix(lambda,mu,2)
Dhalf = hooke_matrix^0.5
bc_displacement = 1e-2

dim = 2
sdim = HDGElasticity.symmetric_tensor_dim(dim)
polyorder = 1
quadorder = 2
stabilization = 1e2

basis = TensorProductBasis(2,polyorder)
surface_basis = TensorProductBasis(1,polyorder)
quad = TensorProductQuadratureRule(2,quadorder)
surface_quad = ReferenceQuadratureRule(quadorder)

NF = number_of_basis_functions(basis)
NHF = number_of_basis_functions(surface_basis)

jac = HDGElasticity.AffineMapJacobian(mesh)

lop = HDGElasticity.LocalOperator(basis,quad,mesh,Dhalf,stabilization)
lhop = HDGElasticity.LocalHybridCoupling(basis,surface_basis,surface_quad,mesh,Dhalf,stabilization)

bc_right = HDGElasticity.DisplacementComponentBC(surface_basis,surface_quad,
    bc_displacement,[1.0,0.0],jac.jac[2])
bc_bottom = HDGElasticity.DisplacementComponentBC(surface_basis,surface_quad,0.0,
    [0.0,-1.0],jac.jac[1])
bc_left = HDGElasticity.DisplacementComponentBC(surface_basis,surface_quad,0.0,
    [-1.0,0.0],jac.jac[2],penalty=1e4)

matrix = HDGElasticity.SystemMatrix()
rhs = HDGElasticity.SystemRHS()

HDGElasticity.assemble_local_operator!(matrix,vec(lop.local_operator),
    mesh.total_number_of_elements,dim,sdim,NF)

total_element_dofs = mesh.total_number_of_elements*
    HDGElasticity.get_dofs_per_element(dim,sdim,NF)

HDGElasticity.assemble_local_hybrid_operator!(matrix,
    vec.(lhop.local_hybrid_operator),mesh.face_to_hybrid_element_number,
    mesh.total_number_of_elements,mesh.faces_per_element,total_element_dofs,
    dim,sdim,NF,NHF)

HDGElasticity.assemble_hybrid_local_operator!(matrix,
    Array.(vec.(transpose.(lhop.local_hybrid_operator))),
    mesh.face_to_hybrid_element_number,mesh.face_indicator,
    mesh.total_number_of_elements,mesh.faces_per_element,total_element_dofs,
    dim,sdim,NF,NHF)

#
HDGElasticity.assemble_hybrid_mass_operator!(matrix,vec(lhop.UhatUhat),
    mesh.face_to_hybrid_element_number,mesh.face_indicator,
    mesh.total_number_of_elements,mesh.faces_per_element,total_element_dofs,
    dim,sdim,NHF)

apply_left_displacement_bc!(matrix,vec(bc_left.op),
    mesh.face_to_hybrid_element_number,total_element_dofs,mesh.nelements,
    dim,NHF)

apply_right_displacement_bc!(matrix,vec(bc_right.op),
    mesh.face_to_hybrid_element_number,total_element_dofs,mesh.nelements,dim,NHF)
apply_right_displacement_bc!(rhs,bc_right.rhs,mesh.face_to_hybrid_element_number,
    total_element_dofs,mesh.nelements,dim,NHF)

apply_bottom_displacement_bc!(matrix,vec(bc_bottom.op),
    mesh.face_to_hybrid_element_number,total_element_dofs,mesh.nelements,dim,NHF)

apply_top_traction_bc!(matrix,
    Array.(vec.(transpose.(lhop.local_hybrid_operator))),
    vec(lhop.UhatUhat),mesh.face_to_hybrid_element_number,total_element_dofs,
    mesh.nelements,dim,sdim,NF,NHF)

apply_bottom_traction_bc!(matrix,lhop,mesh.face_to_hybrid_element_number,
    total_element_dofs,nelements,dim,sdim,NF,NHF)

apply_left_traction_bc!(matrix,lhop,mesh.face_to_hybrid_element_number,
    total_element_dofs,nelements,dim,sdim,NF,NHF)
apply_right_traction_bc!(matrix,lhop,mesh.face_to_hybrid_element_number,
    total_element_dofs,nelements,dim,sdim,NF,NHF)


A = dropzeros!(sparse(matrix.rows,matrix.cols,matrix.vals,36,36))
F = dropzeros!(sparsevec(rhs.rows,rhs.vals,36))
Am = Array(A)
Fv = Array(F)
r = rank(Array(A))

D = Am\Fv
lsol = reshape(D[1:20],5,:)
hsol = reshape(D[21:end],2,:)
