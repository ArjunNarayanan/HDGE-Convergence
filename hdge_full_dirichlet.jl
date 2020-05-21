using SparseArrays, LinearAlgebra, StaticArrays, PyPlot
using DataFrames, CSV
using ImplicitDomainQuadrature
# using Revise
using CartesianMesh
using HDGElasticity
include("full_dirichlet_bc.jl")

function get_stress_dofs(total_num_elements,dim,sdim,NF)
    stressdofs = Int[]
    for elid in 1:total_num_elements
        dofs = HDGElasticity.element_stress_dofs(elid,dim,sdim,NF)
        append!(stressdofs,dofs)
    end
    return stressdofs
end

function get_displacement_dofs(total_num_elements,dim,sdim,NF)
    dispdofs = Int[]
    for elid in 1:total_num_elements
        dofs = HDGElasticity.element_displacement_dofs(elid,dim,sdim,NF)
        append!(dispdofs,dofs)
    end
    return dispdofs
end

function get_hybrid_dofs(num_elements,num_hybrid_elements,dim,sdim,NF,NHF)
    total_element_dofs = num_elements*
        HDGElasticity.get_dofs_per_element(dim,sdim,NF)
    start = total_element_dofs+1
    stop = start+num_hybrid_elements*NHF*dim-1
    return start:stop
end

function get_total_mesh_dofs(total_element_dofs,num_hybrid_elements,
    dim,sdim,NF,NHF)

    total_hybrid_dofs = num_hybrid_elements*NHF*dim
    return total_element_dofs+total_hybrid_dofs
end

function required_quadrature_order(polyorder)
    quadorder = ceil(Int,polyorder/2)+1
    quadorder = isodd(quadorder) ? quadorder + 1 : quadorder
    return quadorder
end

function affine_map(xi,xL,xR)
    return xL .+ 0.5*(xi .+ 1.0).*(xR-xL)
end

function element_bottom_left(elx,ely,x0,element_size)
    xL = x0 + [(elx-1)*element_size[1],(ely-1)*element_size[2]]
end

function get_element_node_coordinates(elx,ely,x0,element_size,basis)

    xL = element_bottom_left(elx,ely,x0,element_size)
    xR = element_bottom_left(elx+1,ely+1,x0,element_size)

    mapped_points = Matrix{Float64}(undef,2,0)

    for idx in 1:size(basis.points)[2]
        coords = affine_map(basis.points[:,idx],xL,xR)
        mapped_points = hcat(mapped_points,coords)
    end
    return mapped_points
end

function get_mesh_node_coordinates(x0,nelements,element_size,basis)

    node_coordinates = Matrix{Float64}(undef,2,0)

    for elx in 1:nelements[1]
        for ely in 1:nelements[2]
            coords = get_element_node_coordinates(elx,ely,x0,element_size,basis)
            node_coordinates = hcat(node_coordinates,coords)
        end
    end
    return node_coordinates
end

function get_mesh_node_coordinates(mesh,basis)
    return get_mesh_node_coordinates(mesh.x0,mesh.nelements,mesh.element_size,basis)
end

function assemble_global_operator!(matrix,lop,lhop,mesh,
    total_element_dofs,dim,sdim,NF,NHF)

    HDGElasticity.assemble_local_operator!(matrix,vec(lop.local_operator),
        1:mesh.total_number_of_elements,dim,sdim,NF)

    local_hybrid_operator = [[-lhop.LH[i];-lhop.UH[i]] for i = 1:4]

    HDGElasticity.assemble_local_hybrid_operator!(matrix,
        vec.(local_hybrid_operator),mesh.face_to_hybrid_element_number,
        1:mesh.total_number_of_elements,1:mesh.faces_per_element,
        total_element_dofs,dim,sdim,NF,NHF)

    HDGElasticity.assemble_hybrid_local_operator!(matrix,
        Array.(vec.(transpose.(local_hybrid_operator))),
        mesh,total_element_dofs,NF,NHF)
    HDGElasticity.assemble_hybrid_operator!(matrix,vec.(lhop.HH),
        mesh,total_element_dofs,NHF)

end

function assemble_boundary_condition!(matrix,rhs,hybrid_coupling,displacement,
    surface_basis,surface_quad,mesh,total_element_dofs,dim,NHF,penalty,jac)

    bcop = penalty*vec.(hybrid_coupling)
    surface_basis_vals = [surface_basis(p) for p in surface_quad.points]

    apply_bottom_displacement_bc!(matrix,bcop[1],
        mesh.face_to_hybrid_element_number,total_element_dofs,
        mesh.nelements,dim,NHF)
    apply_right_displacement_bc!(matrix,bcop[2],
        mesh.face_to_hybrid_element_number,total_element_dofs,
        mesh.nelements,dim,NHF)
    apply_top_displacement_bc!(matrix,bcop[3],
        mesh.face_to_hybrid_element_number,total_element_dofs,
        mesh.nelements,dim,NHF)
    apply_left_displacement_bc!(matrix,bcop[4],
        mesh.face_to_hybrid_element_number,total_element_dofs,
        mesh.nelements,dim,NHF)

    apply_bottom_displacement_bc!(rhs,displacement,surface_basis_vals,
        surface_quad,mesh.face_to_hybrid_element_number,mesh.x0,
        mesh.widths,mesh.nelements,mesh.element_size,total_element_dofs,
        dim,NHF,penalty,jac)
    apply_right_displacement_bc!(rhs,displacement,surface_basis_vals,
        surface_quad,mesh.face_to_hybrid_element_number,mesh.x0,
        mesh.widths,mesh.nelements,mesh.element_size,total_element_dofs,
        dim,NHF,penalty,jac)
    apply_top_displacement_bc!(rhs,displacement,surface_basis_vals,
        surface_quad,mesh.face_to_hybrid_element_number,mesh.x0,
        mesh.widths,mesh.nelements,mesh.element_size,total_element_dofs,
        dim,NHF,penalty,jac)
    apply_left_displacement_bc!(rhs,displacement,surface_basis_vals,
        surface_quad,mesh.face_to_hybrid_element_number,mesh.x0,
        mesh.widths,mesh.nelements,mesh.element_size,total_element_dofs,
        dim,NHF,penalty,jac)


end


function assemble_body_force!(rhs,body_force,basis,quad,mesh,dim,NF,jac)

    basis_vals = [basis(p) for (p,w) in quad]
    apply_body_force!(rhs,body_force,basis_vals,quad,mesh.x0,
        mesh.nelements,mesh.element_size,dim,NF,jac)

end

function solve_full_dirichlet(mesh,Dhalf,body_force,displacement,
    basis::TensorProductBasis{dim},surface_basis,quad,
    surface_quad,stabilization,penalty) where {dim}

    sdim = HDGElasticity.symmetric_tensor_dim(dim)

    NF = HDGElasticity.number_of_basis_functions(basis)
    NHF = HDGElasticity.number_of_basis_functions(surface_basis)

    total_element_dofs = mesh.total_number_of_elements*
        HDGElasticity.get_dofs_per_element(dim,sdim,NF)

    jac = HDGElasticity.AffineMapJacobian(mesh,quad)
    lop = HDGElasticity.LocalOperator(basis,quad,surface_quad,Dhalf,jac,
        stabilization)
    lhop = HDGElasticity.LocalHybridCoupling(basis,surface_basis,
        surface_quad,Dhalf,jac,stabilization)
    hybrid_coupling = HDGElasticity.get_hybrid_coupling(surface_basis,
        surface_quad,1.0,jac)

    matrix = HDGElasticity.SystemMatrix()
    rhs = HDGElasticity.SystemRHS()

    assemble_global_operator!(matrix,lop,lhop,mesh,total_element_dofs,dim,
        sdim,NF,NHF)
    assemble_body_force!(rhs,body_force,basis,quad,mesh,dim,NF,jac)
    assemble_boundary_condition!(matrix,rhs,hybrid_coupling,displacement,
        surface_basis,surface_quad,mesh,total_element_dofs,
        dim,NHF,penalty,jac)

    total_dofs = get_total_mesh_dofs(total_element_dofs,
        mesh.total_number_of_hybrid_elements,dim,sdim,NF,NHF)
    op = dropzeros!(sparse(matrix.rows,matrix.cols,matrix.vals,total_dofs,total_dofs))
    F = dropzeros!(sparsevec(rhs.rows,rhs.vals,total_dofs))

    sol = op\Array(F)

    stressdofs = get_stress_dofs(mesh.total_number_of_elements,dim,sdim,NF)
    dispdofs = get_displacement_dofs(mesh.total_number_of_elements,dim,sdim,NF)
    hybdofs = get_hybrid_dofs(mesh.total_number_of_elements,
        mesh.total_number_of_hybrid_elements,dim,sdim,NF,NHF)

    stress = -Dhalf*reshape(sol[stressdofs],sdim,:)
    disp = reshape(sol[dispdofs],dim,:)
    hsol = reshape(sol[hybdofs],dim,:)

    return stress,disp,hsol
end

function compute_element_error(elmt_coeffs,basis_vals,error_quad,
    exact_solution,jac,xL,xR)

    numrows = size(elmt_coeffs)[1]
    elmt_error = zeros(numrows)

    for (idx,(p,w)) in enumerate(error_quad)
        nsol = elmt_coeffs*basis_vals[idx]
        x = affine_map(p,xL,xR)
        esol = exact_solution(x)
        elmt_error .+= (nsol - esol).^2*jac.detjac*w
    end
    return elmt_error
end

function integrate(exact_solution,error_quad,numrows,jac,xL,xR)

    val = zeros(numrows)

    for (p,w) in error_quad
        x = affine_map(p,xL,xR)
        esol = exact_solution(x)
        val .+= esol.^2*jac.detjac*w
    end
    return val
end

function element_dof_start(elid,dofs_per_element)
    return (elid-1)*dofs_per_element + 1
end

function element_dof_stop(elid,dofs_per_element)
    return elid*dofs_per_element
end

function compute_relative_error(coeffs,basis::TensorProductBasis{dim,T,NF},
    error_quad,exact_solution,x0,nelements,element_size,jac) where {dim,T,NF}

    numrows = size(coeffs)[1]
    err = zeros(numrows)
    normalizer = zeros(numrows)

    basis_vals = [basis(p) for (p,w) in error_quad]

    elid = 1
    for elx in 1:nelements[1]
        for ely in 1:nelements[2]
            xL = element_bottom_left(elx,ely,x0,element_size)
            xR = element_bottom_left(elx+1,ely+1,x0,element_size)
            edofs = element_dof_start(elid,NF):element_dof_stop(elid,NF)

            err += compute_element_error(coeffs[:,edofs],basis_vals,
                error_quad,exact_solution,jac,xL,xR)
            normalizer += integrate(exact_solution,error_quad,numrows,
                jac,xL,xR)

            elid += 1
        end
    end

    relative_error = sqrt.(err) ./ sqrt.(normalizer)
    return relative_error

end


function compute_displacement_error(nelements::Int64,polyorder)

    dim = 2
    sdim = HDGElasticity.symmetric_tensor_dim(dim)
    quadorder = required_quadrature_order(2polyorder)

    mesh = UniformMesh(x0,widths,[nelements,nelements])
    hooke_matrix = HDGElasticity.plane_strain_voigt_hooke_matrix(lambda,mu,2)
    Dhalf = hooke_matrix^0.5

    basis = TensorProductBasis(dim,polyorder)
    surface_basis = TensorProductBasis(dim-1,polyorder)
    quad = TensorProductQuadratureRule(dim,quadorder)
    surface_quad = TensorProductQuadratureRule(dim-1,quadorder)
    error_quad = TensorProductQuadratureRule(dim,quadorder+2)
    jacobian = HDGElasticity.AffineMapJacobian(mesh,quad)

    stress,disp,hsol = solve_full_dirichlet(mesh,Dhalf,
        x->test_body_force(x,lambda,mu,displacement_amplitude),
        x->test_displacement_field(x,displacement_amplitude),
        basis,surface_basis,quad,surface_quad,stabilization,penalty)

    disp_error = compute_relative_error(disp,basis,error_quad,
        x->test_displacement_field(x,displacement_amplitude),mesh.x0,
        mesh.nelements,mesh.element_size,jacobian)

    return disp_error,mesh.element_size[1]

end

function compute_stress_error(nelements::Int64,polyorder)

    dim = 2
    sdim = HDGElasticity.symmetric_tensor_dim(dim)
    quadorder = required_quadrature_order(2polyorder)

    mesh = UniformMesh(x0,widths,[nelements,nelements])
    hooke_matrix = HDGElasticity.plane_strain_voigt_hooke_matrix(lambda,mu,2)
    Dhalf = hooke_matrix^0.5

    basis = TensorProductBasis(dim,polyorder)
    surface_basis = TensorProductBasis(dim-1,polyorder)
    quad = TensorProductQuadratureRule(dim,quadorder)
    surface_quad = TensorProductQuadratureRule(dim-1,quadorder)
    error_quad = TensorProductQuadratureRule(dim,quadorder+2)
    jacobian = HDGElasticity.AffineMapJacobian(mesh,quad)

    stress,disp,hsol = solve_full_dirichlet(mesh,Dhalf,
        x->test_body_force(x,lambda,mu,displacement_amplitude),
        x->test_displacement_field(x,displacement_amplitude),
        basis,surface_basis,quad,surface_quad,stabilization,penalty)

    stress_error = compute_relative_error(stress,basis,error_quad,
        x->test_stress_field(x,lambda,mu,displacement_amplitude),mesh.x0,
        mesh.nelements,mesh.element_size,jacobian)

    return stress_error,mesh.element_size[1]

end

function save_displacement_error(nelements,polyorder)

    error_and_step_size = [compute_displacement_error(n,polyorder) for n in nelements]

    u1_error = [e[1][1] for e in error_and_step_size]
    u2_error = [e[1][2] for e in error_and_step_size]
    dx = [e[2] for e in error_and_step_size]

    filename = "displacement-convergence-order-"*string(polyorder)*".txt"
    df = DataFrame(u1 = u1_error, u2 = u2_error, dx = dx)
    CSV.write(filename,df)

end

function save_stress_error(nelements,polyorder)

    error_and_step_size = [compute_stress_error(n,polyorder) for n in nelements]

    s1_error = [e[1][1] for e in error_and_step_size]
    s2_error = [e[1][2] for e in error_and_step_size]
    s3_error = [e[1][2] for e in error_and_step_size]
    dx = [e[2] for e in error_and_step_size]

    filename = "stress-convergence-order-"*string(polyorder)*".txt"
    df = DataFrame(s1 = s1_error, s2 = s2_error, s3 = s3_error, dx = dx)
    CSV.write(filename,df)

end

const lambda = 1.0
const mu = 2.0
const displacement_amplitude = 0.01
const x0 = [0.0,0.0]
const widths = [1.0,1.0]
const stabilization = 10.0
const penalty = 1.0

polyorder = 4
nelements = [2,4,8,16,32]

save_stress_error(nelements,polyorder)
# save_displacement_error(nelements,polyorder)


# error_and_step_size = [compute_displacement_error(n,polyorder) for n in nelements]

# err,dx = compute_displacement_error(2,polyorder)
# nelements = [2,2]
# mesh = UniformMesh(x0,widths,nelements)
#
# hooke_matrix = HDGElasticity.plane_strain_voigt_hooke_matrix(lambda,mu,2)
# Dhalf = hooke_matrix^0.5
#
# dim = 2
# sdim = HDGElasticity.symmetric_tensor_dim(dim)
# polyorder = 2
# quadorder = required_quadrature_order(2polyorder)
# stabilization = 10.0
# penalty = 1.0
#
# basis = TensorProductBasis(dim,polyorder)
# surface_basis = TensorProductBasis(dim-1,polyorder)
# quad = TensorProductQuadratureRule(dim,quadorder)
# surface_quad = TensorProductQuadratureRule(dim-1,quadorder)
# error_quad = TensorProductQuadratureRule(dim,quadorder+2)
# jacobian = HDGElasticity.AffineMapJacobian(mesh,quad)
#
# stress,disp,hsol = solve_full_dirichlet(mesh,Dhalf,
#     x->test_body_force(x,lambda,mu,displacement_amplitude),
#     x->test_displacement_field(x,displacement_amplitude),
#     basis,surface_basis,quad,surface_quad,stabilization,penalty)
#
#
# disp_error = compute_relative_error(disp,basis,error_quad,
#     x->test_displacement_field(x,displacement_amplitude),mesh.x0,
#     mesh.nelements,mesh.element_size,jacobian)
#
# println("error = ",disp_error)

# basis_vals = [basis(p) for (p,w) in error_quad]
# err = compute_element_error(disp[:,[1,2,3,4]],basis_vals,
#     error_quad,x->test_displacement_field(x,displacement_amplitude),
#     jacobian,[0.0,0.0],[0.1,0.1])

# node_coordinates = get_mesh_node_coordinates(mesh,basis)
# fig,ax = PyPlot.subplots()
# tcf = ax.tricontourf(node_coordinates[1,:],node_coordinates[2,:],
#     disp[1,:],20)
# fig.colorbar(tcf)
# fig


# NF = HDGElasticity.number_of_basis_functions(basis)
# NHF = HDGElasticity.number_of_basis_functions(surface_basis)
#
# total_element_dofs = mesh.total_number_of_elements*
#     HDGElasticity.get_dofs_per_element(dim,sdim,NF)
#
# jac = HDGElasticity.AffineMapJacobian(mesh,quad)
#
# lop = HDGElasticity.LocalOperator(basis,quad,surface_quad,Dhalf,
#     jac,stabilization)
# lhop = HDGElasticity.LocalHybridCoupling(basis,surface_basis,
#     surface_quad,Dhalf,jac,stabilization)
#
#
#
# matrix = HDGElasticity.SystemMatrix()
# rhs = HDGElasticity.SystemRHS()
#
# HDGElasticity.assemble_local_operator!(matrix,vec(lop.local_operator),
#     1:mesh.total_number_of_elements,dim,sdim,NF)
#
# basis_vals = [basis(p) for (p,w) in quad]
# apply_body_force!(rhs,x->test_body_force(x,lambda,mu,displacement_amplitude),
#     basis_vals,quad,mesh.x0,mesh.nelements,mesh.element_size,dim,NF,jac)
#
# local_hybrid_operator = [[-lhop.LH[i];-lhop.UH[i]] for i = 1:4]
#
# HDGElasticity.assemble_local_hybrid_operator!(matrix,
#     vec.(local_hybrid_operator),mesh.face_to_hybrid_element_number,
#     1:mesh.total_number_of_elements,1:mesh.faces_per_element,
#     total_element_dofs,dim,sdim,NF,NHF)
#
# HDGElasticity.assemble_hybrid_local_operator!(matrix,
#     Array.(vec.(transpose.(local_hybrid_operator))),
#     mesh,total_element_dofs,NF,NHF)
# total_element_dofs,
#
# HDGElasticity.assemble_hybrid_operator!(matrix,vec.(lhop.HH),
#     mesh,total_element_dofs,NHF)
#
# hybrid_mass = HDGElasticity.get_hybrid_coupling(surface_basis,surface_quad,1.0,jac)
# penalty = 1.0
# bc_op = penalty*vec.(hybrid_mass)
# apply_bottom_displacement_bc!(matrix,bc_op[1],
#     mesh.face_to_hybrid_element_number,total_element_dofs,mesh.nelements,
#     dim,NHF)
# apply_right_displacement_bc!(matrix,bc_op[2],
#     mesh.face_to_hybrid_element_number,total_element_dofs,mesh.nelements,
#     dim,NHF)
# apply_top_displacement_bc!(matrix,bc_op[3],
#     mesh.face_to_hybrid_element_number,total_element_dofs,mesh.nelements,
#     dim,NHF)
# apply_left_displacement_bc!(matrix,bc_op[4],
#     mesh.face_to_hybrid_element_number,total_element_dofs,mesh.nelements,
#     dim,NHF)
#
# surface_basis_vals = [surface_basis(p) for p in surface_quad.points]
#
# apply_bottom_displacement_bc!(rhs,
#     x->test_displacement_field(x,displacement_amplitude),
#     surface_basis_vals,surface_quad,mesh.face_to_hybrid_element_number,
#     mesh.x0,mesh.widths,mesh.nelements,mesh.element_size,total_element_dofs,
#     dim,NHF,penalty,jac)
# apply_right_displacement_bc!(rhs,
#     x->test_displacement_field(x,displacement_amplitude),
#     surface_basis_vals,surface_quad,mesh.face_to_hybrid_element_number,
#     mesh.x0,mesh.widths,mesh.nelements,mesh.element_size,total_element_dofs,
#     dim,NHF,penalty,jac)
# apply_top_displacement_bc!(rhs,
#     x->test_displacement_field(x,displacement_amplitude),
#     surface_basis_vals,surface_quad,mesh.face_to_hybrid_element_number,
#     mesh.x0,mesh.widths,mesh.nelements,mesh.element_size,total_element_dofs,
#     dim,NHF,penalty,jac)
# apply_left_displacement_bc!(rhs,
#     x->test_displacement_field(x,displacement_amplitude),
#     surface_basis_vals,surface_quad,mesh.face_to_hybrid_element_number,
#     mesh.x0,mesh.widths,mesh.nelements,mesh.element_size,total_element_dofs,
#     dim,NHF,penalty,jac)
#
#
# total_dofs = get_total_mesh_dofs(total_element_dofs,
#     mesh.total_number_of_hybrid_elements,dim,sdim,NF,NHF)
# op = dropzeros!(sparse(matrix.rows,matrix.cols,matrix.vals,total_dofs,total_dofs))
# F = dropzeros!(sparsevec(rhs.rows,rhs.vals,total_dofs))
#
# sol = op\Array(F)
#
# stressdofs = get_stress_dofs(mesh.total_number_of_elements,dim,sdim,NF)
# dispdofs = get_displacement_dofs(mesh.total_number_of_elements,dim,sdim,NF)
# hybdofs = get_hybrid_dofs(mesh.total_number_of_elements,
#     mesh.total_number_of_hybrid_elements,dim,sdim,NF,NHF)
# stress = -Dhalf*reshape(sol[stressdofs],sdim,:)
# disp = reshape(sol[dispdofs],dim,:)
# hsol = reshape(sol[hybdofs],dim,:)
#
# node_coordinates = get_mesh_node_coordinates(mesh,basis)
# fig,ax = PyPlot.subplots()
# tcf = ax.tricontourf(node_coordinates[1,:],node_coordinates[2,:],
#     disp[2,:],20)
# fig.colorbar(tcf)
# fig
