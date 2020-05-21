function linear_displacement(x::Vector{T};alpha=1e-2) where {T}
    return alpha*x
end

function quadratic_displacement(x;alpha=1e-2)
    return alpha*(x.^2)
end

function shear_displacement(x;alpha=1e-2)
    return [0.0,alpha*x[1]]
end

function quadratic_body_force(lambda,mu;alpha=1e-2)
    return -2*alpha*(lambda+2mu)*[1.0,1.0]
end

function test_displacement_field(x,alpha)
    u1 = x[2]*sin(pi*x[1])
    u2 = x[1]^3 + cos(pi*x[2])
    return alpha*[u1,u2]
    # return linear_displacement(x)
    # return quadratic_displacement(x)
    # return shear_displacement(x)
end

function test_stress_field(x,lambda,mu,alpha)

    s1 = (lambda+2mu)*alpha*pi*x[2]*cos(pi*x[1]) -
        lambda*alpha*pi*sin(pi*x[2])
    s2 = -(lambda+2mu)*alpha*pi*sin(pi*x[2]) +
        lambda*alpha*pi*x[2]*cos(pi*x[1])
    s3 = alpha*mu*(3x[1]^2+sin(pi*x[1]))

    return [s1,s2,s3]

end

function test_stress_divergence(x,lambda,mu,alpha)

    s1 = -(lambda+2mu)*pi^2*x[2]*sin(pi*x[1])
    s2 = mu*6x[1] + (lambda+mu)*pi*cos(pi*x[1]) -
         (lambda+2mu)*pi^2*cos(pi*x[2])
    return alpha*[s1,s2]
end

function test_body_force(x,lambda,mu,alpha)

    return -1.0*test_stress_divergence(x,lambda,mu,alpha)
    # return quadratic_body_force(lambda,mu)
    # return [0.0,0.0]
end

function apply_bottom_displacement_bc!(matrix::HDGElasticity.SystemMatrix,
    vec_bc_operator,face_to_hybrid_element_number,total_element_dofs,nelements,
    dim,NHF)

    elskip = nelements[2]
    elstop = nelements[1]*nelements[2]

    faceid = 1
    for elid in 1:elskip:elstop
        hid = face_to_hybrid_element_number[faceid,elid]
        hdofs = HDGElasticity.hybrid_dofs(hid,total_element_dofs,dim,NHF)
        rows,cols = HDGElasticity.element_dofs_to_operator_dofs(hdofs,hdofs)
        HDGElasticity.update!(matrix,rows,cols,vec_bc_operator)
    end

end

function apply_right_displacement_bc!(matrix::HDGElasticity.SystemMatrix,
    vec_bc_operator,face_to_hybrid_element_number,total_element_dofs,nelements,
    dim,NHF)

    elstart = (nelements[1]-1)*nelements[2]+1
    elstop = nelements[1]*nelements[2]

    faceid = 2
    for elid in elstart:elstop
        hid = face_to_hybrid_element_number[faceid,elid]
        hdofs = HDGElasticity.hybrid_dofs(hid,total_element_dofs,dim,NHF)
        rows,cols = HDGElasticity.element_dofs_to_operator_dofs(hdofs,hdofs)
        HDGElasticity.update!(matrix,rows,cols,vec_bc_operator)
    end

end

function apply_top_displacement_bc!(matrix::HDGElasticity.SystemMatrix,
    vec_bc_operator,face_to_hybrid_element_number,total_element_dofs,nelements,
    dim,NHF)

    elstart = nelements[2]
    elskip = nelements[2]
    elstop = nelements[1]*nelements[2]

    faceid = 3
    for elid in elstart:elskip:elstop
        hid = face_to_hybrid_element_number[faceid,elid]
        hdofs = HDGElasticity.hybrid_dofs(hid,total_element_dofs,dim,NHF)
        rows,cols = HDGElasticity.element_dofs_to_operator_dofs(hdofs,hdofs)
        HDGElasticity.update!(matrix,rows,cols,vec_bc_operator)
    end

end

function apply_left_displacement_bc!(matrix::HDGElasticity.SystemMatrix,
    vec_bc_operator,face_to_hybrid_element_number,total_element_dofs,nelements,
    dim,NHF)

    faceid = 4
    for elid in 1:nelements[2]
        hid = face_to_hybrid_element_number[faceid,elid]
        hdofs = HDGElasticity.hybrid_dofs(hid,total_element_dofs,dim,NHF)
        rows,cols = HDGElasticity.element_dofs_to_operator_dofs(hdofs,hdofs)
        HDGElasticity.update!(matrix,rows,cols,vec_bc_operator)
    end

end


function bc_rhs(surface_basis_vals,surface_quad,prescribed_displacement,
    element_bounds,dim,NHF,penalty,jac)

    rhs = zeros(dim*NHF)

    for (idx,(pvec,w)) in enumerate(surface_quad)
        p = pvec[1]
        vals = surface_basis_vals[idx]
        x = affine_map(p,element_bounds[1],element_bounds[2])
        uD = prescribed_displacement(x)
        r = vcat([v*uD for v in vals]...)
        rhs += penalty*r*jac*w
    end
    return rhs
end

function body_force_rhs(basis_vals,quad,prescribed_body_force,xL,xR,
    dim,NF,jac)

    rhs = zeros(dim*NF)

    for (idx,(pt,wt)) in enumerate(quad)
        vals = basis_vals[idx]
        x = affine_map(pt,xL,xR)
        force = prescribed_body_force(x)
        r = vcat([v*force for v in vals]...)
        rhs += r*jac*wt
    end
    return rhs
end

function apply_body_force!(rhs::HDGElasticity.SystemRHS,
    prescribed_body_force::Function,basis_vals,quad,x0,nelements,
    element_size,dim,NF,jac::HDGElasticity.AffineMapJacobian)

    elid = 1
    sdim = HDGElasticity.symmetric_tensor_dim(dim)

    for elx in 1:nelements[1]
        for ely in 1:nelements[2]
            xL = element_bottom_left(elx,ely,x0,element_size)
            xR = element_bottom_left(elx+1,ely+1,x0,element_size)
            edofs = HDGElasticity.element_displacement_dofs(elid,dim,sdim,NF)

            rhs_vals = body_force_rhs(basis_vals,quad,prescribed_body_force,
                xL,xR,dim,NF,jac.detjac)

            append!(rhs.rows,edofs)
            append!(rhs.vals,rhs_vals)

            elid += 1
        end
    end

end

function apply_bottom_displacement_bc!(rhs::HDGElasticity.SystemRHS,
    boundary_displacement::Function,surface_basis_vals,surface_quad,
    face_to_hybrid_element_number,x0,widths,nelements,element_sizes,
    total_element_dofs,dim,NHF,penalty,jac)

    elskip = nelements[2]
    elstop = nelements[2]*nelements[1]
    ebounds = [x0[1],element_sizes[1]]

    faceid = 1
    for elid in 1:elskip:elstop
        hid = face_to_hybrid_element_number[faceid,elid]
        hdofs = HDGElasticity.hybrid_dofs(hid,total_element_dofs,dim,NHF)
        rhs_vals = bc_rhs(surface_basis_vals,surface_quad,
            x->boundary_displacement([x,x0[2]]),ebounds,dim,NHF,
            penalty,jac.jac[1])
        append!(rhs.rows,hdofs)
        append!(rhs.vals,rhs_vals)
        ebounds .+= element_sizes[1]
    end

end

function apply_right_displacement_bc!(rhs::HDGElasticity.SystemRHS,
    boundary_displacement::Function,surface_basis_vals,surface_quad,
    face_to_hybrid_element_number,x0,widths,nelements,element_sizes,
    total_element_dofs,dim,NHF,penalty,jac)

    elstart = (nelements[1]-1)*nelements[2]+1
    elstop = nelements[1]*nelements[2]
    ebounds = [x0[2],element_sizes[2]]
    right_bdry = x0[1]+widths[1]

    faceid = 2
    for elid in elstart:elstop
        hid = face_to_hybrid_element_number[faceid,elid]
        hdofs = HDGElasticity.hybrid_dofs(hid,total_element_dofs,dim,NHF)
        rhs_vals = bc_rhs(surface_basis_vals,surface_quad,
            x->boundary_displacement([right_bdry,x]),ebounds,dim,NHF,
            penalty,jac.jac[2])
        append!(rhs.rows,hdofs)
        append!(rhs.vals,rhs_vals)
        ebounds .+= element_sizes[2]
    end

end

function apply_top_displacement_bc!(rhs::HDGElasticity.SystemRHS,
    boundary_displacement::Function,surface_basis_vals,surface_quad,
    face_to_hybrid_element_number,x0,widths,nelements,element_sizes,
    total_element_dofs,dim,NHF,penalty,jac)

    elstart = nelements[2]
    elskip = nelements[2]
    elstop = nelements[1]*nelements[2]

    faceid = 3
    ebounds = [x0[1],element_sizes[1]]
    top_bdry = x0[2]+widths[2]

    for elid in elstart:elskip:elstop
        hid = face_to_hybrid_element_number[faceid,elid]
        hdofs = HDGElasticity.hybrid_dofs(hid,total_element_dofs,dim,NHF)
        rhs_vals = bc_rhs(surface_basis_vals,surface_quad,
            x->boundary_displacement([x,top_bdry]),ebounds,dim,NHF,
            penalty,jac.jac[1])
        append!(rhs.rows,hdofs)
        append!(rhs.vals,rhs_vals)
        ebounds .+= element_sizes[1]
    end

end

function apply_left_displacement_bc!(rhs::HDGElasticity.SystemRHS,
    boundary_displacement::Function,surface_basis_vals,surface_quad,
    face_to_hybrid_element_number,x0,widths,nelements,element_sizes,
    total_element_dofs,dim,NHF,penalty,jac)

    faceid = 4
    ebounds = [x0[2],element_sizes[2]]

    for elid in 1:nelements[2]
        hid = face_to_hybrid_element_number[faceid,elid]
        hdofs = HDGElasticity.hybrid_dofs(hid,total_element_dofs,dim,NHF)
        rhs_vals = bc_rhs(surface_basis_vals,surface_quad,
            x->boundary_displacement([x0[1],x]),ebounds,dim,NHF,
            penalty,jac.jac[2])
        append!(rhs.rows,hdofs)
        append!(rhs.vals,rhs_vals)
        ebounds .+= element_sizes[2]
    end

end
