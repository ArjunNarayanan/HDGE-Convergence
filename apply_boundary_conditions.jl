function apply_left_displacement_bc!(matrix::HDGElasticity.SystemMatrix{T},
    vec_bc_operator::Vector{T},face_to_hybrid_element_number::Matrix{Int64},
    total_element_dofs::Int64,nelements::SVector{2},dim::Int64,NHF::Int64) where {T}

    for elid in 1:nelements[2]
        faceid = 4
        hid = face_to_hybrid_element_number[faceid,elid]
        hdofs = HDGElasticity.hybrid_dofs(hid,total_element_dofs,dim,NHF)
        rows,cols = HDGElasticity.element_dofs_to_operator_dofs(hdofs,hdofs)
        HDGElasticity.update!(matrix,rows,cols,vec_bc_operator)
    end
end

function apply_right_displacement_bc!(matrix::HDGElasticity.SystemMatrix{T},
    vec_bc_operator::Vector{T},face_to_hybrid_element_number::Matrix{Int64},
    total_element_dofs::Int64,nelements::SVector{2},dim::Int64,NHF::Int64) where {T}

    elstart = (nelements[1]-1)*nelements[2]+1
    elstop = nelements[1]*nelements[2]

    for elid in elstart:elstop
        faceid = 2
        hid = face_to_hybrid_element_number[faceid,elid]
        hdofs = HDGElasticity.hybrid_dofs(hid,total_element_dofs,dim,NHF)
        rows,cols = HDGElasticity.element_dofs_to_operator_dofs(hdofs,hdofs)
        HDGElasticity.update!(matrix,rows,cols,vec_bc_operator)
    end
end

function apply_right_displacement_bc!(rhs::HDGElasticity.SystemRHS{T},
    vec_bc_rhs::Vector{T},face_to_hybrid_element_number::Matrix{Int64},
    total_element_dofs::Int64,nelements::SVector{2},dim::Int64,NHF::Int64) where {T}

    elstart = (nelements[1]-1)*nelements[2]+1
    elstop = nelements[1]*nelements[2]

    for elid in elstart:elstop
        faceid = 2
        hid = face_to_hybrid_element_number[faceid,elid]
        hdofs = HDGElasticity.hybrid_dofs(hid,total_element_dofs,dim,NHF)
        HDGElasticity.update!(rhs,hdofs,vec_bc_rhs)
    end
end

function apply_bottom_displacement_bc!(matrix::HDGElasticity.SystemMatrix{T},
    vec_bc_operator::Vector{T},face_to_hybrid_element_number::Matrix{Int64},
    total_element_dofs::Int64,nelements::SVector{2},dim::Int64,NHF::Int64) where {T}

    elskip = nelements[2]
    elstop = nelements[1]*nelements[2]

    for elid in 1:elskip:elstop
        faceid = 1
        hid = face_to_hybrid_element_number[faceid,elid]
        hdofs = HDGElasticity.hybrid_dofs(hid,total_element_dofs,dim,NHF)
        rows,cols = HDGElasticity.element_dofs_to_operator_dofs(hdofs,hdofs)
        HDGElasticity.update!(matrix,rows,cols,vec_bc_operator)
    end
end

function apply_top_traction_bc!(matrix::HDGElasticity.SystemMatrix{T},
    vec_hybrid_local_operator_vals::Vector{Vector{T}},
    vec_hybrid_mass_operator_vals::Vector{T},
    face_to_hybrid_element_number::Matrix{Int64},total_element_dofs::Int64,
    nelements::SVector{2},dim::Int64,sdim::Int64,NF::Int64,NHF::Int64) where {T}

    elstart = nelements[2]
    elskip = nelements[2]
    elstop = nelements[1]*nelements[2]

    for elid in elstart:elskip:elstop
        faceid = 3
        hid = face_to_hybrid_element_number[faceid,elid]

        edofs = HDGElasticity.element_dofs(elid,dim,sdim,NF)
        hdofs = HDGElasticity.hybrid_dofs(hid,total_element_dofs,dim,NHF)

        rows,cols = HDGElasticity.element_dofs_to_operator_dofs(hdofs,edofs)
        HDGElasticity.update!(matrix,rows,cols,vec_hybrid_local_operator_vals[faceid])

        rows,cols = HDGElasticity.element_dofs_to_operator_dofs(hdofs,hdofs)
        HDGElasticity.update!(matrix,rows,cols,vec_hybrid_mass_operator_vals)
    end

end

function apply_bottom_traction_bc!(matrix::HDGElasticity.SystemMatrix{T},
    lhop::HDGElasticity.LocalHybridCoupling{T},face_to_hybrid_element_number,
    total_element_dofs,nelements,dim,sdim,NF,NHF) where {T}

    faceid = 1
    vec_hybrid_local = Array(vec(transpose(lhop.local_hybrid_operator[faceid])))
    vec_hybrid_mass = Array(vec(lhop.UhatUhat))

    elskip = nelements[2]
    elstop = nelements[1]*nelements[2]

    for elid in 1:elskip:elstop
        hid = face_to_hybrid_element_number[faceid,elid]

        edofs = HDGElasticity.element_dofs(elid,dim,sdim,NF)
        hdofs = HDGElasticity.hybrid_dofs(hid,total_element_dofs,dim,NHF)

        rows,cols = HDGElasticity.element_dofs_to_operator_dofs(hdofs,edofs)
        HDGElasticity.update!(matrix,rows,cols,vec_hybrid_local)

        rows,cols = HDGElasticity.element_dofs_to_operator_dofs(hdofs,hdofs)
        HDGElasticity.update!(matrix,rows,cols,vec_hybrid_mass)
    end
end

function apply_right_traction_bc!(matrix::HDGElasticity.SystemMatrix{T},
    lhop::HDGElasticity.LocalHybridCoupling{T},face_to_hybrid_element_number,
    total_element_dofs,nelements,dim,sdim,NF,NHF) where {T}

    faceid = 2
    vec_hybrid_local = Array(vec(transpose(lhop.local_hybrid_operator[faceid])))
    vec_hybrid_mass = Array(vec(lhop.UhatUhat))

    elstart = (nelements[1]-1)*nelements[2]+1
    elstop = nelements[1]*nelements[2]

    for elid in elstart:elstop
        hid = face_to_hybrid_element_number[faceid,elid]

        edofs = HDGElasticity.element_dofs(elid,dim,sdim,NF)
        hdofs = HDGElasticity.hybrid_dofs(hid,total_element_dofs,dim,NHF)

        rows,cols = HDGElasticity.element_dofs_to_operator_dofs(hdofs,edofs)
        HDGElasticity.update!(matrix,rows,cols,vec_hybrid_local)

        rows,cols = HDGElasticity.element_dofs_to_operator_dofs(hdofs,hdofs)
        HDGElasticity.update!(matrix,rows,cols,vec_hybrid_mass)
    end
end

function apply_left_traction_bc!(matrix::HDGElasticity.SystemMatrix{T},
    lhop::HDGElasticity.LocalHybridCoupling{T},face_to_hybrid_element_number,
    total_element_dofs,nelements,dim,sdim,NF,NHF) where {T}

    faceid = 4
    vec_hybrid_local = Array(vec(transpose(lhop.local_hybrid_operator[faceid])))
    vec_hybrid_mass = Array(vec(lhop.UhatUhat))

    for elid in 1:nelements[2]
        hid = face_to_hybrid_element_number[faceid,elid]

        edofs = HDGElasticity.element_dofs(elid,dim,sdim,NF)
        hdofs = HDGElasticity.hybrid_dofs(hid,total_element_dofs,dim,NHF)

        rows,cols = HDGElasticity.element_dofs_to_operator_dofs(hdofs,edofs)
        HDGElasticity.update!(matrix,rows,cols,vec_hybrid_local)

        rows,cols = HDGElasticity.element_dofs_to_operator_dofs(hdofs,hdofs)
        HDGElasticity.update!(matrix,rows,cols,vec_hybrid_mass)
    end
end
