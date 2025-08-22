module HOOMDWriter

using ..GSDFiles: GSDFilesHandle, write_chunk_raw!, rowmajor

export write_configuration_step!, write_configuration_dimensions!, write_configuration_box!,
       write_particles_N!, write_particles_types!, write_particles_typeid!,
       write_particles_position!, write_particles_velocity!,
       write_particles_diameter!, write_particles_diameter_by_type!,
       write_particles_type_shapes!, write_particles_orientation!,
       write_particles_inertia!, write_particles_angmom!

export write_bonds_N!, write_bonds_types!, write_bonds_typeid!, write_bonds_group!,
       write_angles_N!, write_angles_types!, write_angles_typeid!, write_angles_group!,
       write_dihedrals_N!, write_dihedrals_types!, write_dihedrals_typeid!, write_dihedrals_group!,
       write_impropers_N!, write_impropers_types!, write_impropers_typeid!, write_impropers_group!

# OVITO on-disk type codes (empirically: +1 vs upstream C enum)
const OVITO_UINT8    = UInt8(1)
const OVITO_UINT32   = UInt8(3)
const OVITO_UINT64   = UInt8(4)
const OVITO_INT8     = UInt8(5)
const OVITO_FLOAT32  = UInt8(9)
# (FLOAT64 would be 10, CHAR would be 11, if/when needed)

# ------------------------------
# configuration/*
# ------------------------------

"configuration/step : uint64 1×1 (force OVITO code)"
function write_configuration_step!(h::GSDFilesHandle, step::Integer)
    data = UInt64[UInt64(step)]
    write_chunk_raw!(h.user, "configuration/step";
                     type_code = OVITO_UINT64, N = 1, M = 1, data = data)
    return nothing
end

"configuration/dimensions : uint8 1×1 (2 or 3) (force OVITO code)"
function write_configuration_dimensions!(h::GSDFilesHandle, dims::Integer)
    d = UInt8(dims)
    @assert d == 2 || d == 3 "configuration/dimensions must be 2 or 3"
    write_chunk_raw!(h.user, "configuration/dimensions";
                     type_code = OVITO_UINT8, N = 1, M = 1, data = UInt8[d])
    return nothing
end

"configuration/box : float32 6×1  — [lx,ly,lz,xy,xz,yz] (force OVITO code)"
function write_configuration_box!(h::GSDFilesHandle, box)
    @assert length(box) == 6 "box must be a 6‑tuple/list [lx,ly,lz,xy,xz,yz]"
    b = Float32[Float32(box[1]), Float32(box[2]), Float32(box[3]),
                Float32(box[4]), Float32(box[5]), Float32(box[6])]
    write_chunk_raw!(h.user, "configuration/box";
                     type_code = OVITO_FLOAT32, N = 6, M = 1, data = b)
    return nothing
end

# ------------------------------
# particles/*
# ------------------------------

"particles/N : uint32 1×1 (force OVITO code)"
function write_particles_N!(h::GSDFilesHandle, N::Integer)
    @assert N ≥ 0 "particles/N must be ≥ 0"
    write_chunk_raw!(h.user, "particles/N";
                     type_code = OVITO_UINT32, N = 1, M = 1, data = UInt32[UInt32(N)])
    return nothing
end

# Build Int8 NT×M types table (row-major), each row is UTF-8 name + NUL + padding
function _make_types_table(typenames::Vector{String})
    NT = length(typenames)
    @assert NT ≥ 1 "At least one particle type required"
    maxlen = maximum(length(codeunits(s)) for s in typenames)
    M = maxlen + 1  # +1 for NUL
    A = fill(Int8(0), NT, M)
    for i in 1:NT
        b = Int8.(codeunits(typenames[i]))
        if !isempty(b)
            A[i, 1:length(b)] .= b
        end
        A[i, length(b)+1] = 0x00
    end
    return A, NT, M
end


"particles/diameter : float32 N×1 (per‑particle diameters)"
function write_particles_diameter!(h::GSDFilesHandle, diameter::AbstractVector{<:Real})
    N = length(diameter)
    data = Float32.(diameter)
    write_chunk_raw!(h.user, "particles/diameter";
                     type_code = OVITO_FLOAT32, N = N, M = 1, data = data)
    return nothing
end

"Convenience: set diameters from type IDs and a per‑type list (0‑based ids)"
function write_particles_diameter_by_type!(h::GSDFilesHandle,
                                           typeid::AbstractVector{<:Integer},
                                           per_type::AbstractVector{<:Real})
    N = length(typeid)
    @assert !isempty(h.typenames) "Call write_particles_types! (or set types) before diameters"
    @assert length(per_type) == length(h.typenames) "per_type length must match number of types"
    data = Vector{Float32}(undef, N)
    @inbounds for i in 1:N
        tid = typeid[i]
        @assert 0 <= tid < length(per_type) "typeid out of range"
        data[i] = Float32(per_type[tid+1])    # type ids are 0‑based; Julia vectors 1‑based
    end
    write_chunk_raw!(h.user, "particles/diameter";
                     type_code = OVITO_FLOAT32, N = N, M = 1, data = data)
    return nothing
end

"particles/types : int8 NT×M (row‑major UTF‑8 with trailing NUL) (force OVITO code)"
function write_particles_types!(h::GSDFilesHandle, typenames::Vector{String})
    if !isempty(h.typenames)
        @assert typenames == h.typenames "Type order changed across frames; keep it stable"
    else
        h.typenames = copy(typenames)
        empty!(h.type_index)
        @inbounds for (i,s) in enumerate(typenames)
            h.type_index[s] = UInt32(i-1)  # 0-based per HOOMD
        end
    end
    A, NT, M = _make_types_table(typenames)
    write_chunk_raw!(h.user, "particles/types";
                     type_code = OVITO_INT8, N = NT, M = M, data = rowmajor(A))
    return nothing
end

"particles/typeid : uint32 N×1 (force OVITO code)"
function write_particles_typeid!(h::GSDFilesHandle, ids::AbstractVector{<:Integer})
    N = length(ids)
    out = Vector{UInt32}(undef, N)
    @inbounds for i in 1:N
        v = ids[i]
        @assert v ≥ 0 "typeid must be ≥ 0"
        out[i] = UInt32(v)
    end
    if !isempty(h.typenames)
        NT = length(h.typenames)
        @inbounds for i in 1:N
            @assert out[i] < UInt32(NT) "typeid=$(out[i]) out of range for NT=$NT"
        end
    end
    write_chunk_raw!(h.user, "particles/typeid";
                     type_code = OVITO_UINT32, N = N, M = 1, data = out)
    return nothing
end

"particles/position : float32 N×3 (row‑major) (force OVITO code)"
function write_particles_position!(h::GSDFilesHandle, pos::AbstractMatrix{<:Real})
    N, M = size(pos); @assert M == 3 "particles/position must be N×3"
    A = Array{Float32}(undef, N, 3)
    @inbounds for i in 1:N, j in 1:3
        A[i,j] = Float32(pos[i,j])
    end
    write_chunk_raw!(h.user, "particles/position";
                     type_code = OVITO_FLOAT32, N = N, M = 3, data = rowmajor(A))
    return nothing
end

"particles/velocity : float32 N×3 (row‑major) (force OVITO code)"
function write_particles_velocity!(h::GSDFilesHandle, vel::AbstractMatrix{<:Real})
    N, M = size(vel); @assert M == 3 "particles/velocity must be N×3"
    A = Array{Float32}(undef, N, 3)
    @inbounds for i in 1:N, j in 1:3
        A[i,j] = Float32(vel[i,j])
    end
    write_chunk_raw!(h.user, "particles/velocity";
                     type_code = OVITO_FLOAT32, N = N, M = 3, data = rowmajor(A))
    return nothing
end


# In HOOMDWriter.jl (reusing your OVITO_* codes and write_chunk_raw!)
# 1) Per-type shapes (JSON strings), stored contiguously as bytes with \0 terminators
function write_particles_type_shapes!(h::GSDFilesHandle, shapes::Vector{String})
    @assert !isempty(h.typenames) "Call write_particles_types! first"
    @assert length(shapes) == length(h.typenames) "One shape JSON per type"
    # Build a single byte blob: "json\0json\0...\0"
    blob = UInt8[]
    for s in shapes
        append!(blob, codeunits(s)); push!(blob, 0x00)
    end
    push!(blob, 0x00)  # empty terminator (nice for readers)
    write_chunk_raw!(h.user, "particles/type_shapes";
                     type_code = OVITO_INT8,  # stored as bytes
                     N = length(blob), M = 1, data = blob)
    nothing
end

# 2) Per-particle orientations (unit quaternions, w,x,y,z)
function write_particles_orientation!(h::GSDFilesHandle, q::AbstractMatrix{<:Real})
    N, M = size(q); @assert M == 4 "orientation must be N×4 (w,x,y,z)"
    A = Array{Float32}(undef, N, 4)
    @inbounds for i in 1:N, j in 1:4
        A[i,j] = Float32(q[i,j])
    end
    # (Optional) normalize rows here if you like.
    write_chunk_raw!(h.user, "particles/orientation";
                     type_code = OVITO_FLOAT32, N = N, M = 4, data = rowmajor(A))
    nothing
end

# 3) Optional: inertia and angmom
function write_particles_inertia!(h::GSDFilesHandle, I::AbstractMatrix{<:Real})
    N, M = size(I); @assert M == 3
    write_chunk_raw!(h.user, "particles/inertia";
                     type_code = OVITO_FLOAT32, N = N, M = 3,
                     data = rowmajor(Float32.(I)))
end

function write_particles_angmom!(h::GSDFilesHandle, L::AbstractMatrix{<:Real})
    N, M = size(L); @assert M == 4
    write_chunk_raw!(h.user, "particles/angmom";
                     type_code = OVITO_FLOAT32, N = N, M = 4,
                     data = rowmajor(Float32.(L)))
end

# ========== BONDS ==========
"bonds/N : uint32 1×1"
function write_bonds_N!(h::GSDFilesHandle, Nb::Integer)
    @assert Nb ≥ 0
    write_chunk_raw!(h.user, "bonds/N"; type_code=OVITO_UINT32, N=1, M=1, data=UInt32[UInt32(Nb)])
    nothing
end

"bonds/types : int8 NT×M (UTF‑8 with NUL padding)"
function write_bonds_types!(h::GSDFilesHandle, typenames::Vector{String})
    A, NT, M = _make_types_table(typenames)
    write_chunk_raw!(h.user, "bonds/types"; type_code=OVITO_INT8, N=NT, M=M, data=rowmajor(A))
    nothing
end

"bonds/typeid : uint32 Nb×1 (0‑based type ids)"
function write_bonds_typeid!(h::GSDFilesHandle, typeid::AbstractVector{<:Integer})
    Nb = length(typeid)
    write_chunk_raw!(h.user, "bonds/typeid"; type_code=OVITO_UINT32, N=Nb, M=1, data=UInt32.(typeid))
    nothing
end

"bonds/group : uint32 Nb×2 (indices of the two bonded particles)"
function write_bonds_group!(h::GSDFilesHandle, group::AbstractMatrix{<:Integer})
    Nb, m = size(group); @assert m == 2 "bonds/group must be Nb×2"
    write_chunk_raw!(h.user, "bonds/group"; type_code=OVITO_UINT32, N=Nb, M=2, data=rowmajor(UInt32.(group)))
    nothing
end

# ========== ANGLES ==========
"angles/N : uint32 1×1"
function write_angles_N!(h::GSDFilesHandle, Na::Integer)
    @assert Na ≥ 0
    write_chunk_raw!(h.user, "angles/N"; type_code=OVITO_UINT32, N=1, M=1, data=UInt32[UInt32(Na)])
    nothing
end

"angles/types : int8 NT×M"
function write_angles_types!(h::GSDFilesHandle, typenames::Vector{String})
    A, NT, M = _make_types_table(typenames)
    write_chunk_raw!(h.user, "angles/types"; type_code=OVITO_INT8, N=NT, M=M, data=rowmajor(A))
    nothing
end

"angles/typeid : uint32 Na×1"
function write_angles_typeid!(h::GSDFilesHandle, typeid::AbstractVector{<:Integer})
    Na = length(typeid)
    write_chunk_raw!(h.user, "angles/typeid"; type_code=OVITO_UINT32, N=Na, M=1, data=UInt32.(typeid))
    nothing
end

"angles/group : uint32 Na×3 (triplets)"
function write_angles_group!(h::GSDFilesHandle, group::AbstractMatrix{<:Integer})
    Na, m = size(group); @assert m == 3 "angles/group must be Na×3"
    write_chunk_raw!(h.user, "angles/group"; type_code=OVITO_UINT32, N=Na, M=3, data=rowmajor(UInt32.(group)))
    nothing
end

# ========== DIHEDRALS ==========
"dihedrals/N : uint32 1×1"
function write_dihedrals_N!(h::GSDFilesHandle, Nd::Integer)
    @assert Nd ≥ 0
    write_chunk_raw!(h.user, "dihedrals/N"; type_code=OVITO_UINT32, N=1, M=1, data=UInt32[UInt32(Nd)])
    nothing
end

"dihedrals/types : int8 NT×M"
function write_dihedrals_types!(h::GSDFilesHandle, typenames::Vector{String})
    A, NT, M = _make_types_table(typenames)
    write_chunk_raw!(h.user, "dihedrals/types"; type_code=OVITO_INT8, N=NT, M=M, data=rowmajor(A))
    nothing
end

"dihedrals/typeid : uint32 Nd×1"
function write_dihedrals_typeid!(h::GSDFilesHandle, typeid::AbstractVector{<:Integer})
    Nd = length(typeid)
    write_chunk_raw!(h.user, "dihedrals/typeid"; type_code=OVITO_UINT32, N=Nd, M=1, data=UInt32.(typeid))
    nothing
end

"dihedrals/group : uint32 Nd×4 (quadruplets)"
function write_dihedrals_group!(h::GSDFilesHandle, group::AbstractMatrix{<:Integer})
    Nd, m = size(group); @assert m == 4 "dihedrals/group must be Nd×4"
    write_chunk_raw!(h.user, "dihedrals/group"; type_code=OVITO_UINT32, N=Nd, M=4, data=rowmajor(UInt32.(group)))
    nothing
end

# ========== IMPROPERS ==========
"impropers/N : uint32 1×1"
function write_impropers_N!(h::GSDFilesHandle, Ni::Integer)
    @assert Ni ≥ 0
    write_chunk_raw!(h.user, "impropers/N"; type_code=OVITO_UINT32, N=1, M=1, data=UInt32[UInt32(Ni)])
    nothing
end

"impropers/types : int8 NT×M"
function write_impropers_types!(h::GSDFilesHandle, typenames::Vector{String})
    A, NT, M = _make_types_table(typenames)
    write_chunk_raw!(h.user, "impropers/types"; type_code=OVITO_INT8, N=NT, M=M, data=rowmajor(A))
    nothing
end

"impropers/typeid : uint32 Ni×1"
function write_impropers_typeid!(h::GSDFilesHandle, typeid::AbstractVector{<:Integer})
    Ni = length(typeid)
    write_chunk_raw!(h.user, "impropers/typeid"; type_code=OVITO_UINT32, N=Ni, M=1, data=UInt32.(typeid))
    nothing
end

"impropers/group : uint32 Ni×4"
function write_impropers_group!(h::GSDFilesHandle, group::AbstractMatrix{<:Integer})
    Ni, m = size(group); @assert m == 4 "impropers/group must be Ni×4"
    write_chunk_raw!(h.user, "impropers/group"; type_code=OVITO_UINT32, N=Ni, M=4, data=rowmajor(UInt32.(group)))
    nothing
end

end # module HOOMDWriter