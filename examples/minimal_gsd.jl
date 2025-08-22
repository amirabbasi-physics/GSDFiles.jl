using GSDFiles, LinearAlgebra

# ---- helper: build nearest A–B bonds (no PBC needed for this layout) ----
function nearest_AB_bonds(pos::AbstractMatrix{<:Real}, typeid::AbstractVector{<:Integer})
    @assert size(pos,2) == 3 "pos must be N×3"
    N = size(pos,1)
    A_idx = findall(==(0), typeid)  # type id 0 -> "A"
    B_idx = findall(==(1), typeid)  # type id 1 -> "B"
    @assert !isempty(A_idx) && !isempty(B_idx) "Need at least one A and one B"

    pairs = Vector{Tuple{Int,Int}}()
    for ai in A_idx
        # find nearest B to this A
        pA = @view pos[ai, :]
        best_d2 = typemax(Float64)
        best_b  = B_idx[1]
        for bi in B_idx
            pB = @view pos[bi, :]
            d2 = (pA[1]-pB[1])^2 + (pA[2]-pB[2])^2 + (pA[3]-pB[3])^2
            if d2 < best_d2
                best_d2 = d2
                best_b  = bi
            end
        end
        push!(pairs, (ai-1, best_b-1))  # 0-based for HOOMD/OVITO
    end

    # Build Nb×2 group (UInt32), and Nb type ids (all 0 = "A-B")
    Nb = length(pairs)
    bond_group  = Array{UInt32}(undef, Nb, 2)
    @inbounds for i in 1:Nb
        bond_group[i,1] = UInt32(pairs[i][1])
        bond_group[i,2] = UInt32(pairs[i][2])
    end
    bond_typeid = fill(UInt32(0), Nb)  # all bonds are type 0 -> "A-B"
    return bond_group, bond_typeid
end

# ---- main example ----
w = GSDFiles.GSDWriter("ovito_compatible.gsd";
                  application="NonEqSim",
                  schema="hoomd",
                  schema_version=(1,4))

h = GSDFiles.open_gsd(w)

N = 8
types  = ["A","B"]
typeid = UInt32[1,1,1,1, 0,0,0,0]   # first 4 B (1), last 4 A (0)
box    = (4.0f0,4.0f0,4.0f0, 0.0f0,0.0f0,0.0f0)

pos = Float32[
    0 0 0; 1.5 0 0; 0 1.5 0; 1.5 1.5 0;
    0 0 1.5; 1.5 0 1.5; 0 1.5 1.5; 1.5 1.5 1.5
]
vel = Float32[
     0.1  0.0  0.0; -0.1 0.0  0.0;
     0.0  0.1  0.0;  0.0 -0.1  0.0;
     0.0  0.0  0.1;  0.1  0.0  0.0;
    -0.1  0.0  0.0;  0.0  0.1  0.0
]

# uniform diameters (radius=0.5 in OVITO)
per_type_diam = [1.0f0, 1.0f0]

# --- write configuration ---
GSDFiles.write_configuration_step!(h, 0)
GSDFiles.write_configuration_dimensions!(h, 3)
GSDFiles.write_configuration_box!(h, box)

# --- write particles ---
GSDFiles.write_particles_N!(h, N)
GSDFiles.write_particles_types!(h, types)
GSDFiles.write_particles_typeid!(h, typeid)
GSDFiles.write_particles_diameter_by_type!(h, typeid, per_type_diam)
GSDFiles.write_particles_position!(h, pos)
GSDFiles.write_particles_velocity!(h, vel)

# --- build and write nearest A–B bonds ---
bond_group, bond_typeid = nearest_AB_bonds(pos, Int.(typeid))  # Int for findall(==)
GSDFiles.write_bonds_types!(h, ["A-B"])        # only one bond type in this example
GSDFiles.write_bonds_N!(h, size(bond_group,1))
GSDFiles.write_bonds_typeid!(h, bond_typeid)   # 0-based type ids
GSDFiles.write_bonds_group!(h, bond_group)     # Nb×2, 0-based particle indices

GSDFiles.end_frame!(h)
GSDFiles.close_gsd(h)