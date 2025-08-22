using GSDFiles, Random

# ---------------------------
# Build a 2D hexagonal lattice in the xy-plane
# ---------------------------
function create_hexagonal_lattice(; n::Int=10, d::Float32=1.0f0)
    spacing = d
    sqrt_3 = sqrt(Float32(3))

    N = n * n
    positions = Array{Float32}(undef, N, 3)

    idx = 1
    @inbounds for i in 0:n-1
        y = i * (sqrt_3/2) * spacing
        for j in 0:n-1
            x = j * spacing
            if isodd(i)
                x += spacing/2
            end
            positions[idx, 1] = x
            positions[idx, 2] = y
            positions[idx, 3] = 0.0f0
            idx += 1
        end
    end

    # Box large enough to contain the lattice (no tilt)
    box_x = n * spacing
    box_y = (n-1) * (sqrt_3/2) * spacing + spacing/2
    box_z = 10.0f0  # non-zero z for OVITO
    box = (box_x, box_y, box_z, 0.0f0, 0.0f0, 0.0f0)

    return N, positions, box
end

# ---------------------------
# Write one HOOMD/OVITO-compatible frame
# ---------------------------
function create_hexagonal_gsd(path::AbstractString)
    # Create the writer with HOOMD schema (v1.4 schema is widely supported)
    w = GSDFiles.GSDWriter(path; application="HexLattice", schema="hoomd", schema_version=(1,4))
    h = GSDFiles.open_gsd(w)

    # Lattice
    N, positions, box = create_hexagonal_lattice(n=10, d=1.0f0)

    # System: single type "A", 0-based type ids
    types  = ["A"]
    typeid = fill(UInt32(0), N)     # all particles are type 0 ("A")

    # Small random velocities (Float32)
    Random.seed!(42)
    velocities = 0.01f0 .* rand(Float32, N, 3)

    # Optional: uniform diameters (OVITO radius = diameter/2)
    diameter = fill(1.0f0, N)

    # --- Write configuration chunks ---
    GSDFiles.write_configuration_step!(h, 0)
    GSDFiles.write_configuration_dimensions!(h, 3)
    GSDFiles.write_configuration_box!(h, box)

    # --- Write particles ---
    GSDFiles.write_particles_N!(h, N)
    GSDFiles.write_particles_types!(h, types)
    GSDFiles.write_particles_typeid!(h, typeid)
    GSDFiles.write_particles_diameter!(h, diameter)
    GSDFiles.write_particles_position!(h, positions)
    GSDFiles.write_particles_velocity!(h, velocities)

    # Finish frame and close
    GSDFiles.end_frame!(h)
    GSDFiles.close_gsd(h)

    println("Created hexagonal lattice file: $path")
    println("Number of particles: $N")
    println("File size: ", filesize(path), " bytes")
end

# Run it
create_hexagonal_gsd("hexagonal.gsd")