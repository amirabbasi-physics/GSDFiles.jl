using Test
using GSDFiles

# -------------------------------
# Helpers to inspect a .gsd file
# -------------------------------

const GSD_MAGIC = 0x65DF65DF65DF65DF
const HDR_BYTES = 256
const IDX_ENT_BYTES = 32

# GSD type codes (OVITO/GSD v2)
const GSD_TYPE_UINT8  = UInt8(0x01)
const GSD_TYPE_UINT32 = UInt8(0x03)
const GSD_TYPE_UINT64 = UInt8(0x04)
const GSD_TYPE_INT8   = UInt8(0x05)
const GSD_TYPE_FLOAT  = UInt8(0x09)  # Float32
const GSD_TYPE_DOUBLE = UInt8(0x0A)  # Float64

# Map code -> element size (bytes)
const GSD_TYPE_SIZE = Dict(
    GSD_TYPE_UINT8  => 1,
    GSD_TYPE_INT8   => 1,
    GSD_TYPE_UINT32 => 4,
    GSD_TYPE_UINT64 => 8,
    GSD_TYPE_FLOAT  => 4,
    GSD_TYPE_DOUBLE => 8,
)

# Read header as a NamedTuple (without depending on internal structs)
function read_header(io)::NamedTuple
    seek(io, 0)
    magic                    = read(io, UInt64)
    index_location           = read(io, UInt64)
    index_allocated_entries  = read(io, UInt64)
    namelist_location        = read(io, UInt64)
    namelist_alloc_entries   = read(io, UInt64)
    schema_version           = read(io, UInt32)
    gsd_version              = read(io, UInt32)
    application              = read!(io, Vector{UInt8}(undef, 64))
    schema                   = read!(io, Vector{UInt8}(undef, 64))
    reserved                 = read!(io, Vector{UInt8}(undef, 80))
    return (;
        magic,
        index_location,
        index_allocated_entries,
        namelist_location,
        namelist_alloc_entries,
        schema_version,
        gsd_version,
        application,
        schema,
        reserved
    )
end

# Parse 0-separated namelist blob (v2.x). Returns Vector{String} in id order.
function read_names(io, hdr)::Vector{String}
    seek(io, Int(hdr.namelist_location))
    blob = read!(io, Vector{UInt8}(undef, Int(hdr.namelist_alloc_entries) * 64))
    names = String[]
    start = 1
    for i in 1:length(blob)
        if blob[i] == 0x00
            if i >= start
                push!(names, String(copy(blob[start:i-1])))
            else
                push!(names, "")
            end
            start = i+1
        end
    end
    return names
end

# Read all index entries as NamedTuples
# Helper: read index entries but stop at the sentinel (location==0)
function read_index(io, hdr)
    seek(io, Int(hdr.index_location))
    entries = NamedTuple[]
    for _ in 1:UInt(hdr.index_allocated_entries)
        frame     = read(io, UInt64)
        N         = read(io, UInt64)
        location  = read(io, Int64)
        M         = read(io, UInt32)
        id        = read(io, UInt16)
        typ       = read(io, UInt8)
        flags     = read(io, UInt8)
        if location == 0
            break                       # <-- sentinel: end-of-list
        end
        push!(entries, (frame=frame, N=N, location=location, M=M, id=id, typ=typ, flags=flags))
    end
    return entries
end

# Fetch all index entries for a given name (any frame)
function find_entries_by_name(entries, names, target_name::String)
    # ids are 0-based; names vector is 1-based
    ids = findall(x -> x == target_name, names)
    isempty(ids) && return NamedTuple[]
    id0 = UInt16(first(ids) - 1)
    return filter(e -> e.id == id0, entries)
end

# Read a chunk's raw vector given an index entry
function read_chunk_vector(io, e)::AbstractVector
    T = if e.typ == GSD_TYPE_INT8
        Int8
    elseif e.typ == GSD_TYPE_UINT32
        UInt32
    elseif e.typ == GSD_TYPE_FLOAT
        Float32
    elseif e.typ == GSD_TYPE_DOUBLE
        Float64
    else
        error("Unsupported gsd type code: $(e.typ)")
    end
    count = Int(e.N) * Int(e.M)
    seek(io, Int(e.location))
    buf = Vector{T}(undef, count)
    read!(io, buf)
    return buf
end

# Read exactly one index entry at byte offset `off`
read_one_index(io, off) = begin
    seek(io, off)
    frame    = read(io, UInt64)
    N        = read(io, UInt64)
    location = read(io, Int64)
    M        = read(io, UInt32)
    id       = read(io, UInt16)
    typ      = read(io, UInt8)
    flags    = read(io, UInt8)
    (; frame, N, location, M, id, typ, flags)
end

# Find the sentinel by scanning entries starting at index_location
function read_sentinel(io, hdr)
    seek(io, Int(hdr.index_location))
    off = Int(hdr.index_location)
    for _ in 1:UInt(hdr.index_allocated_entries)
        e = read_one_index(io, off)
        if e.location == 0
            return e
        end
        off += 32  # each index entry is 32 bytes
    end
    error("No sentinel found in index (location==0).")
end

# Rebuild an N×M Matrix in Julia from a row-major flat vector
function unflatten_rowmajor(vec::AbstractVector, N::Int, M::Int)
    A = Array{eltype(vec)}(undef, N, M)
    k = 1
    @inbounds for i in 1:N, j in 1:M
        A[i,j] = vec[k]
        k += 1
    end
    return A
end

# Decode particles/types table (Int8 NT×M), returning Vector{String}
function decode_types_table(vec::Vector{Int8}, NT::Int, M::Int)::Vector{String}
    A = unflatten_rowmajor(vec, NT, M)
    types = String[]
    for i in 1:NT
        # find NUL (0x00) or use full row
        row = reinterpret(UInt8, A[i, :])
        nulpos = findfirst(==(0x00), row)
        if nulpos === nothing
            push!(types, String(copy(row)))
        else
            push!(types, String(copy(row[1:nulpos-1])))
        end
    end
    return types
end

# ------------------------------------
# Smoke: little endian platform (FYI)
# ------------------------------------
# Simple, portable little‑endian detection (no stdlib deps)
is_little_endian() = reinterpret(UInt8, UInt16[0x0102])[1] == 0x02

@testset "Platform assumptions" begin
    @test is_little_endian()   # x86_64/aarch64 Linux, macOS => true
end

# ------------------------------------
# Core tests
# ------------------------------------
@testset "GSD v2 + HOOMD writer end-to-end" begin
    tmp = mktempdir()
    path = joinpath(tmp, "ovito_compatible.gsd")

    # ----- build one frame
    w = GSDFiles.GSDWriter(path; application="NonEqSim", schema="hoomd", schema_version=(1,4))
    h = GSDFiles.open_gsd(w)

    N = 8
    types  = ["A","B"]
    typeid = UInt32[1,1,1,1, 0,0,0,0] # bottom B (1), top A (0)
    box    = (4.0f0,4.0f0,4.0f0, 0,0,0)

    pos = Float32[
        0 0 0; 2 0 0; 0 2 0; 2 2 0;
        0 0 2; 2 0 2; 0 2 2; 2 2 2
    ]
    vel = Float32[
         0.1  0.0  0.0; -0.1 0.0  0.0;
         0.0  0.1  0.0;  0.0 -0.1  0.0;
         0.0  0.0  0.1;  0.1  0.0  0.0;
        -0.1  0.0  0.0;  0.0  0.1  0.0
    ]

    GSDFiles.write_configuration_box!(h, box)
    GSDFiles.write_particles_N!(h, N)
    GSDFiles.write_particles_types!(h, types)
    GSDFiles.write_particles_typeid!(h, typeid)
    GSDFiles.write_particles_position!(h, pos)
    GSDFiles.write_particles_velocity!(h, vel)

    GSDFiles.end_frame!(h)
    GSDFiles.close_gsd(h)

    # ----- inspect raw file
    io = open(path, "r")
    hdr = read_header(io)

    @test hdr.magic == GSD_MAGIC
    # schema_version 1.4 => (1<<16) | 4
    @test hdr.schema_version == UInt32((1 << 16) | 4)
    # gsd_version 2.x => (2<<16) | 0
    @test hdr.gsd_version == UInt32((2 << 16) | 0)

    # Name list and index pointers must be non-zero, correctly sized
    @test hdr.namelist_location > 0
    @test hdr.index_location > 0
    @test hdr.index_allocated_entries > 0
    @test hdr.namelist_alloc_entries > 0

    names   = read_names(io, hdr)
    entries = read_index(io, hdr)

    # Expected chunk names present
    expected_names = Set(["configuration/box","particles/N","particles/types",
                          "particles/typeid","particles/position","particles/velocity"])
    @test expected_names ⊆ Set(names)

    # Verify index entries have valid type codes and consistent sizes
    for e in entries
        @test haskey(GSD_TYPE_SIZE, e.typ)
        @test e.N ≥ 1
        @test e.M ≥ 1
        # data region must be within file
        @test e.location ≥ HDR_BYTES
    end

    # Helper to get single entry by name (frame 0)
    function one_entry(name)
        ents = find_entries_by_name(entries, names, name)
        @test !isempty(ents)
        for e in ents; @test e.frame == 0; end
        @test length(ents) == 1
        return first(ents)
    end

    # configuration/box
    ebox = one_entry("configuration/box")
    @test ebox.typ == GSD_TYPE_FLOAT
    @test ebox.N == 6 && ebox.M == 1
    box_vec = read_chunk_vector(io, ebox)
    @test box_vec == Float32[box...]

    # particles/N
    eN = one_entry("particles/N")
    @test eN.typ == GSD_TYPE_UINT32
    @test eN.N == 1 && eN.M == 1
    N_vec = read_chunk_vector(io, eN)
    @test length(N_vec) == 1 && N_vec[1] == UInt32(N)

    # particles/types (NT×M Int8 with NUL)
    etypes = one_entry("particles/types")
    @test etypes.typ == GSD_TYPE_INT8
    types_vec = read_chunk_vector(io, etypes)
    NT = Int(etypes.N)
    M  = Int(etypes.M)
    @test NT == length(types)
    decoded = decode_types_table(types_vec, NT, M)
    @test decoded == types
    # M must be >= longest name + 1 (for NUL)
    maxlen = maximum(length(codeunits(t)) for t in types)
    @test M ≥ maxlen + 1

    # typeid
    etid = one_entry("particles/typeid")
    @test etid.typ == GSD_TYPE_UINT32
    @test etid.N == UInt64(N) && etid.M == 1
    tid_vec = read_chunk_vector(io, etid)
    @test tid_vec == typeid

    # position (row-major N×3)
    epos = one_entry("particles/position")
    @test epos.typ == GSD_TYPE_FLOAT
    @test epos.N == UInt64(N) && epos.M == 3
    pos_vec = read_chunk_vector(io, epos)
    pos_mat = unflatten_rowmajor(pos_vec, N, 3)
    @test pos_mat == pos

    # velocity (row-major N×3)
    evel = one_entry("particles/velocity")
    @test evel.typ == GSD_TYPE_FLOAT
    @test evel.N == UInt64(N) && evel.M == 3
    vel_vec = read_chunk_vector(io, evel)
    vel_mat = unflatten_rowmajor(vel_vec, N, 3)
    @test vel_mat == vel

    @test hdr.index_allocated_entries ≥ 1  # sentinel included
    @test hdr.namelist_alloc_entries ≥ 2   # at least final empty name is 1 byte (actually ≥2 total)
    seek(io, Int(hdr.index_location))
    # Read last 32 bytes of index block and assert location==0:
    sentinel = read_sentinel(io, hdr)
    @test sentinel.location == 0
    
    seek(io, Int(hdr.namelist_location) + Int(hdr.namelist_alloc_entries) * 64 - 1)
    @test read(io, UInt8) == 0x00  # last byte is 0 (empty final name)
    close(io)

    close(io)
end

# ------------------------------------
# Multi-frame + sorting of index
# ------------------------------------
@testset "Multiple frames, stable types, index sorted" begin
    tmp = mktempdir()
    path = joinpath(tmp, "two_frames.gsd")

    w = GSDFiles.GSDWriter(path; application="NonEqSim", schema="hoomd", schema_version=(1,4))
    h = GSDFiles.open_gsd(w)

    N = 4
    types = ["A","B"]
    typeid = UInt32[0,0,1,1]
    box = (3.0f0,3.0f0,3.0f0, 0,0,0)

    pos1 = Float32[0 0 0; 1 0 0; 0 1 0; 1 1 0]
    pos2 = Float32[0 0 1; 1 0 1; 0 1 1; 1 1 1]

    GSDFiles.write_configuration_box!(h, box)
    GSDFiles.write_particles_N!(h, N)
    GSDFiles.write_particles_types!(h, types)
    GSDFiles.write_particles_typeid!(h, typeid)
    GSDFiles.write_particles_position!(h, pos1)
    GSDFiles.end_frame!(h)

    # Frame 1 (same types order)
    GSDFiles.write_configuration_box!(h, box)
    GSDFiles.write_particles_N!(h, N)
    GSDFiles.write_particles_types!(h, types)  # same order enforced
    GSDFiles.write_particles_typeid!(h, typeid)
    GSDFiles.write_particles_position!(h, pos2)
    GSDFiles.end_frame!(h)

    GSDFiles.close_gsd(h)

    io = open(path, "r")
    hdr = read_header(io)
    names = read_names(io, hdr)
    entries = read_index(io, hdr)

    # index must be sorted by (frame, id)
    @test issorted(entries, by = e -> (e.frame, e.id))

    # Validate positions in both frames
    function read_pos_for_frame(frame)
        ents = find_entries_by_name(entries, names, "particles/position")
        e = only(filter(e->e.frame==frame, ents))
        vec = read_chunk_vector(io, e)
        return unflatten_rowmajor(vec, Int(e.N), Int(e.M))
    end
    @test read_pos_for_frame(0) == pos1
    @test read_pos_for_frame(1) == pos2

    close(io)
end

# ------------------------------------
# Error conditions: shape/type/order checks
# ------------------------------------
@testset "Writer guards throw on misuse" begin
    tmp = mktempdir()
    path = joinpath(tmp, "errors.gsd")

    w = GSDFiles.GSDWriter(path; application="NonEqSim", schema="hoomd", schema_version=(1,4))
    h = GSDFiles.open_gsd(w)

    # Set types once
    GSDFiles.write_particles_types!(h, ["A","B"])

    # 1) Changing type order across frames should throw
    GSDFiles.end_frame!(h)
    @test_throws AssertionError GSDFiles.write_particles_types!(h, ["B","A"])

    # 2) Out-of-range typeid
    @test_throws AssertionError GSDFiles.write_particles_typeid!(h, [0,2])  # NT=2 -> ids must be 0 or 1

    # 3) Bad shapes for position/velocity
    @test_throws AssertionError GSDFiles.write_particles_position!(h, Float32[0 1; 2 3])  # M≠3
    @test_throws AssertionError GSDFiles.write_particles_velocity!(h, Float32[0 0 0 0])   # not N×3

    # 4) particles/N negative
    @test_throws AssertionError GSDFiles.write_particles_N!(h, -1)

    # Close cleanly
    GSDFiles.close_gsd(h)
end

# ------------------------------------
# Types table with long names (padding & NUL)
# ------------------------------------
@testset "Types table encoding with long names" begin
    tmp = mktempdir()
    path = joinpath(tmp, "types_long.gsd")

    w = GSDFiles.GSDWriter(path; application="NonEqSim", schema="hoomd", schema_version=(1,4))
    h = GSDFiles.open_gsd(w)

    # Long second name
    types = ["A", "LongTypeNameXYZ"]
    GSDFiles.write_particles_types!(h, types)
    GSDFiles.end_frame!(h)
    GSDFiles.close_gsd(h)

    io = open(path, "r")
    hdr = read_header(io)
    names = read_names(io, hdr)
    entries = read_index(io, hdr)
    et = only(find_entries_by_name(entries, names, "particles/types"))
    vec = read_chunk_vector(io, et)
    NT = Int(et.N); M = Int(et.M)
    decoded = decode_types_table(vec, NT, M)
    @test decoded == types
    @test M ≥ maximum(length(codeunits(s)) for s in types) + 1
    close(io)
end

# ============================================================
# GSDFiles.jl reader tests (exercise writer+reader round-trip)
# ============================================================

using Test
using GSDFiles
using .GSDFiles.HOOMDWriter  # configuration/* and particles/* helpers

# temp filename
_tmpfile(name="reader_roundtrip") = joinpath(mktempdir(), name*".gsd")

# ---- helpers to write a single HOOMD frame ----
function _write_frame_2d!(h::GSDFiles.GSDFilesHandle; step::Integer, Lx::Float32, Ly::Float32,
                          pos::AbstractMatrix{<:Real}, vel::AbstractMatrix{<:Real},
                          types::Vector{String}, typeid0::AbstractVector{<:Integer})
    @assert size(pos,2) == 3 && size(vel,2) == 3  # already padded to 3
    N = size(pos,1)
    GSDFiles.HOOMDWriter.write_configuration_step!(h, step)
    GSDFiles.HOOMDWriter.write_configuration_dimensions!(h, 2)
    # thin Lz=1
    GSDFiles.HOOMDWriter.write_configuration_box!(h, (Lx, Ly, 1.0f0, 0.0f0, 0.0f0, 0.0f0))
    GSDFiles.HOOMDWriter.write_particles_N!(h, N)
    GSDFiles.HOOMDWriter.write_particles_types!(h, types)
    GSDFiles.HOOMDWriter.write_particles_typeid!(h, typeid0)
    GSDFiles.HOOMDWriter.write_particles_position!(h, pos)
    GSDFiles.HOOMDWriter.write_particles_velocity!(h, vel)
    GSDFiles.end_frame!(h)
end

function _write_frame_3d!(h::GSDFiles.GSDFilesHandle; step::Integer, L::NTuple{3,Float32},
                          pos::AbstractMatrix{<:Real}, vel::AbstractMatrix{<:Real},
                          types::Vector{String}, typeid0::AbstractVector{<:Integer})
    @assert size(pos,2) == 3 && size(vel,2) == 3
    N = size(pos,1)
    GSDFiles.HOOMDWriter.write_configuration_step!(h, step)
    GSDFiles.HOOMDWriter.write_configuration_dimensions!(h, 3)
    GSDFiles.HOOMDWriter.write_configuration_box!(h, (L[1], L[2], L[3], 0.0f0, 0.0f0, 0.0f0))
    GSDFiles.HOOMDWriter.write_particles_N!(h, N)
    GSDFiles.HOOMDWriter.write_particles_types!(h, types)
    GSDFiles.HOOMDWriter.write_particles_typeid!(h, typeid0)
    GSDFiles.HOOMDWriter.write_particles_position!(h, pos)
    GSDFiles.HOOMDWriter.write_particles_velocity!(h, vel)
    GSDFiles.end_frame!(h)
end

# convenient position/velocity padding (2D -> N×3 with z=0)
_pad3(X::AbstractMatrix{<:Real}) = hcat(X, zeros(eltype(X), size(X,1), 1))

# -----------------------------
@testset "Reader: 2D round-trip, multi-frame" begin
    path = _tmpfile("reader_2d")
    w = GSDFiles.GSDWriter(path; application="Test", schema="hoomd", schema_version=(1,4))
    h = GSDFiles.open_gsd(w)

    N = 5
    pos2a = [0.1 0.2; 0.3 0.4; 0.5 0.6; 0.7 0.8; 0.9 1.0] .|> Float32
    vel2a = [ 0.01 0.02; 0.03 0.04; 0.05 0.06; 0.07 0.08; 0.09 0.10] .|> Float32
    pos2b = pos2a .+ Float32(0.5)
    vel2b = vel2a .* Float32(2)

    pos3a = _pad3(pos2a); vel3a = _pad3(vel2a)
    pos3b = _pad3(pos2b); vel3b = _pad3(vel2b)

    types = ["A","B"]
    typeid0 = UInt32.(rand(0:1, N))

    _write_frame_2d!(h; step=1, Lx=10f0, Ly=12f0, pos=pos3a, vel=vel3a, types=types, typeid0=typeid0)
    _write_frame_2d!(h; step=5, Lx=10f0, Ly=12f0, pos=pos3b, vel=vel3b, types=types, typeid0=typeid0)
    GSDFiles.close_gsd(h)

    r = GSDFiles.open_read(path)
    @test GSDFiles.nframes(r) == 2

    f1 = GSDFiles.read_frame(r, 1)
    @test f1.configuration.step == UInt64(1)
    @test f1.configuration.dimensions == UInt8(2)
    @test collect(f1.configuration.box) == Float32[10, 12, 1, 0, 0, 0]
    @test f1.particles.N == N
    @test Vector{String}(f1.particles.types) == types
    @test f1.particles.typeid == typeid0
    @test size(f1.particles.position) == (N,3)
    @test size(f1.particles.velocity) == (N,3)
    @test all(isapprox.(f1.particles.position[:,3], 0f0))

    f2 = GSDFiles.read_frame(r, 2)
    @test f2.configuration.step == UInt64(5)
    @test f2.particles.position ≈ pos3b
    @test f2.particles.velocity ≈ vel3b

    # bounds
    @test_throws BoundsError GSDFiles.read_frame(r, 0)
    @test_throws BoundsError GSDFiles.read_frame(r, 3)

    GSDFiles.close(r)
end

# -----------------------------
@testset "Reader: 3D round-trip, zero velocities" begin
    path = _tmpfile("reader_3d")
    w = GSDFiles.GSDWriter(path; application="Test", schema="hoomd", schema_version=(1,4))
    h = GSDFiles.open_gsd(w)

    N = 4
    pos3 = [0.1 0.2 0.3;
            0.4 0.5 0.6;
            0.7 0.8 0.9;
            1.0 1.1 1.2] .|> Float32
    vel3 = zeros(Float32, N, 3)  # explicitly zero velocities
    L = (8f0, 9f0, 10f0)
    types = ["X","Y","Z"]
    typeid0 = fill(UInt32(2), N)  # "Z" = id 2

    _write_frame_3d!(h; step=0, L=L, pos=pos3, vel=vel3, types=types, typeid0=typeid0)
    GSDFiles.close_gsd(h)

    r = GSDFiles.open_read(path)
    @test GSDFiles.nframes(r) == 1
    f = GSDFiles.read_frame(r, 1)
    @test f.configuration.step == UInt64(0)
    @test f.configuration.dimensions == UInt8(3)
    @test collect(f.configuration.box) == Float32[L[1],L[2],L[3],0,0,0]
    @test f.particles.N == N
    @test f.particles.typeid == typeid0
    @test f.particles.position ≈ pos3
    @test all(isapprox.(f.particles.velocity, 0f0))
    GSDFiles.close(r)
end

# -----------------------------
@testset "Reader: close() idempotency and extra chunks" begin
    path = _tmpfile("reader_extra")
    w = GSDFiles.GSDWriter(path; application="Test", schema="hoomd", schema_version=(1,4))
    h = GSDFiles.open_gsd(w)

    # minimal single particle
    N = 1
    pos3 = Float32[0 0 0]
    vel3 = Float32[0 0 0]
    types = ["A"]
    typeid0 = UInt32[0]

    # --- write everything into the SAME frame (no end_frame! until the end) ---
    GSDFiles.HOOMDWriter.write_configuration_step!(h, 42)
    GSDFiles.HOOMDWriter.write_configuration_dimensions!(h, 2)
    GSDFiles.HOOMDWriter.write_configuration_box!(h, (5f0, 6f0, 1f0, 0f0, 0f0, 0f0))
    GSDFiles.HOOMDWriter.write_particles_N!(h, N)
    GSDFiles.HOOMDWriter.write_particles_types!(h, types)
    GSDFiles.HOOMDWriter.write_particles_typeid!(h, typeid0)
    GSDFiles.HOOMDWriter.write_particles_position!(h, pos3)
    GSDFiles.HOOMDWriter.write_particles_velocity!(h, vel3)

    # Extra topology chunks in the SAME frame
    GSDFiles.HOOMDWriter.write_bonds_types!(h, ["A-A"])
    GSDFiles.HOOMDWriter.write_bonds_N!(h, 0)
    GSDFiles.HOOMDWriter.write_bonds_typeid!(h, UInt32[])
    GSDFiles.HOOMDWriter.write_bonds_group!(h, reshape(UInt32[], 0, 2))

    # Now end the frame exactly once
    GSDFiles.end_frame!(h)
    GSDFiles.close_gsd(h)

    r = GSDFiles.open_read(path)
    @test GSDFiles.nframes(r) == 1
    f = GSDFiles.read_frame(r, 1)
    @test f.configuration.step == UInt64(42)
    @test f.particles.N == 1
    GSDFiles.close(r)
    @test_nowarn GSDFiles.close(r)   # idempotent close
end

@testset "Reader: bonds/angles/dihedrals/impropers round-trip" begin
    # temp file
    path = joinpath(mktempdir(), "topo.gsd")

    # ---------- WRITE ----------
    w = GSDFiles.GSDWriter(path; application="TopoTest", schema="hoomd", schema_version=(1,4))
    h = GSDFiles.open_gsd(w)

    # Minimal particle set: 5 atoms (0-based indices in topology)
    N = 5
    pos = Float32[
        0 0 0;
        1 0 0;
        2 0 0;
        0 1 0;
        0 2 0
    ]
    vel = zeros(Float32, N, 3)
    types = ["A","B","C"]
    typeid = UInt32[0,0,1,1,2]

    # Configuration
    GSDFiles.HOOMDWriter.write_configuration_step!(h, 123)
    GSDFiles.HOOMDWriter.write_configuration_dimensions!(h, 3)
    GSDFiles.HOOMDWriter.write_configuration_box!(h, (10f0, 10f0, 10f0, 0f0, 0f0, 0f0))

    # Particles
    GSDFiles.HOOMDWriter.write_particles_N!(h, N)
    GSDFiles.HOOMDWriter.write_particles_types!(h, types)
    GSDFiles.HOOMDWriter.write_particles_typeid!(h, typeid)
    GSDFiles.HOOMDWriter.write_particles_position!(h, pos)
    GSDFiles.HOOMDWriter.write_particles_velocity!(h, vel)

    # --- Topology: Bonds (N×2) ---
    bond_types = ["A-B", "B-C"]
    bond_typeid = UInt32[0, 1]
    bond_group = UInt32[
        0 1;
        2 3
    ]
    GSDFiles.HOOMDWriter.write_bonds_types!(h, bond_types)
    GSDFiles.HOOMDWriter.write_bonds_N!(h, size(bond_group,1))
    GSDFiles.HOOMDWriter.write_bonds_typeid!(h, bond_typeid)
    GSDFiles.HOOMDWriter.write_bonds_group!(h, bond_group)

    # --- Topology: Angles (N×3) ---
    angle_types = ["A-B-C"]
    angle_typeid = UInt32[0]
    angle_group = UInt32[
        0 1 2
    ]
    GSDFiles.HOOMDWriter.write_angles_types!(h, angle_types)
    GSDFiles.HOOMDWriter.write_angles_N!(h, size(angle_group,1))
    GSDFiles.HOOMDWriter.write_angles_typeid!(h, angle_typeid)
    GSDFiles.HOOMDWriter.write_angles_group!(h, angle_group)

    # --- Topology: Dihedrals (N×4) ---
    dihedral_types = ["A-B-C-D"]
    dihedral_typeid = UInt32[0]
    dihedral_group = UInt32[
        0 1 2 3
    ]
    GSDFiles.HOOMDWriter.write_dihedrals_types!(h, dihedral_types)
    GSDFiles.HOOMDWriter.write_dihedrals_N!(h, size(dihedral_group,1))
    GSDFiles.HOOMDWriter.write_dihedrals_typeid!(h, dihedral_typeid)
    GSDFiles.HOOMDWriter.write_dihedrals_group!(h, dihedral_group)

    # --- Topology: Impropers (N×4) ---
    improper_types = ["A-B-C-D"]
    improper_typeid = UInt32[0]
    improper_group = UInt32[
        1 0 3 4
    ]
    GSDFiles.HOOMDWriter.write_impropers_types!(h, improper_types)
    GSDFiles.HOOMDWriter.write_impropers_N!(h, size(improper_group,1))
    GSDFiles.HOOMDWriter.write_impropers_typeid!(h, improper_typeid)
    GSDFiles.HOOMDWriter.write_impropers_group!(h, improper_group)

    # Close the frame and file
    GSDFiles.end_frame!(h)
    GSDFiles.close_gsd(h)

    # ---------- READ ----------
    r = GSDFiles.open_read(path)
    @test GSDFiles.nframes(r) == 1

    # sanity check the main frame too
    f = GSDFiles.read_frame(r, 1)
    @test f.configuration.step == UInt64(123)
    @test f.particles.N == N
    @test f.particles.types == types

    # Bonds
    b = GSDFiles.read_bonds(r, 1)
    @test b.N == 2
    @test b.types == bond_types
    @test b.typeid == bond_typeid
    @test size(b.group) == (2,2)
    @test b.group == bond_group

    # Angles
    a = GSDFiles.read_angles(r, 1)
    @test a.N == 1
    @test a.types == angle_types
    @test a.typeid == angle_typeid
    @test size(a.group) == (1,3)
    @test a.group == angle_group

    # Dihedrals
    d = GSDFiles.read_dihedrals(r, 1)
    @test d.N == 1
    @test d.types == dihedral_types
    @test d.typeid == dihedral_typeid
    @test size(d.group) == (1,4)
    @test d.group == dihedral_group

    # Impropers
    im = GSDFiles.read_impropers(r, 1)
    @test im.N == 1
    @test im.types == improper_types
    @test im.typeid == improper_typeid
    @test size(im.group) == (1,4)
    @test im.group == improper_group

    # idempotent close
    GSDFiles.close(r)
    @test_nowarn GSDFiles.close(r)
end