module GSDFiles

import Base: close

# =========================
# Public API (exports)
# =========================
export GSDWriter, write_chunk!, write_chunk_raw!, end_frame!, close!,
       open_gsd, close_gsd, end_frame!,
       write_configuration_box!, write_configuration_step!, write_configuration_dimensions!,
       write_particles_N!, write_particles_types!, write_particles_typeid!,
       write_particles_position!, write_particles_velocity!

export open_read, close, nframes, read_frame,
       read_bonds, read_angles, read_dihedrals, read_impropers

# =========================
# File-layer constants (GSD v2.x)
# =========================
const GSD_MAGIC::UInt64 = 0x65DF65DF65DF65DF
const GSD_NAME_SIZE::Int = 64
const HDR_BYTES        = 256
const IDX_ENTRY_BYTES  = 32

# Map Julia dtype symbols to GSD enum codes (matches enum gsd_type in gsd.h)
const GSD_TYPE_UINT8   ::UInt8 = 0
const GSD_TYPE_UINT16  ::UInt8 = 1
const GSD_TYPE_UINT32  ::UInt8 = 2
const GSD_TYPE_UINT64  ::UInt8 = 3
const GSD_TYPE_INT8    ::UInt8 = 4
const GSD_TYPE_INT16   ::UInt8 = 5
const GSD_TYPE_INT32   ::UInt8 = 6
const GSD_TYPE_INT64   ::UInt8 = 7
const GSD_TYPE_FLOAT   ::UInt8 = 8   # 32-bit
const GSD_TYPE_DOUBLE  ::UInt8 = 9   # 64-bit
const GSD_TYPE_CHAR    ::UInt8 = 10  # 8-bit character

const _DTYPE = Dict{Symbol,UInt8}(
    :uint8   => GSD_TYPE_UINT8,
    :uint16  => GSD_TYPE_UINT16,
    :uint32  => GSD_TYPE_UINT32,
    :uint64  => GSD_TYPE_UINT64,
    :int8    => GSD_TYPE_INT8,
    :int16   => GSD_TYPE_INT16,
    :int32   => GSD_TYPE_INT32,
    :int64   => GSD_TYPE_INT64,
    :float32 => GSD_TYPE_FLOAT,
    :float64 => GSD_TYPE_DOUBLE,
    :char    => GSD_TYPE_CHAR,
)

# Pack versions: (major<<16) | minor
pack_version(major::Integer, minor::Integer)::UInt32 =
    (UInt32(major) << 16) | (UInt32(minor) & 0x0000_FFFF)

# =========================
# Header (exactly 256 bytes)
# =========================
mutable struct Header
    magic::UInt64
    index_location::UInt64
    index_allocated_entries::UInt64
    namelist_location::UInt64
    namelist_allocated_entries::UInt64   # v2: count of 64-byte name segments
    schema_version::UInt32
    gsd_version::UInt32
    application::NTuple{64,UInt8}
    schema::NTuple{64,UInt8}
    reserved::NTuple{80,UInt8}
end

function write_header!(io::IO, h::Header)
    seek(io, 0)
    write(io, h.magic)
    write(io, h.index_location)
    write(io, h.index_allocated_entries)
    write(io, h.namelist_location)
    write(io, h.namelist_allocated_entries)
    write(io, h.schema_version)
    write(io, h.gsd_version)
    write(io, h.application...)  # splat NTuple bytes
    write(io, h.schema...)       # splat NTuple bytes
    write(io, h.reserved...)     # splat NTuple bytes
    return nothing
end

update_header_fields!(io::IO, h::Header) = write_header!(io, h)

# =========================
# Name list (v2.x: 0-separated blob)
# =========================
function _pack_padded(s::AbstractString, width::Int)
    bytes = codeunits(s)
    nt = ntuple(i -> UInt8(i <= length(bytes) ? bytes[i] : 0x00), width)
    return nt
end

function ensure_name_id!(writer, name::AbstractString)::UInt16
    id = get(writer.name_to_id, name, UInt16(0xFFFF))
    if id != 0xFFFF
        return id
    end
    new_id = UInt16(length(writer.name_to_id))   # 0-based
    append!(writer.names_blob, codeunits(String(name)))
    push!(writer.names_blob, 0x00)               # NUL terminator
    writer.name_to_id[name] = new_id
    return new_id
end

function _pad_to_multiple!(v::Vector{UInt8}, m::Integer)
    r = length(v) % m
    if r != 0
        append!(v, zeros(UInt8, m - r))
    end
    return v
end

# =========================
# Row-major flattener
# =========================
function rowmajor(A::AbstractMatrix{T}) where {T}
    N, M = size(A)
    out = Vector{T}(undef, N*M)
    k = 1
    @inbounds for i in 1:N, j in 1:M
        out[k] = A[i,j]; k += 1
    end
    return out
end

# =========================
# Index entry (32 bytes)
# =========================
struct IndexEntry
    frame::UInt64
    N::UInt64
    location::Int64
    M::UInt32
    id::UInt16
    type::UInt8
    flags::UInt8
end

# =========================
# The writer object
# =========================
mutable struct GSDWriter
    io::IO
    header::Header
    index::Vector{IndexEntry}
    name_to_id::Dict{String,UInt16}
    names_blob::Vector{UInt8}
    current_frame::UInt64
end

function GSDWriter(path::AbstractString; application="NonEqSim", schema="hoomd", schema_version=(1,4))
    io = open(path, "w")   # binary by default
    hdr = Header(
        GSD_MAGIC,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,  # v2: 64B segments in namelist
        pack_version(schema_version[1], schema_version[2]),
        pack_version(2, 0),  # GSD file-layer 2.0
        _pack_padded(application, 64),
        _pack_padded(schema, 64),
        ntuple(_->0x00, 80),
    )
    write_header!(io, hdr)
    return GSDWriter(io, hdr, IndexEntry[], Dict{String,UInt16}(), UInt8[], 0x0000000000000000)
end

# =========================
# Chunk writers
# =========================

# Normal path: dtype given as a Symbol is mapped via _DTYPE
function write_chunk!(w::GSDWriter, name::AbstractString; dtype::Symbol, N::Integer, M::Integer, data::AbstractVector)
    # Hard guard for common schema gotcha:
    if name === "configuration/step" && dtype !== :uint64
        error("configuration/step must be :uint64, got $dtype")
    end
    code = get(_DTYPE, dtype) do
        throw(ArgumentError("Unsupported dtype: $dtype"))
    end
    id = ensure_name_id!(w, name)
    seekend(w.io)
    loc = position(w.io)
    write(w.io, data)
    push!(w.index, IndexEntry(UInt64(w.current_frame), UInt64(N), Int64(loc), UInt32(M), id, code, 0x00))
    return nothing
end

# Raw path: bypass _DTYPE, set the exact on-disk type code explicitly
function write_chunk_raw!(w::GSDWriter, name::AbstractString; type_code::UInt8, N::Integer, M::Integer, data::AbstractVector)
    id = ensure_name_id!(w, name)
    seekend(w.io)
    loc = position(w.io)
    write(w.io, data)
    push!(w.index, IndexEntry(UInt64(w.current_frame), UInt64(N), Int64(loc), UInt32(M), id, type_code, 0x00))
    return nothing
end

# =========================
# Frame control
# =========================
function end_frame!(w::GSDWriter)
    w.current_frame += 0x0000000000000001
    return nothing
end

# =========================
# Finalize
# =========================
function close!(w::GSDWriter)
    # Build namelist: ensure trailing empty name and 64B align
    names_blob = copy(w.names_blob)
    if isempty(names_blob) || names_blob[end] != 0x00
        push!(names_blob, 0x00)
    end
    push!(names_blob, 0x00)
    _pad_to_multiple!(names_blob, GSD_NAME_SIZE)

    # Write index first (sorted)
    sort!(w.index; by = e -> (e.frame, e.id))
    seekend(w.io)
    idx_loc = position(w.io)
    for e in w.index
        write(w.io, e.frame); write(w.io, e.N); write(w.io, e.location)
        write(w.io, e.M); write(w.io, e.id); write(w.io, e.type); write(w.io, e.flags)
    end
    # Sentinel (location==0)
    write(w.io, UInt64(0)); write(w.io, UInt64(0)); write(w.io, Int64(0))
    write(w.io, UInt32(0)); write(w.io, UInt16(0)); write(w.io, UInt8(0)); write(w.io, UInt8(0))
    idx_alloc = UInt64(length(w.index) + 1)

    # Namelist
    seekend(w.io)
    nl_loc = position(w.io)
    write(w.io, names_blob)
    nl_segments = UInt64(div(length(names_blob), GSD_NAME_SIZE))

    # Patch header
    w.header.index_location             = UInt64(idx_loc)
    w.header.index_allocated_entries    = idx_alloc
    w.header.namelist_location          = UInt64(nl_loc)
    w.header.namelist_allocated_entries = nl_segments
    update_header_fields!(w.io, w.header)
    Base.close(w.io)
    return nothing
end

# =========================
# High-level shim handle
# =========================
mutable struct GSDFilesHandle
    user::GSDWriter
    frame::UInt32
    typenames::Vector{String}
    type_index::Dict{String,UInt32}
end

open_gsd(w::GSDWriter) = GSDFilesHandle(w, 0x00000000, String[], Dict{String,UInt32}())
close_gsd(h::GSDFilesHandle) = close!(h.user)
function end_frame!(h::GSDFilesHandle)
    h.frame += 0x00000001
    return end_frame!(h.user)
end


# =========================
# Reader (GSD v2, OVITO type codes)
# =========================

# OVITO-style type codes used by our writer (pinned)
const _R_UINT8   = UInt8(1)
const _R_UINT32  = UInt8(3)
const _R_UINT64  = UInt8(4)
const _R_INT8    = UInt8(5)
const _R_FLOAT32 = UInt8(9)

# Read-only header subset
struct _ROHeader
    index_location::UInt64
    namelist_location::UInt64
    namelist_segments::UInt64  # number of 64-byte segments
end

mutable struct GSDReader
    io::IO
    hdr::_ROHeader
    names::Vector{String}
    index::Vector{IndexEntry}
end

function open_read(path::AbstractString)::GSDReader
    io = open(path, "r")
    # header (256 bytes)
    seek(io, 0)
    magic = read(io, UInt64)
    magic == GSD_MAGIC || (close(io); error("Bad GSD magic in $path"))
    idx_loc   = read(io, UInt64)
    _idx_alloc= read(io, UInt64)
    nm_loc    = read(io, UInt64)
    nm_alloc  = read(io, UInt64)  # number of 64-byte segments
    _schema_v = read(io, UInt32)
    _gsd_v    = read(io, UInt32)
    seek(io, 256)
    hdr = _ROHeader(idx_loc, nm_loc, nm_alloc)

    # namelist: 0-separated, padded to 64B segments
    names = String[]
    total_bytes = Int(nm_alloc) * GSD_NAME_SIZE
    if total_bytes > 0
        seek(io, Int(nm_loc))
        buf = read(io, total_bytes)
        cur = IOBuffer()
        for b in buf
            if b == 0x00
                push!(names, String(take!(cur)))
            else
                write(cur, b)
            end
        end
        while !isempty(names) && names[end] == ""
            pop!(names)
        end
    end

    # index: array of 32-byte entries, terminated by sentinel with location==0
    seek(io, Int(idx_loc))
    idx = IndexEntry[]
    while true
        frame    = read(io, UInt64)
        N        = read(io, UInt64)
        location = read(io, Int64)
        M        = read(io, UInt32)
        id       = read(io, UInt16)
        typ      = read(io, UInt8)
        flags    = read(io, UInt8)
        if location == 0
            break
        end
        push!(idx, IndexEntry(frame, N, location, M, id, typ, flags))
    end

    return GSDReader(io, hdr, names, idx)
end

close(r::GSDReader) = (close(r.io); nothing)

function nframes(r::GSDReader)::Int
    isempty(r.index) && return 0
    return Int(maximum(e -> e.frame, r.index)) + 1
end

# internal helpers
_entries_for_frame(r::GSDReader, fid::UInt64) = filter(e -> e.frame == fid, r.index)
_name_id(r::GSDReader, name::AbstractString) = begin
    i = findfirst(==(name), r.names)
    i === nothing ? nothing : UInt16(i-1)
end
_byname(r::GSDReader, ents::Vector{IndexEntry}, name::AbstractString) = begin
    id = _name_id(r, name)
    id === nothing ? IndexEntry[] : filter(e -> e.id == id, ents)
end

# decode particles/types (int8 NT×M row-major)
function _decode_types(io::IO, e::IndexEntry)::Vector{String}
    e.type == _R_INT8 || error("particles/types dtype mismatch")
    NT = Int(e.N); M = Int(e.M)
    seek(io, e.location)
    raw = read!(io, Array{Int8}(undef, NT*M))
    A = reshape(raw, (M, NT))'  # row-major -> transpose
    out = String[]
    for r in 1:NT
        bytes = Vector{UInt8}(A[r, :])
        nul = findfirst(==(0x00), bytes)
        nul !== nothing && resize!(bytes, nul-1)
        push!(out, String(bytes))
    end
    return out
end

# read scalar/vector/matrix helpers
_read_scalar(io::IO, e::IndexEntry, ::Type{T}) where {T} = (seek(io, e.location); read(io, T))
function _read_vec(io::IO, e::IndexEntry, ::Type{T}) where {T}
    seek(io, e.location)
    read!(io, Array{T}(undef, Int(e.N)))
end
function _read_mat_f32(io::IO, e::IndexEntry)
    e.type == _R_FLOAT32 || error("expected float32 chunk")
    seek(io, e.location)
    N = Int(e.N); M = Int(e.M)
    flat = read!(io, Array{Float32}(undef, N*M))
    reshape(flat, (M, N))'  # row-major -> N×M
end

# Read N×M row-major UInt32 -> return Matrix{UInt32}(N,M)
function _read_mat_u32(io::IO, e::IndexEntry)
    e.type == _R_UINT32 || error("expected uint32 chunk")
    seek(io, e.location)
    N = Int(e.N); M = Int(e.M)
    flat = read!(io, Array{UInt32}(undef, N*M))
    reshape(flat, (M, N))'  # row-major -> transpose
end

# Find-only-if-present helper: returns `nothing` if the chunk name is absent in the frame
function _maybe_one(r::GSDReader, ents::Vector{IndexEntry}, name::AbstractString)
    id = _name_id(r, name)
    id === nothing && return nothing
    matches = filter(e -> e.id == id, ents)
    isempty(matches) && return nothing
    return only(matches)
end

# Try several possible chunk names; return the first that exists in this frame
function _maybe_one_of(r::GSDReader, ents::Vector{IndexEntry}, names::Vector{String})
    for n in names
        x = _maybe_one(r, ents, n)
        x === nothing || return x
    end
    return nothing
end

struct Frame
    configuration::NamedTuple
    particles::NamedTuple
end

function read_frame(r::GSDReader, i::Integer)::Frame
    nf = nframes(r)
    1 ≤ i ≤ nf || throw(BoundsError("frame index $i out of 1:$nf"))
    fid = UInt64(i-1)
    ents = _entries_for_frame(r, fid)

    # configuration/*
    step_e = only(_byname(r, ents, "configuration/step"))
    step_e.type == _R_UINT64 || error("configuration/step dtype mismatch")
    step = _read_scalar(r.io, step_e, UInt64)

    dim_e = only(_byname(r, ents, "configuration/dimensions"))
    dim_e.type == _R_UINT8 || error("configuration/dimensions dtype mismatch")
    dims = _read_scalar(r.io, dim_e, UInt8)

    box_e = only(_byname(r, ents, "configuration/box"))
    box_e.type == _R_FLOAT32 || error("configuration/box dtype mismatch")
    box = _read_vec(r.io, box_e, Float32)  # length 6

    # particles/*
    N_e = only(_byname(r, ents, "particles/N"))
    N_e.type == _R_UINT32 || error("particles/N dtype mismatch")
    N = Int(_read_scalar(r.io, N_e, UInt32))

    types_e = only(_byname(r, ents, "particles/types"))
    types = _decode_types(r.io, types_e)

    tid_e = only(_byname(r, ents, "particles/typeid"))
    tid_e.type == _R_UINT32 || error("particles/typeid dtype mismatch")
    typeid = _read_vec(r.io, tid_e, UInt32)

    pos_e = only(_byname(r, ents, "particles/position"))
    position = _read_mat_f32(r.io, pos_e)  # N×3

    vel_e = only(_byname(r, ents, "particles/velocity"))
    velocity = _read_mat_f32(r.io, vel_e)  # N×3

    conf = (step = step, dimensions = dims, box = box)
    parts = (N = N, types = types, typeid = typeid, position = position, velocity = velocity)
    Frame(conf, parts)
end


# ---- Topology readers -----------------------------------------------------
# Each returns (N, types, typeid, group) where group is N×k (k=2 bonds, 3 angles, 4 dihedrals/impropers)

function read_bonds(r::GSDReader, i::Integer)
    nf = nframes(r); 1 ≤ i ≤ nf || throw(BoundsError("frame index $i out of 1:$nf"))
    fid = UInt64(i-1); ents = _entries_for_frame(r, fid)

    # types table (optional)
    types_e = _maybe_one(r, ents, "bonds/types")
    types = types_e === nothing ? String[] : _decode_types(r.io, types_e)

    N_e  = _maybe_one(r, ents, "bonds/N")
    tid_e = _maybe_one(r, ents, "bonds/typeid")
    grp_e = _maybe_one(r, ents, "bonds/group")

    if N_e === nothing || tid_e === nothing || grp_e === nothing
        return (N = 0, types = types, typeid = UInt32[], group = reshape(UInt32[], 0, 2))
    end

    N = Int(_read_scalar(r.io, N_e, UInt32))
    tid_e.type == _R_UINT32 || error("bonds/typeid dtype mismatch")
    grp_e.type == _R_UINT32 || error("bonds/group dtype mismatch")
    typeid = _read_vec(r.io, tid_e, UInt32)
    group  = _read_mat_u32(r.io, grp_e)
    @assert size(group,2) == 2 "bonds/group must have 2 columns"
    return (N = N, types = types, typeid = typeid, group = group)
end

function read_angles(r::GSDReader, i::Integer)
    nf = nframes(r); 1 ≤ i ≤ nf || throw(BoundsError("frame index $i out of 1:$nf"))
    fid = UInt64(i-1); ents = _entries_for_frame(r, fid)

    types_e = _maybe_one(r, ents, "angles/types")
    types = types_e === nothing ? String[] : _decode_types(r.io, types_e)

    N_e  = _maybe_one(r, ents, "angles/N")
    tid_e = _maybe_one(r, ents, "angles/typeid")
    grp_e = _maybe_one_of(r, ents, ["angles/angle", "angles/group"])

    if N_e === nothing || tid_e === nothing || grp_e === nothing
        return (N = 0, types = types, typeid = UInt32[], group = reshape(UInt32[], 0, 3))
    end

    N = Int(_read_scalar(r.io, N_e, UInt32))
    tid_e.type == _R_UINT32 || error("angles/typeid dtype mismatch")
    grp_e.type == _R_UINT32 || error("angles/angle dtype mismatch")
    typeid = _read_vec(r.io, tid_e, UInt32)
    group  = _read_mat_u32(r.io, grp_e)
    @assert size(group,2) == 3 "angles/angle must have 3 columns"
    return (N = N, types = types, typeid = typeid, group = group)
end

function read_dihedrals(r::GSDReader, i::Integer)
    nf = nframes(r); 1 ≤ i ≤ nf || throw(BoundsError("frame index $i out of 1:$nf"))
    fid = UInt64(i-1); ents = _entries_for_frame(r, fid)

    types_e = _maybe_one(r, ents, "dihedrals/types")
    types = types_e === nothing ? String[] : _decode_types(r.io, types_e)

    N_e  = _maybe_one(r, ents, "dihedrals/N")
    tid_e = _maybe_one(r, ents, "dihedrals/typeid")
    grp_e = _maybe_one_of(r, ents, ["dihedrals/dihedral", "dihedrals/group"])

    if N_e === nothing || tid_e === nothing || grp_e === nothing
        return (N = 0, types = types, typeid = UInt32[], group = reshape(UInt32[], 0, 4))
    end

    N = Int(_read_scalar(r.io, N_e, UInt32))
    tid_e.type == _R_UINT32 || error("dihedrals/typeid dtype mismatch")
    grp_e.type == _R_UINT32 || error("dihedrals/dihedral dtype mismatch")
    typeid = _read_vec(r.io, tid_e, UInt32)
    group  = _read_mat_u32(r.io, grp_e)
    @assert size(group,2) == 4 "dihedrals/dihedral must have 4 columns"
    return (N = N, types = types, typeid = typeid, group = group)
end

function read_impropers(r::GSDReader, i::Integer)
    nf = nframes(r); 1 ≤ i ≤ nf || throw(BoundsError("frame index $i out of 1:$nf"))
    fid = UInt64(i-1); ents = _entries_for_frame(r, fid)

    types_e = _maybe_one(r, ents, "impropers/types")
    types = types_e === nothing ? String[] : _decode_types(r.io, types_e)

    N_e  = _maybe_one(r, ents, "impropers/N")
    tid_e = _maybe_one(r, ents, "impropers/typeid")
    grp_e = _maybe_one_of(r, ents, ["impropers/improper", "impropers/group"])

    if N_e === nothing || tid_e === nothing || grp_e === nothing
        return (N = 0, types = types, typeid = UInt32[], group = reshape(UInt32[], 0, 4))
    end

    N = Int(_read_scalar(r.io, N_e, UInt32))
    tid_e.type == _R_UINT32 || error("impropers/typeid dtype mismatch")
    grp_e.type == _R_UINT32 || error("impropers/improper dtype mismatch")
    typeid = _read_vec(r.io, tid_e, UInt32)
    group  = _read_mat_u32(r.io, grp_e)
    @assert size(group,2) == 4 "impropers/improper must have 4 columns"
    return (N = N, types = types, typeid = typeid, group = group)
end


# =========================
# Bring in HOOMD helpers
# =========================
include("HOOMDWriter.jl")
using .HOOMDWriter


end # module GSDFiles