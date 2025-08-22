# GSDFiles.jl

A pure-Julia writer/reader for the GSD v2 format (HOOMD schema compatible), tested with OVITO.

## Features
- Write OVITO/HOOMD-compatible `configuration/*` and `particles/*` chunks
- Topology: bonds, angles, dihedrals, impropers
- Row-major compliance for matrices
- Reader for single frames and topology families

## Quick start

```julia
using GSDFiles

w = GSDFiles.GSDWriter("demo.gsd"; application="MyApp", schema="hoomd", schema_version=(1,4))
h = GSD.open_gsd(w)

GSD.write_configuration_step!(h, 0)
GSD.write_configuration_dimensions!(h, 3)
GSD.write_configuration_box!(h, (10f0,10f0,10f0,0,0,0))

N = 2
GSD.write_particles_N!(h, N)
GSD.write_particles_types!(h, ["A","B"])
GSD.write_particles_typeid!(h, UInt32[0,1])
GSD.write_particles_position!(h, Float32[0 0 0; 1 0 0])
GSD.write_particles_velocity!(h, zeros(Float32, 2, 3))

GSD.end_frame!(h)
GSD.close_gsd(h)

r = GSD.open_read("demo.gsd")
f = GSD.read_frame(r, 1)
GSD.close(r)