# Bounded native dump stream

This example converts a sorted, complete orthorhombic PPP LAMMPS dump on stdin
into a binary pipe. It evaluates no potential and performs no dynamics.

Build with a C++17 compiler:

```sh
g++ -std=c++17 -O3 -Wall -Wextra -Werror stream.cpp -o stream
python check_stream.py ./stream
```

The identity file contains an atom count, then one `id type` row per atom.
IDs must be contiguous and sorted. Invoke `stream identity_file max_frames`,
where zero means consume the full source. Decompression belongs to the caller.
The required fields are `id type x y z ix iy iz vx vy vz`; box and identities
must remain fixed. Values must be finite and image flags integral.

Each binary frame is a native-endian int64 step, six float64 bounds, then an
N-by-9 float64 matrix `[x,y,z,ix,iy,iz,vx,vy,vz]`. The audited caller requires a
little-endian host, exact byte counts, its own expected cadence and frame count,
and a source SHA256. Always inspect both decompressor and reader exit codes.
A deliberately limited prefix must close the upstream decompressor explicitly.

This is an intentionally narrow pipe, not a replacement for the general text
reader. Frame parity against the general reader and original inputs is required
when binding a new scientific source. Successful parsing does not establish
ensemble, chemical, sampling, or model accuracy.
