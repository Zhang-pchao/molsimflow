# Shared Trajectory Utilities

## Scope

The first shared low-level migration extracts utilities that were previously
embedded in bridge-water dewetting:

- `molsimflow.io.lammps_dump`;
- `molsimflow.postprocess.time_alignment`.

These modules are intentionally small.  They cover the orthorhombic LAMMPS dump
reader and nearest-time alignment behavior already used by migrated workflows.
More trajectory-heavy modules can extend them instead of copying local readers.

## LAMMPS Dump Reader

```python
from molsimflow.io.lammps_dump import iter_lammps_dump_frames

frames = iter_lammps_dump_frames(
    "dump.lammpstrj",
    needed_atom_ids=[1, 2, 3],
    max_frames=100,
)
for frame in frames:
    print(frame.frame_index, frame.timestep, frame.selected_positions[1])
```

The reader supports coordinate columns named `x/y/z`, `xu/yu/zu`, or scaled
`xs/ys/zs`.  Scaled coordinates are converted into box coordinates.  It assumes
orthorhombic box bounds.

## Periodic Geometry

`molsimflow.io.lammps_dump` also provides:

- `box_lengths`;
- `minimum_image_vectors`;
- `periodic_center`;
- `midpoint_minimum_image`;
- `cylinder_membership`.

These helpers are used by `molsimflow.postprocess.bridge_water_dewetting` and
should be reused by future water-orientation, escape-direction, and H-bond
workflow migrations.

## Time Alignment

```python
from molsimflow.postprocess.time_alignment import (
    infer_timestep_time_scale,
    nearest_row_index,
)

idx = nearest_row_index(colvar_rows, time_value=0.125, tolerance=0.001)
scale = infer_timestep_time_scale([0, 1000, 2000], colvar_rows, tolerance=0.001)
```

The helpers operate on row dictionaries with a time column, defaulting to
`time_ns`.
