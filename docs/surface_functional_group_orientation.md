# Surface functional-group orientation

`surface-functional-group-orientation` measures CH3 and SiOH orientation from
one or more LAMMPS trajectories against a fixed initial surface structure.

## Contract

- CH3 uses the Si-to-C axis; its three C-H bonds are an integrity gate only.
- SiOH reports both Si-to-O and O-to-current-H axes plus the Si-O-H angle.
- Global +z and a configurable local Si-plane normal are reported separately.
- Direct cosine and azimuth histograms are written without smoothing or gap
  interpolation; block variation is descriptive within-trajectory variation.

## Required inputs

- Atom ranges, the initial structure, and the surface reference plane are
  explicit CLI inputs rather than project-specific constants.
- Expected CH3/SiOH site counts are optional identity checks.
- The H-integrity threshold is a gate, not a rule for retroactively relabeling
  reactive events.
