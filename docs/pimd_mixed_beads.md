# Fixed mixed-bead paths

`molsimflow.postprocess.pimd.mixed_beads` records fixed atom identities and
nested bead counts. It distinguishes independent coordinates from the complete
configurations passed to a potential. This is separate from a CV, bias mode,
coordinate-contraction parameter, FES estimator or path reweighting scheme.

```python
from molsimflow.postprocess.pimd.mixed_beads import MixedBeadTopology

topology = MixedBeadTopology(atom_ids=(1, 2, 3), bead_counts=(8, 2, 1), partitions=8)
metadata = topology.to_metadata()
```

The versioned metadata contains `schema`, `nuclear_ensemble`, `mapping`,
`partitions`, `atom_ids` and `bead_counts`. Its nuclear ensemble is
`fixed_mixed_bead` and its mapping is `contiguous_nested`. Atom IDs are unique
positive integers in the declared atom-axis order. Counts and partitions are
positive powers of two, with the largest count equal to the partition count.
The all-classical and uniform-P endpoints are admitted. This does not establish
bead convergence to quantum statistics.

For zero-based partition b and count p_i, the unique bead index is floor(b*p_i/P).
The representative partition for unique bead a is a*P/p_i. Unassigned atom IDs
are not inferred, and atom identities may not change with the reaction center.
`bead_indices()` and `representatives()` describe the expanded configurations.
`unique_coordinates()` checks repeated coordinates before returning one array
per atom. Copies must agree in continuous unwrapped Cartesian coordinates;
wrapped proximity is insufficient, and inconsistent copies are not averaged.

For an atom-resolved observable, each unique bead has weight 1/p_i. Averaging
the corresponding expanded values over P configurations gives the same result.
For a complete-configuration energy E, the physical potential is sum_b E_b/P.
`physical_force_pullback()` returns the gradient with respect to each unique
coordinate, sum_{b in bucket} f_b/P. It does not apply fictitious-mass rescaling
or add spring forces. Counts, coordinate dimensions and finite values are
validated explicitly.

These per-bead observable weights are not independent statistical sample weights.
One complete path has one frame weight; uncertainty uses frames and independent
replicas. Pair or many-body observables must retain the declared joint
configuration mapping. An all-pairs bead average generally defines a different
finite-bead observable or Hamiltonian.

## Native LAMMPS dumps

Use unwrapped positions and the `pimd/mixed` per-atom metadata:

```text
dump path all custom 100 path.dump id type xu yu zu vx vy vz f_pi[1] f_pi[2] f_pi[3]
dump_modify path format float %.17g
```

Each partition must write a distinct file. Supply a hash-locked manifest mapping
zero-based partition IDs to those immutable files. The columns from fix `pi`
mean bead count, one-based unique bead index and representative flag. A different
fix ID can be selected through `metadata_columns`.

```python
from pathlib import Path
from molsimflow.postprocess.pimd.mixed_beads import iter_mixed_pimd_path_frames

paths = {b: Path(f"partition-{b}.dump") for b in range(8)}
frames = iter_mixed_pimd_path_frames(
    paths, topology=topology, expected_identity={1: 1, 2: 2, 3: 2},
    copy_tolerance=1e-10, velocity_copy_tolerance=1e-10,
)
for frame in frames:
    unique_positions = frame.topology.unique_coordinates(frame.positions)
```

Distance and velocity tolerances use their respective input units; this reader
performs no implicit unit conversion. All velocity columns are optional together.
Every frame is aligned by atom ID and physical timestep. The reader rejects
missing partitions, duplicate paths, changed identities/counts/indices, copy
drift, non-finite values, different frame counts, duplicated timesteps and
incomplete tails. Exhaust the iterator to validate the entire input.

This first reader reuses the existing orthorhombic dump parser. Boxes must remain
fixed, with each direction `pp` or `ff`; tilted and changing cells are explicitly
rejected. The native integrator's supported cell set can be larger than this
reader's. Existing `PimdPathFrame` and `iter_pimd_path_frames` retain their wrapped,
uniform-path semantics.

## FES boundary

The existing quantum-FES pipeline does not yet have an admitted mixed-ensemble
output schema. If `source.nuclear_quantization` declares heterogeneous counts,
it fails before reading trajectories or writing a misleading quantum-FES result.
Uniform metadata must agree with the number of bead CV files. Existing contracts
without this field retain their behavior; absence of metadata cannot prove that
an arbitrary data set was generated with uniform quantization.

This module admits topology and complete frames and exposes weighted-coordinate
arithmetic. It does not establish equilibrium, a correct bias history, converged
free energies, a full-quantum target, or the accuracy of a learned potential.
