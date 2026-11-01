# Conservative local-distance callback example

A one-MPI-rank driver for the installed LAMMPS C library. It evaluates an identity-invariant smooth nearest-distance coordinate, a harmonic bias, point forces, anchor reaction, and global bias energy/virial. It does not modify or rebuild LAMMPS.

The coordinate is the regularized soft minimum with a compact quintic taper and a constant reservoir. The NumPy reference is `molsimflow.postprocess.local_pair_coordination.soft_minimum_distance`.

## Contract

Use a fixed orthorhombic fully periodic box whose shortest side exceeds twice the cutoff. Atom IDs must be contiguous from 1 through the sealed atom count. Exactly one MPI rank is required. Selection uses atom IDs, independent of local sorting and water identity. The anchor cannot be among the selected points.

Configuration starts with: `anchor_id atom_count kappa target softness regularizer taper_inner cutoff force_limit`. Remaining whitespace-separated integers are the unique selected atom IDs. Quantities use the LAMMPS input unit system; the caller must ensure consistency.

The setup input must create `fix CV all external pf/callback 1 1` and contain no subsequent run before the driver attaches its callback. The driver then executes a separate run input. When using `-sf kk`, surround creation of this fix with `suffix off` and `suffix kk` to select the base host fix explicitly. Qualification against the actual installed backend is required.

Invocation: `local_distance_driver config setup.lmp run.lmp [LAMMPS arguments]`. Compile with the matching MPI compiler, C++17, `LAMMPS_LIB_MPI`, `LAMMPS_SMALLBIG`, the installed LAMMPS include path, and `-llammps`.

Use `fix_modify CV energy yes virial yes`. Its scalar is global bias energy; vector entries 1-4 are distance, bias energy, maximum selected-point force magnitude, and anchor-force magnitude. With energy enabled, LAMMPS potential energy already includes the bias: do not add it twice. Global virial is the minimum-image displacement/force pair sum; pressure-based scientific interpretation needs separate validation.

## Verification limits

Compare static values, energies and forces to the independent NumPy implementation; use finite differences of engine energy, periodic translations, and atom permutations. Also test synthetic NVE energy/momentum and full-system throughput before scientific dynamics. Compilation or a static pass does not validate physical chemistry or a reaction coordinate. Case-specific paths, configurations, results and budgets belong outside this reusable example.
