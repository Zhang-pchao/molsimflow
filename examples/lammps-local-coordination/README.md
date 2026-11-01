# Conservative local coordination penalty

One MPI rank, orthorhombic fully periodic cell, fixed contiguous atom IDs.
The driver evaluates U = epsilon * sum_i s(r_i), where s is one below the inner
radius, zero above the cutoff, and a C2 decreasing quintic between them.
Epsilon is nonnegative (energy units). This is a constructed Hamiltonian
intervention, not a chemical substitution or a free-energy estimator.

Configuration: anchor_id atom_count epsilon inner cutoff force_limit,
followed by selected atom IDs. The anchor cannot be selected. The cutoff
must be below half every cell length. All selected atoms and the anchor
receive conservative force contributions; total anchor force is capped
as well as each selected force. The bias energy and virial enter LAMMPS.

Use the installed LAMMPS library and matching MPI compiler. Setup must create
fix CV all external pf/callback 1 1 and enable its energy and virial. The
driver registers the callback and vector length between setup and run input.
Vectors: coordination, bias energy, maximum selected force, anchor force norm.
Retain the common one-rank runtime limitations documented for the distance driver.

The independent NumPy oracle is molsimflow.postprocess.local_pair_coordination
(pair_coordination). Validate static forces, periodicity, energy derivatives,
native conservation and actual-model force differences before research use.

For the validated Kokkos runtime, surround fix CV creation with suffix off and
suffix kk, selecting the host external callback while leaving DeepMD on GPU.
The automatically suffixed external/kk path can abort in repeated static
run-zero evaluations with a concurrent host/device DualView modification.
Keep unique timestep IDs for separately edited static control geometries.
