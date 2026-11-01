# Two conservative local distance biases

This one-rank LAMMPS C-library example composes two previously verified local-distance callbacks. Each creates its own external fix force array, energy and virial; LAMMPS adds both contributions. The selected point sets must be disjoint and share one anchor and sealed atom count. The anchor receives both reactions.

Invocation: joint_distance_driver config1 config2 setup.lmp run.lmp [LAMMPS arguments].

Each configuration uses the existing local-distance format:
anchor_id atom_count kappa target softness regularizer taper_inner cutoff force_limit
followed by the unique selected atom IDs. The setup creates ordinary host fix CV1 all external pf/callback 1 1 and fix CV2 all external pf/callback 1 1. With a Kokkos suffix, create these inside suffix off, restoring the suffix afterwards. Both require fix_modify energy yes virial yes. No run may occur after fix creation until callbacks are attached.

For each fix, vectors 1-4 report distance, energy, maximum point force and anchor force. Potential energy already contains both bias energies. Do not add them again. Individual anchor limits L1 and L2 guarantee the combined anchor magnitude is at most L1+L2. Point sets are disjoint, so point forces do not add across these two callbacks. This cap concerns external forces only; physical-force guards remain separate.

The coordinate is the regularized compact soft minimum implemented by soft_minimum_distance in local_pair_coordination.py. Fully periodic orthorhombic boxes, support below half each side, contiguous IDs and one MPI rank are required. The constant reservoir is a numerical term, not an atom; all points outside support yield zero gradient.

Qualification must verify summed forces and energies, anchor reaction, finite differences, periodicity/identity invariance, zero-biased controls and synthetic NVE before model dynamics. Two individually qualified callbacks do not by themselves qualify their combined engine integration. Biased local positions do not measure whole-object translation, equilibrium free energies or unbiased rates.
