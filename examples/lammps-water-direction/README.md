# Water donor direction intervention

The Python reference is `molsimflow.postprocess.water_direction_bias`. This
example uses one MPI rank, orthorhombic periodic cells, and the fixed atom order
site O, water O, H1, H2, anchor Si, anchor O, N1, N2. The two H atoms enter a
permutation-symmetric smooth maximum of their donor cosines. The gradient acts
on the water H/O and site O; nine distance springs also act on the two remaining
surface anchors and both N atoms. Total construction force and torque vanish.
The water remains flexible. This is not a rigid-body rotational constraint, and
molecular topology must be checked throughout construction.

`reference_cli.cpp` compiles without LAMMPS and evaluates one configuration.
`water_direction_driver.cpp` requires the exact matching LAMMPS/MPI library;
create `fix CV all external pf/callback 1 1` with suffix off in setup, restore
the engine suffix, and use `fix_modify CV energy yes virial yes`. Do not run the
engine until after callback registration. Native compilation/engine validation
is separate from standalone kernel validation. Never infer native readiness
from a passing reference CLI.

Configuration is whitespace separated: atom_count, arm (0 aligned,1 away,2
sham), eight IDs, initial_q, per-atom force_cap, absolute_protocol_work_cap,
time_step_ps, nine reference distances (water O,N1,N2 crossed with site O,Si,O).
The native wrapper currently requires type indices H=1,O=2,N=3,C=7,Si=8;
this is validated, not inferred from the material name. Other mappings require
an explicit wrapper change and qualification.

The frozen schedule ramps common and orientation springs during 0--0.5 ps,
changes the away target to 0.5 during 0.5--1.5 ps, holds to 2 ps and sets ALL
construction forces to exactly zero at 2 ps. It retains the callback for the
zero-force/work receipt. Background drive, tether and thermostat remain the
caller's explicit responsibility. `run 0` plus normal consecutive steps is
supported. Restart/reseeding/skipped callback histories are not supported.

The work ledger records every step and all 24 construction-force components.
Protocol work uses U(x_n,parameters_n)-U(x_n,parameters_previous), including
-U_before at force-off. It is a discrete parameter-work convention, not physical
force work or a thermostat ledger. Native energy closure and time-step behavior
must still be checked. Guards stop on preparation topology changes, excessive
bias force or absolute protocol work; they never clamp forces silently.
After release, topology changes are observed offline, not forbidden by a bias.
This example establishes no chemistry, contact probability, free energy, pKa,
friction, or causal mechanism by itself.
