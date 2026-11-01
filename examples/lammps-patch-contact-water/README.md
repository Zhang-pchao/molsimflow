# Patch contact / water coordinate prototype

This single-rank orthorhombic C++ prototype implements a compact quintic switch,
molecular-density-gated gas proximity, compact any-solvent-O site hydration,
a Gaussian-times-compact solvent preparation field, and fixed-parent-weight gas
height relative to a separate surface-O plane (circular in periodic z).
The height gradient includes the reference-plane reaction and no dynamic
molecular-density derivative. Contact includes its full density chain rule.
Exact-support pruning and rebuilt water cell bins leave the formulas unchanged.

This is a coordinate and native run-0 harness, **not a released MD production
operator**. The installed LAMMPS library is opened with pair_style zero and
run 0; no physical potential is evaluated. Mass 1 in that harness is irrelevant
to zero-step force testing and is not a production mass map. The fixture type
contract is O=2, N=3, terminal C=7; use a qualified explicit map for other models.
No cell derivative / bias virial is provided. Do not use this prototype for NPT,
pressure, friction, pKa or chemically unique pair-force attribution.

`replay.cpp`: config frame output C_target W_drive_target h_target.
`native_zero.cpp`: same arguments; link the matching LAMMPS C library/MPI.
`edge_checks.cpp`: compact-switch boundaries, zero support and saturated gate.
`benchmark.cpp`: config frame bounded_repeats; repeated existing geometry only.

Frame format: atom count; three lo/hi/periodic lines; id/type/x/y/z per atom.
Config: atom count; four count+ID-list lines (contact terminals, hydration O,
reference plane O, solvent O); gas pair count; pair IDs and fixed weight per pair.
IDs must be the complete contiguous inventory, input order may vary. All original
solvent O is eligible regardless of proton ownership. Molecules are original N2
pairs admitted at 0.85-1.50 Angstrom. No original water or H is restrained.

The native harness uses metal units, complete conservative atomic forces,
fixed-center diagnostic springs and hard stop guards; it verifies exact zero
added force/energy after unfix. A production operator additionally requires
prospective time schedule, protocol/mechanical work, impulse, seam ownership,
zero-strength fast path with decimated observers, full current-chemistry and
material-domain guards, verified physical inputs and runtime scaling.

## Finite-step engineering ledger (Q0b)

`protocol_ledger.hpp` implements an absolute-step quintic schedule, termwise
protocol/mechanical work and atomwise impulse, duplicate-step rejection, hard
caps and restart sidecars. For U=lambda*V with fixed centers and springs:
protocol increment is average(V)*delta(lambda); mechanical work on atoms is
average(F) dot unwrapped displacement; impulse is average(F)*dt. Fixed box,
stable-ID ordering and immutable bias parameters are required. Callback velocity
is not an endpoint velocity under native velocity Verlet. No per-pair physical
force attribution or free-energy estimator is implemented.

`advance_zero` skips CV/storage work only for inactive intervals. The driver must
continue cheap inventory/finite-state guards, evaluate the last withdrawal
interval, and supply `refresh_zero` at inactive checkpoints or before activation.
The refresh may only replace a zero-strength endpoint and never adds work.
Sample context must hash actual model, masses, box, IDs, parent, target parameters,
schedule, runtime and source; the synthetic test context binds its own fixture.
`restart_bundle.hpp` uses OpenSSL EVP SHA256 to bind native restart and sidecar;
the bundle manifest is published last and validated before reading either state.
Changed code/context/dt/step or payloads reject; no automatic seam repair.

`native_ledger.cpp` is a two-atom pair-zero integration test, not a production
SiO2 driver. It uses unequal illustrative masses, a shortened schedule and the
matching LAMMPS library. `ledger_checks.cpp` tests negative contracts and schedule
duty counts; `bundle_checks.cpp` tests checkpoint identity/corruption.
`patch_ledger_replay.cpp` applies prescribed small coordinate deformations to
existing complete frames and verifies work refinement. These are not new MD
samples. Build native and bundle tools with OpenSSL (-lcrypto); standalone ledger
and coordinate replay need only C++17. The worktree is not automatically committed.

The combined full-system GPU callback, current-chemistry/material guards, native
periodic image integration, thermostat state/force bookkeeping and full production
checkpoint bundle remain unqualified. Single-rank CPU synthetic restart does not
certify Langevin random-state or GPU restart equivalence. Keep production release
false until a project-specific end-to-end test and parent admission pass.
