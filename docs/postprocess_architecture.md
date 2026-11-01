# Postprocessing architecture and reuse

Package boundaries define scientific ownership. Branches track integration and
release state: a short-lived feature joins `devel`, and a qualified release joins
`main`. A module-specific branch is not a permanent alternative package.

## Find an owner before implementing

The generated [module catalog](module_catalog.md) lists all package modules and
public definitions without loading optional dependencies. Regenerate it after
adding or moving an API; CI checks that it matches the source.

| Task | Starting point | More detail |
| --- | --- | --- |
| Trajectory/structure formats and hashes | `molsimflow.io` | `shared_trajectory_utilities.md` |
| Structure preparation and molecular relocation | `molsimflow.structure` | CLI `structure --help` and owning synthetic tests |
| PLUMED configuration | `molsimflow.plumed` | CLI `plumed --help` and explicit CV contracts |
| PIMD estimators, alignment, path weights and reports | `postprocess.pimd` | `pimd_architecture.md`, `pimd_reweight.md`, `pimd_optional_features.md` |
| Fixed-umbrella free energies | `postprocess.umbrella_wham` | `umbrella_wham.md` |
| Contact geometry, identities, transport and front observables | Existing `postprocess.interfacial_*` owners | `interfacial_architecture.md` |
| Sampled contact states, turnover, residence and reconstruction | `surface_contact_*`, `surface_water_renewal`, `surface_state_persistence` | `contact_state_module_map.md`; endpoint recurrence, sampled survival and historical outcome thresholds are distinct |
| Local geometry, registered references and bounded placement | `local_density_front`, `curved_density_front`, `scaffold_reference`, `molecular_placement` | Owning geometry contracts; material-specific reference adapters remain explicit |
| Driven nanobubble observers | `nanobubble_flow_response`, `nanobubble_local_flow`, `nanobubble_edge_*`, `junction_release` | Explicit contracts and original material identities; temporal causal filters are not causal identification |
| Offline OPES replay and path coverage | `opes_replay`, `cv_coverage`, `glycine_states` | `opes_replay.md`; keep recorded-kernel replay separate from frozen-state evaluation and geometric state candidates |
| Frozen input and archive primitives | `io.live_snapshot`, `io.archive_lifecycle` | `live_junction_species.md`, `archive_lifecycle.md`; no implicit cleanup or host discovery |
| Constant-force and event-table analysis | `postprocess.constant_force_*`, `contact_graph` | Owning docs/tests; distinguish a recorded intervention from an equilibrium estimator |
| Plotting and media | `molsimflow.plotting`, `molsimflow.media` | Keep numerical estimators outside presentation adapters |
| CLI/configuration/workflow composition | `cli.py`, `config`, `workflows` | Parse and compose; numerical kernels belong in their domain |

Search relevant modules and their callers, then compare units, atom/frame
identity, boundary flags, sample weights, missing observations and output schema.
Extend the matching owner; add a focused module only when no existing owner fits.
Do not infer equivalence from a similar name or hide a scientific difference in
a generic utility. Stable cross-domain reuse must be demonstrated before lifting
code into a general layer.

## Input to output responsibilities

1. Read and validate explicit input formats; retain source identity.
2. Align atom/frame identities, units, temporal seams and coordinate charts.
3. Apply the owning estimator or numerical kernel with its declared support.
4. Write reusable tables/receipts, then render figures from those fixed outputs.
5. Keep material configuration, source selection, scheduler scripts and scientific
   conclusions in the consuming project.

PIMD keeps estimator, observable, bias mode and weight provider separate. One
complete path has one reweighting weight; beads are not independent samples.
Two-dimensional observations preserve same-frame/same-bead pairing. These rules
must not be replaced by the sampling contract of a different estimator such as
fixed-umbrella WHAM.

Source repositories contain reusable code, synthetic tests and portable docs.
Models, real trajectories, generated results and private provenance belong in
the consuming project. Build/install artifacts live outside source. Scientific
consumers should select an exact qualified package; editable installations are
for explicit development environments.

## Working across development hosts

Record the active writer, worktree, branch and base commit for each task. Parallel
tasks use separate branches and agreed scopes. Stop the old writer before handing
off the exact commit, test receipt and remaining uncommitted work. Share Git
history rather than copying working directories. Preserve dirty files and never
rewrite shared history to manufacture agreement.

Build once from the qualified release commit, record the package hash and Python
environment, and install that artifact into new versioned prefixes. Verify import
origin outside source checkouts and check editable/PYTHONPATH shadowing in actual
consumers. Preserve the previous package and entrypoint for rollback.

Retire a feature branch only after its work is integrated or explicitly retained
elsewhere. Removing a worktree additionally requires preserved untracked/ignored
payloads, no active writer/reader, and no remaining consumer path dependency.
Branch names themselves consume negligible space; worktree files and environments
account for storage. See [CONTRIBUTING.md](../CONTRIBUTING.md) for release checks.

Constant-force kinematics accepts one common start/end analysis window. Ion
distribution sampling accepts an explicit step stride and origin. These are
selection contracts, not changes to estimator normalization. Nanobubble CLI
handlers live in `postprocess.nanobubble_cli`; material-specific defaults remain
visible and must be reviewed by each consuming project.
