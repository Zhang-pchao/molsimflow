# Contact-state analysis: reusable modules and boundary contracts

This map identifies existing helpers for a source-aware contact-state adapter.
It does not certify a force field, validate a physical cutoff, or establish
independence of frames. Keep material atom maps, source lineage, time offsets,
cutoffs and output locations in project configuration.

| Need | Existing public module | Actual capability and boundary |
| --- | --- | --- |
| Existing-table inventory | `postprocess.contact_state_inventory` | Exact source/epoch/time joins, partition closure and censored contact episodes. Historical contact rows have no raw hash; provenance remains a legacy index match. No new coordinate observation. |
| Trajectory input | `io.lammps_dump` | Record streaming and bounded reads; preserve the raw box header and atom fields. The project must reject unsupported cell/boundary contracts before geometric operations. A bounded successful read does not validate unread tails. |
| O-H assignment | `postprocess.species_assignment` | Nearest oxygen with explicit per-axis periodicity and cutoff; local indices and counts. Oxygen-only assignment does not implement competition with carbon. Geometric counts are not electronic charges or pKa. |
| Surface proton changes | `postprocess.surface_proton_transfer` | Public O/C-competitive H assignment already exists here but assumes PPP; event/occupancy helpers distinguish identity exchange from count changes. Atom identity and actual site proton occupancy must remain distinct. |
| Gas pair centers / largest cluster | `postprocess.nanobubble_attachment` | Existing helpers assume periodicity in all three axes. Consecutive coordinate pairs are assumed to be intact diatomics. Do not apply directly to PPF, incomplete gas selections or a gas of different molecular identity. |
| Gas connectivity tables | `postprocess.gas_connectivity` | PPP molecular proximity edges and aggregate tables. A molecular cluster is not necessarily an isolated spherical cap; winding must be qualified separately. |
| Contact graph tables | `postprocess.contact_graph` | Graph summaries from declared edge rows; graph connectivity does not establish force transmission or mechanical pinning. |
| Molecular footprint support | `postprocess.surface_contact_process` | XY periodic capsule membership, union/sample bookkeeping and source-aware episodes. No Z assumption in XY helper; supply/contact definitions remain caller responsibilities. |
| Identity-aware reconstruction | `postprocess.surface_reconstruction` | Existing frozen molecular outcome reducer, inclusive dwell confirmation, connected-column departure and staged transition ledger. Thresholds belong to its documented historical contract; they are not universal physical constants. |
| Water identity turnover | `postprocess.surface_water_renewal` | Molecular-cohort histories, site restoration, periodic-grid components and support runs. Endpoint recurrence is not continuous molecular retention. |
| State persistence | `postprocess.surface_state_persistence` | Sampled core-state confirmation. Preserve cadence, unknown states, source seams and observation limits. |
| Excursions / molecular kinetics | `postprocess.surface_excursions`, `postprocess.molecular_kinetics` | Reusable event reduction from already aligned observations; molecular observations and episodes are not independent material replicas. |
| First passage / inventory | `postprocess.core_to_core_access` | Explicit initial/late starts, arrival versus right censoring, return excursions and inventory closure on supplied arrays. Region definitions, time mapping and seam segmentation are external. No starting molecule means an empty risk set for that region/window, not globally absent transport. |
| Observer reference | `postprocess.scaffold_reference` | Immutable explicit Si/bridge-O IDs and bounded translation registration, PPF orthorhombic only. Requires externally established scaffold identity and rejects deformation/wrapping ambiguities. Not interchangeable with a historical all-heavy-atom PPP reference. |
| Preparation / intervention contract | `postprocess.reconstruction_contracts` | Reusable qualification contracts; their success verifies declared preparation criteria, not chemical accuracy or equilibrium. |

## Thin adapter rules

1. Import package modules directly. Do not insert historical analysis directories
   into `sys.path`, and do not import legacy scripts from package code.
2. Freeze an explicit registry with atom/molecule IDs, physical time, boundary
   flags, chemistry definition, source segment and parent identity. A restart
   seam or changed ensemble cannot be hidden by consecutive row numbering.
3. Reuse cached observables only after checking that their definition, source
   identity and time selection match the task. Preserve a reference to the
   original cache; do not regenerate or rename an old output as a new result.
4. Keep observation reduction separate from plotting. Render from fixed result
   tables; changing a figure must not implicitly rerun trajectory decoding.
5. Set statistical units outside the reducers. Blocks, episodes and multiple
   descendants from one parent do not create independent preparations.
6. Freeze the complete source content used for a run, including untracked
   modules. A Git commit alone is insufficient for a dirty/untracked worktree.
7. Preserve failed test and execution receipts. Do not change a scientific
   threshold to make a result positive or to repair a software regression.

## Verification scope

A focused test selection should exercise source/cadence interruption, unknown
states, inclusive dwell endpoints, false replacement, identity exchange,
periodic/nonperiodic distance, incomplete pairing and chemical competition.
Synthetic tests establish software behavior. Source parity and field checks on
actual project data establish adapter behavior. Neither establishes physical
mechanism, equilibrium, or model accuracy.
