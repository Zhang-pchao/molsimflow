# Interfacial analysis: ownership and scientific contracts

These modules operate on caller-supplied coordinates, identities, masks and
observations. They do not discover a cluster, material, trajectory or simulation
boundary condition. Install the `analysis` extra for SciPy-dependent kernels.

## Select an existing owner

| Need | Owner | Contract to check before reuse |
| --- | --- | --- |
| Probe accessibility and local cover | `surface_exposure` | Probe direction, per-axis PBC, declared atom/site geometry and finite support |
| Conformal contact chart | `interfacial_contact_geometry` | Orthorhombic periodic chart, explicit ownership, holes/disconnected patches, local height and normal conventions |
| Contact fields | `interfacial_contact_fields` | Native cadence, epochs, smoothing scale and periodic grid; gaps cannot become adjacent samples |
| Saved contact events | `interfacial_contact_events` | Binary/unknown states, persistent runs, transition brackets and censored endpoints |
| Contact and particle budgets | `interfacial_contact_renewal` | Same ordered IDs at both endpoints; distinguish motion, geometry and identity exchange |
| Hydrogen-bond pair edges | `interfacial_water_hbond.hydrogen_bond_pair_edges` | Directed donor geometry, pair deduplication and parity with count APIs |
| Explicit graph nodes and anchor paths | `contact_graph` | Include isolates; bounded paths and deterministic identity; connectivity is not a force network |
| Gas motion and shape | `interfacial_gas_kinematics` | PBC winding, unwrapped coordinates, membership, gaps and reference motion |
| Open-population centroid | `interfacial_gas_transport` | Exact finite-step identity budget, not continuous flux or phase conversion |
| State matching | `interfacial_state_matching` | Physical feature units, calipers, no replacement and unsupported populations; no causal identification |
| Marked-site geometry | `interfacial_structure_descriptors` | Caller-defined marks, coordinate chart, directions and edge support |
| Velocity/edge summaries | `interfacial_velocity_profile`, `interfacial_contact_coordination` | Caller-defined height bins and lane/cadence contracts |

Trajectory readers, minimum-image helpers and reference frames remain in their
existing owners; see [shared_trajectory_utilities.md](shared_trajectory_utilities.md).
Do not merge routines solely because their names mention contacts, components
or distance. A grid component, atom graph and periodic winding test have
different contracts.

## Front APIs: canonical choices and preserved legacy behavior

Use `periodic_front_pattern` for explicit periodic-axis neighbor/connectivity
calculations, full-surface tip isolation, synchronous tip/bulk gaps and
complete-window advance fractions. Pass periodic flags from the simulation
contract. Material layout and a chemical seam do not determine periodicity.
An intentionally open observation chart must be documented as an analysis
choice, distinct from the simulation's physical boundary conditions.

`interfacial_front_pattern` remains a material-domain adapter for its historical
Si-parent/OH/CH3 column map and open-chart statistics. Its `bare` label means an
OH-free CH3-terminated anchor, not a universally bare or unsaturated surface.
The adapter is available for reproducing existing analyses. New materials should
supply an explicit, independently documented mapping instead of reusing these
labels by analogy.

| Quantity | Historical API | Preferred explicit API | Compatibility boundary |
| --- | --- | --- | --- |
| Selected-site connectivity | `usable_anchor_connectivity` | `sublattice_connectivity` | Open versus periodic chart; raw coordinate spans are not winding proofs |
| Tip isolation | `tip_isolation` | `tip_isolation_full_surface` | Destination-only neighbors versus every contacted site |
| Tip/bulk comparison | Caller combinations of extrema | `synchronous_tip_bulk_gap` | Both values must come from the same frame; returned final gap uses the last valid pair |
| Forward progress | `column_advance_hazard` | `complete_window_advance_fraction` | Historical selected-column denominator differs from all complete windows, including failed attempts |
| Contact selectivity | Material adapter ratios | `selectivity_with_support` | Weighted mean of supplied ratios is not a pooled numerator/denominator ratio |

There is no automatic alias or replacement between these definitions. Old
imports and result keys remain callable. Record the selected API, chart, units,
time window and support in consuming-project provenance. Migrate a scientific
analysis explicitly, comparing its output and interpretation before adoption.

`complete_window_advance_fraction` assumes one uninterrupted uniformly sampled
segment. Split native gaps and restarts before calling it. A column index of
`-1` means an observed absence of a front, not an unread/missing frame. Its
overlapping-window fraction is descriptive, not an independent-event hazard.

`front_closure` describes within-column associations and supplied permutation
controls. It does not establish that chemistry causes motion, or turn correlated
frames from one trajectory into independent material replicas.

## Compatibility and extension policy

Keep current public import paths during this integration. Add narrowly scoped
validation and synthetic counterexamples where an invalid input could silently
produce a plausible result. Numerical definitions, defaults, support masks and
output schemas require explicit migration decisions if changed.

A future `postprocess/interfacial/` package may collect these owners with root
compatibility facades. A directory move is a separate behavior-preserving change,
not a prerequisite for reusing today's functions. Avoid a generic `common.py`;
lift a primitive only after two independent domains share its stable semantics.
