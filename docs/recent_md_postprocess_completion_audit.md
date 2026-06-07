# Recent md_postprocess Completion Audit

Audit date: 2026-06-07.

This audit closes the recent-update inventory from the legacy `md_postprocess`
tree.  The goal was not to copy every script.  The package keeps reusable,
path-explicit analysis code and leaves publication figures, scheduler
submission, and private case discovery out of the public API.

## Migrated Reusable Code

- PLUMED/CV diagnostics: `molsimflow postprocess plumed-cv-diagnostics`.
- Silica surface summaries: `molsimflow postprocess silica-surface`.
- Particle flotation summaries: `molsimflow postprocess particle-flotation`.
- 2D FES grid processing: `molsimflow postprocess fes2d-grid`.
- 2D FES batch manifest planning: `molsimflow postprocess fes2d-batch-manifest`.
- FES Delta-F convergence: `molsimflow postprocess fes-convergence`.
- Cumulative FES reweight command planning:
  `molsimflow postprocess fes-cumulative-reweight-manifest`.
- Gas contact/radius-sum summaries:
  `molsimflow postprocess gas-contact-summary`.
- Bridge electrostatics/EDL/coupling proxies:
  `molsimflow postprocess bridge-electrostatics`.

## Closed Without Direct Migration

These files are intentionally not copied into `src/molsimflow` as standalone
public modules:

- `merge_gas_connectivity_summary`: fixed 2 x 5 manuscript aggregation and
  figure layer.  Reusable frame metrics and radius-sum summaries are already in
  `molsimflow.postprocess.gas_connectivity`.
- `bridge_electrostatics_edl_from_traces`: case-discovery and trace-adapter
  wrapper.  The reusable prepared-table electrostatics logic is now in
  `molsimflow.postprocess.bridge_electrostatics`.
- `analyze_hcl_event_aligned_sequence_20260604`: HCl-only audit package with
  fixed absolute inputs and publication figures.  Generic event-aligned profile
  and coupling utilities already live in `molsimflow.postprocess.events` and
  `molsimflow.postprocess.coupling`.
- `analyze_same_gap_mechanism_audit_20260604`: fixed case order, fixed inputs,
  and manuscript figure package.  Generic gap-binned summaries are available in
  the migrated bridge/contact/gas modules, and cross-case joins belong in
  `molsimflow.postprocess.case_comparison`.
- Recent plotting/render scripts dated 2026-05-29 through 2026-06-06:
  publication layer only.  Reuse column schemas and transformations manually
  if a future general plotting API is needed.
- Reweight execution and scheduler wrappers: the package now generates explicit
  command manifests; actual driver execution, file deletion, copying legacy
  scripts, and `sbatch` submission stay as workflow scripts or templates.

## Remaining Future Work

No additional reusable code is required from this recent-update inventory.
Future migration should be driven by a repeated need for one of the closed
adapters above, not by copying the legacy scripts verbatim.
