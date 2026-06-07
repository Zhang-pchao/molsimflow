# Recent md_postprocess Updates

Inventory date: 2026-06-07.

This note tracks recently modified legacy `md_postprocess` files that were not
part of the first migration snapshot.  It is intentionally path-neutral so the
public repository does not record private absolute paths.

## Migrated In This Batch

- `plumed_cv_diagnostics`: engineered as `molsimflow.postprocess.plumed_cv_diagnostics` with CLI `molsimflow postprocess plumed-cv-diagnostics`.
- `silica_surface`: engineered as `molsimflow.postprocess.silica_surface` with CLI `molsimflow postprocess silica-surface`.
- `particle_flotation`: engineered as `molsimflow.postprocess.particle_flotation` with CLI `molsimflow postprocess particle-flotation`.
- `fes2d_single`: first-pass regular-grid processing migrated into `molsimflow.postprocess.fes_analysis` with CLI `molsimflow postprocess fes2d-grid`; scheduler and case-manifest wrappers remain out of the public API.
- `fes_robustness_deltaf`: reusable Delta-F convergence tables migrated into `molsimflow.postprocess.fes_analysis` with CLI `molsimflow postprocess fes-convergence`; fixed system lists, figure styling, and manuscript-language judgments remain out of the public API.

## Recent Reusable Candidates Still Pending

- Related 2D FES batch setup and reweight manifests: migrate only the path-explicit batch/table logic; keep scheduler submission as a template or workflow adapter.
- Cumulative reweight generation scripts: migrate only reusable path-explicit setup logic if needed; keep scheduler submission and hardcoded case lists out of package defaults.
- `gas_connectivity_validation`, `radius_sum_contact_validation`, and `merge_gas_connectivity_summary`: extract reusable gas-cluster connectivity checks and summary tables; avoid publication-specific plot layout defaults.
- `bridge_electrostatics_edl`, `bridge_electrostatics_edl_from_traces`, and `bridge_electrostatic_coupling`: extract charge-profile, EDL proxy, and prepared-trace coupling table logic; keep case discovery and legacy-profile lookup as optional workflow adapters.
- `analyze_hcl_event_aligned_sequence` and same-gap mechanism audit scripts: compare against existing `events`, `coupling`, and `case_comparison` modules before adding new APIs.

## Mostly Case-Specific Or Publication Layers

- Figure scripts dated 2026-05-29 through 2026-06-06 should not be copied directly.  Reuse only column conventions, manifest schemas, or plotting transformations that generalize across projects.
- Batch submission wrappers with hardcoded case manifests should become docs or scheduler templates only after their reusable inputs are explicit.
