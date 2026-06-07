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
- `gas_connectivity_validation` / `radius_sum_contact_validation`: reusable gas contact-graph metrics and radius-sum summary tables migrated into `molsimflow.postprocess.gas_connectivity` with CLI `molsimflow postprocess gas-contact-summary`.
- `bridge_electrostatics_edl` / `bridge_electrostatic_coupling`: reusable bridge-axis charge profile, Poisson proxy, gap-binned EDL summaries, and prepared-trace charge-coupling descriptors migrated into `molsimflow.postprocess.bridge_electrostatics` with CLI `molsimflow postprocess bridge-electrostatics`; hardcoded case discovery, legacy-profile lookup, scheduler submission, and publication plotting remain out of the public API.

## Recent Reusable Candidates Still Pending

- Related 2D FES batch setup and reweight manifests: migrate only the path-explicit batch/table logic; keep scheduler submission as a template or workflow adapter.
- Cumulative reweight generation scripts: migrate only reusable path-explicit setup logic if needed; keep scheduler submission and hardcoded case lists out of package defaults.
- `merge_gas_connectivity_summary`: keep 2x5 aggregation, figure styling, and manuscript-language summary as a case-specific workflow adapter unless a broader multi-case manifest need appears.
- `bridge_electrostatics_edl_from_traces`: optional future workflow adapter only.  The core prepared-table electrostatics logic is migrated; the trace adapter should remain path-explicit if it is added later.
- `analyze_hcl_event_aligned_sequence` and same-gap mechanism audit scripts: compare against existing `events`, `coupling`, and `case_comparison` modules before adding new APIs.

## Mostly Case-Specific Or Publication Layers

- Figure scripts dated 2026-05-29 through 2026-06-06 should not be copied directly.  Reuse only column conventions, manifest schemas, or plotting transformations that generalize across projects.
- Batch submission wrappers with hardcoded case manifests should become docs or scheduler templates only after their reusable inputs are explicit.
