# Changelog

## 0.2.0

- Integrated fixed-umbrella WHAM, interfacial geometry, renewal, transport,
  front observables and sampled contact-state utilities into one package.
- Retained distinct historical material adapters and periodic front APIs;
  documented temporal support, PBC and descriptive-inference boundaries.
- Added selected and synchronized LAMMPS dump readers using the existing parser,
  explicit common analysis windows and ion-sampling cadence/origin controls.
- Added junction and nanobubble observers, offline OPES replay and portable
  trajectory/archive utilities with synthetic regression coverage.
- Added a generated module/API catalog, ownership documentation and CI freshness
  checks to guide reuse. New nanobubble CLI handlers have a dedicated adapter.
- Preserved PIMD estimator/observable definitions, complete-path weights, public
  facades, contracts and output schemas. Existing commands retain their defaults.

Install this version into a fresh environment and qualify consuming workflows
before adoption. Keep the previous environment for rollback. The new observers
are reusable implementations with explicit contracts; tests do not establish
material transferability, convergence or causal identification.

## 0.1.1

- Made the PIMD probability-mean and free-energy-mean bead FES estimators
  explicit selectable primaries, added schema-3 pairwise support and canonical
  output fields, and stabilized weighted histograms in log space.
- Added contract-driven centroid, bead-mean, and shared bead-density PIMD
  quantum-FES reweighting; fixed-bias, quasi-static OPES, and precomputed frame
  weights; generic 1D/2D core reports; restart alignment; and
  direct/conditional estimator regression checks.
- Added generic CP2K energy/force parsing and atomistic model-validation helpers.
- Added table-driven Eyring barrier sensitivity and conditional two-channel competition.
- Added streaming LAMMPS dump preparation and configurable reference-layer alignment.
- Added optional image-sequence cropping, annotation, and MP4 generation.
- Documented the `devel` integration and validated `main` branch workflow.

Case-specific reports, scientific conclusions, private paths, generated data,
and legacy migration snapshots remain outside the public package.
