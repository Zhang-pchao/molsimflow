# molsimflow Migration Plan

## Scope

Target repository:

- `molsimflow`

Legacy source inputs:

- legacy MD post-processing package directory;
- legacy double-bubble PLUMED generators;
- legacy double-bubble TiO2/NaOH structure-preparation directories.

Exact private source paths are kept outside the public repository in the local
migration note written near the original analysis directory.

## Repository Name

Chosen initial name: `molsimflow`.

Reasoning:

- broader than `md_postprocess`, because the codebase includes preprocessing,
  structure building, PLUMED generation, post-processing, plotting, and
  visualization;
- less tied to one bubble/TiO2 project than a bubble-specific name;
- currently no package with this name was found through the PyPI JSON endpoint.

## Migration Phases

### Phase 0: Inventory and Quarantine

- Copy source material into `legacy_sources/` for local migration reference.
- Keep `legacy_sources/` ignored by Git until files are sanitized.
- Record source paths, file counts, and hardcoded-path risks in docs.
- Strip generated outputs and backup files from local migration snapshots.
- Run `python scripts/audit_public_repo.py` before committing tracked files.

### Phase 1: Public Package Skeleton

- Create `src/molsimflow` with stable namespaces:
  - `io`: file format readers, writers, and converters;
  - `structure`: geometry and structure-preparation utilities;
  - `plumed`: PLUMED input generators;
  - `postprocess`: migrated MD analysis workflows;
  - future `plotting` or `visualization` namespace if repeated plot code is generalized.
- Add a single CLI entry point: `molsimflow`.
- Add tests for migrated utility functions.
- Keep extension-oriented namespace boundaries so other systems can reuse
  `io`, `structure`, `plumed`, and `postprocess` without TiO2/bubble-specific imports.

### Phase 2: Preprocessing and PLUMED Tools

- Generalize the duplicate PLUMED scripts into one argument-driven module.
- Replace hardcoded input/output paths with required CLI arguments.
- Generalize extended XYZ PBC injection and LAMMPS data conversion.
- Split the TiO2 double-bubble slab builder into configuration, geometry,
  molecule-count, PACKMOL-writing, and ASE-writing units.
- Add a generic prebuilt-interface adapter so non-TiO2 systems can reuse the
  same double-bubble solvent/gas/PACKMOL planning logic.
- Add an explicit optional Packmol runner while keeping file generation as the
  default behavior.
- Add a generic scheduler template that can run structure generation, optional
  Packmol execution, extended-XYZ PBC injection, and LAMMPS-data conversion.
- Run real generated-case smoke checks before migrating post-processing code.
- Begin post-processing migration with centroids and bubble-surface distance.

### Phase 3: MD Post-Processing Migration

- Keep the existing `md_postprocess` source as the main raw source of truth.
- Migrate by workflow family, not by copying all files at once:
  - centroids and bubble surface distance: migrated first pass, with explicit
    trajectory/input arguments and top-level CLI forwarding;
  - ion species and ion distribution: core APIs and CLI commands migrated for
    species classification, classified XYZ/statistics output, and relative
    ion-z distribution TSV generation;
  - bridge-water and bridge-ion descriptors: first table-oriented core migrated
    for bridge-cylinder geometry, water density proxies, ion occupancy/charge,
    and gap-bin/window summaries;
  - FES and barrier analysis: first table-oriented core migrated for 1D FES
    loading, reference shifting, smoothing, zeroing, and configurable barrier
    window summaries;
  - coalescence-state assignment: table-oriented core migrated for COLVAR,
    optional cluster-counter, and optional bubble-evolution inputs;
  - case-comparison descriptor synthesis: first table-oriented core migrated
    for case manifests, descriptor scorecards, pair deltas, and descriptor
    correlations;
  - plotting and publication-figure assembly: first generic CSV-driven plotting
    core migrated for line, scatter, and heatmap figures;
  - case-comparison reports.
- For each workflow, remove case-specific defaults, expose inputs as arguments,
  add a small test or smoke fixture, and write a short usage example.

### Phase 4: Configuration and Scheduler Templates

- Move datasets, queue names, module loads, and conda environments into external
  config files or documented examples.
- Keep SLURM/PBS templates generic and parameterized.
- Do not import scheduler assumptions into Python library code.
- Current status:
  - added a generic SLURM preprocessing template for structure generation,
    optional Packmol execution, and file conversion;
  - added a lightweight INI config helper that can summarize workflow config,
    resolve paths relative to the config file, and print shell exports for
    scheduler templates;
  - added `templates/config/workflow.example.ini` with placeholder-only public
    settings.

### Workflow Namespaces

- Add project-specific orchestration under `src/molsimflow/workflows` only when
  a feature is too tied to a workflow to belong in a generic namespace.
- Current specialized namespace:
  - `molsimflow.workflows.double_bubble_merge` records the recommended stage
    ordering for the double-bubble coalescence system while keeping algorithms
    in generic modules.

### Phase 5: Public Release Readiness

- Run hardcoded path scans before committing.
- Remove or rewrite private absolute paths in docs and examples.
- Add installation instructions, minimal examples, and reproducible tests.
- Add a license only after the intended public license is decided.
- Current status:
  - added README installation and verification commands;
  - added `docs/release_readiness.md` with the first public-push gate;
  - kept license selection as an explicit unresolved user decision.

## First-Pass Deliverables

- `pyproject.toml` with installable package metadata.
- `README.md` with current scope and rules.
- `molsimflow structure add-extxyz-pbc`.
- `molsimflow structure extxyz-to-lammps-data`.
- `molsimflow structure equal-volume-radius`.
- `molsimflow plumed double-bubble`.
- Unit tests for the small reusable utilities.
