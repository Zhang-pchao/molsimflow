# Phase 2 Structure API

Phase 2 starts by converting the legacy one-file TiO2 double-bubble builder into
separate reusable layers.

## Implemented Boundary

- `molsimflow.structure.regions`
  - Basic region primitives such as `BoxRegion`, `SphereRegion`, and
    `CylinderRegion`.
- `molsimflow.structure.packmol`
  - PACKMOL block and input rendering helpers.
- `molsimflow.structure.double_bubble_slab`
  - A configurable double-bubble slab planning and build API.
  - `DoubleBubbleSlabConfig` holds solvent, gas, pH, and PACKMOL settings.
  - `SlabBuildConfig` holds ASE slab construction settings.
  - `PrebuiltInterfaceConfig` holds settings for a user-provided interface
    structure.
  - `plan_double_bubble_slab` is testable without ASE.
  - `build_tio2_double_bubble_inputs` is the TiO2-specific ASE adapter.
  - `build_prebuilt_double_bubble_inputs` is the generic prebuilt-interface
    adapter.

## CLI

Generic prebuilt interface:

```bash
molsimflow structure slab-double-bubble \
  --interface-structure /path/to/interface.xyz \
  --molecule-dir /path/to/molecule_templates \
  --output-dir /path/to/generated_case \
  --gas-radii 19 14 \
  --bubble-spacing 50 \
  --bottom-layer-index 0 \
  --target-ph 13.4
```

Run Packmol as an explicit opt-in step:

```bash
molsimflow structure slab-double-bubble \
  --interface-structure /path/to/interface.xyz \
  --molecule-dir /path/to/molecule_templates \
  --output-dir /path/to/generated_case \
  --gas-radii 19 14 \
  --bubble-spacing 50 \
  --run-packmol \
  --packmol-command packmol
```

TiO2 surface construction:

```bash
molsimflow structure tio2-double-bubble \
  --bulk-structure /path/to/bulk.cif \
  --molecule-dir /path/to/molecule_templates \
  --output-dir /path/to/generated_case \
  --gas-radii 19 14 \
  --bubble-spacing 50 \
  --target-ph 13.4
```

For the equal-volume control pattern:

```bash
molsimflow structure tio2-double-bubble \
  --bulk-structure /path/to/bulk.cif \
  --molecule-dir /path/to/molecule_templates \
  --output-dir /path/to/generated_case \
  --gas-radii 16.87 16.87 \
  --bubble-spacing 50 \
  --water-height-radius 19 \
  --bubble-center-z-radius 19 \
  --target-ph 13.4
```

`--molecule-dir` expects `H2O.xyz`, `N2.xyz`, `Na.xyz`, and `OH-.xyz`.  The
templates can also be passed individually with `--water-xyz`, `--n2-xyz`,
`--cation-xyz`, and `--anion-xyz`.

## Design Notes

- PACKMOL input generation is independent of ASE.
- Packmol execution is optional and explicit.  The builders still default to
  writing files only.
- TiO2-specific surface construction lives in the TiO2 adapter, not in the
  region or PACKMOL helpers.
- The generic prebuilt-interface adapter reads a prepared interface structure
  and does not call ASE `surface()`.
- Generated structures and PACKMOL outputs are user artifacts and should not be
  committed to the package repository.
- Future non-TiO2 systems should add small adapters that call the same plan and
  PACKMOL helpers.

## Next Tasks

1. Add a small synthetic ASE fixture or optional integration test for the TiO2
   and prebuilt-interface adapters.
2. Start migrating the first post-processing family: centroids and
   bubble-surface distance.
