# Bridge Electrostatics Migration

`molsimflow.postprocess.bridge_electrostatics` migrates the reusable table
logic from the legacy bridge-electrostatics scripts.

## Scope

The module accepts explicit CSV inputs and writes reusable descriptor tables:

- bridge-axis charge profile;
- 1D unscreened Poisson-style potential proxy;
- frame-level bridge-core and near-surface-shell charge metrics;
- surface-gap binned summaries;
- optional apparent disjoining-pressure proxy from binned FES slopes;
- core/shell/bridge-cylinder charge-separation and species-budget metrics.

The outputs are proxy descriptors.  They are not a self-consistent
Poisson-Boltzmann, dielectric, or full atomistic electrostatic calculation.

## Inputs

The ion table should contain at least:

- `global_frame`;
- `species_canonical` or a compatible legacy species label;
- `species_charge_e`, or a species name that can be mapped to a formal charge;
- `bridge_axis_s_A` and `bridge_axis_rho_A`.

If `bridge_axis_s_A` and `bridge_axis_rho_A` are absent, pass
`--derive-bridge-coordinates` and provide Cartesian ion positions plus frame
bridge/bubble-center columns:

- ion table: `x_A`, `y_A`, `z_A`;
- frame table: `bridge_center_*_A`, `bubble_A_center_*_A`,
  `bubble_B_center_*_A`.

The optional frame table can also provide `analysis_surface_gap_A`,
`dynamic_surface_gap_est_A`, `d3d_all`, `bridge_core_volume_A3`,
`Nw_bridge_core`, and `fes_free_energy_relative_raw_interp`.

## CLI

```bash
molsimflow postprocess bridge-electrostatics \
  --ion-table bridge_species_position_table.csv \
  --frame-table bridge_microstate_frame_table.csv \
  --output-dir bridge_electrostatics \
  --gap-mode asis
```

Use `--gap-mode dynamic` to overwrite the analysis gap from
`dynamic_surface_gap_est_A`, or `--gap-mode nominal` to compute
`d3d_all - nominal_radius_a_A - nominal_radius_b_A`.

## Not Migrated

The public package intentionally does not include hardcoded case discovery,
legacy profile path lookup, scheduler submission, or manuscript plotting from
the original scripts.  Those can be rebuilt later as explicit workflow adapters
or plotting templates on top of this API.
