# Bridge-Water Dewetting Migration

## Scope

`molsimflow.postprocess.bridge_water_dewetting` migrates the reusable part of
the legacy bridge-water dewetting workflow.  The module computes frame-wise
water occupancy in a bridge cylinder, a dewetting estimate relative to a bulk
water density, and whether bridge waters form a connected cluster spanning the
two cylinder ends.

This is a generic post-processing module.  It does not assume a private case
layout, conda environment, or hardcoded source path.  The double-bubble workflow
only records it as one recommended stage.

## Inputs

Required inputs are:

- a LAMMPS dump trajectory with `id` and coordinate columns;
- a water-oxygen atom id expression, for example `1201-9000:3`;
- two bubble atom groups, either from explicit atom expressions or from a
  PLUMED file containing `bubA_all` and `bubB_all` labels.

Optional inputs are:

- a COLVAR table for `d3d_all` and `bridge_cyl_env.sum`;
- a secondary COLVAR-like table for post-processed coordination columns;
- a user-specified bulk number density for the expected bridge water count.

## CLI Example

```bash
molsimflow postprocess bridge-water-dewetting \
  --dump dump.lammpstrj \
  --plumed in.plumed \
  --water-oxygen-atoms 1201-9000:3 \
  --colvar COLVAR \
  --colvar-post COLVAR_POST \
  --output-dir bridge_water_dewetting \
  --axis z \
  --radius-A 6.5 \
  --lower-A -8.0 \
  --upper-A 8.0 \
  --oo-cutoff-A 3.5 \
  --dump-time-scale-ns 0.000001
```

Use explicit bubble groups instead of a PLUMED file when needed:

```bash
molsimflow postprocess bridge-water-dewetting \
  --dump dump.lammpstrj \
  --water-oxygen-atoms 1201-9000:3 \
  --bubble-a-atoms 1-120 \
  --bubble-b-atoms 121-240 \
  --output-dir bridge_water_dewetting
```

## Outputs

The command writes:

- `bridge_water_dewetting.csv`: frame-wise bridge water count, expected count,
  dewetting fraction, largest water cluster size, connectivity flag, and matched
  CV columns;
- `bridge_water_dewetting_by_cv.csv`: `d3d_all`-binned averages and connection
  probabilities;
- `bridge_water_dewetting_statistics.csv`: matching counts, density source,
  time scale, and bridge-cylinder volume.

## Notes

The LAMMPS dump reader is intentionally small and local to this module for this
first pass.  If more trajectory-heavy legacy modules are migrated, the reader
and time-alignment helpers should be promoted to shared `molsimflow.io` and
`molsimflow.postprocess` utilities.
