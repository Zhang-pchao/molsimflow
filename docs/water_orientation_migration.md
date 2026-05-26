# Water Orientation Migration

## Scope

`molsimflow.postprocess.water_orientation` migrates the reusable core of the
legacy bridge-water orientation workflow.

The first engineered layer covers:

- water dipole/orientation geometry in a two-center reference frame;
- `s`/`rho` projection relative to the bridge axis;
- radial orientation profiles;
- `s-rho` orientation maps;
- CV-binned orientation summaries;
- angle-distribution tables;
- per-frame orientation summaries.

This migration deliberately does not copy the legacy case discovery, plotting,
or monolithic trajectory runner.  Those remain workflow-layer concerns.  The
public API starts from explicit coordinates or an explicit CSV sample table.

## Python API

```python
import numpy as np

from molsimflow.postprocess.water_orientation import (
    WaterOrientationSummaryConfig,
    analyze_water_orientation,
    compute_water_orientation_sample,
)

sample = compute_water_orientation_sample(
    oxygen_position=np.array([0.0, 0.0, 0.0]),
    hydrogen_positions=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    center_a=np.array([-1.0, 0.0, 0.0]),
    center_b=np.array([1.0, 0.0, 0.0]),
    bounds_or_lengths=np.array([40.0, 40.0, 40.0]),
)

outputs = analyze_water_orientation(
    input_csv="water_orientation_samples.csv",
    output_dir="water_orientation_summary",
    config=WaterOrientationSummaryConfig(rho_bins=30, s_bins=40),
)
```

## CLI

```bash
molsimflow postprocess water-orientation-summary \
  --input water_orientation_samples.csv \
  --output-dir water_orientation_summary \
  --rho-bins 30 \
  --rho-max 12 \
  --s-bins 40 \
  --s-min -20 \
  --s-max 20 \
  --cv-bins 40
```

Generated files:

- `water_orientation_frame_summary.csv`;
- `water_orientation_radial_profile.csv`;
- `water_orientation_sr_map.csv`;
- `water_orientation_cv_summary.csv`;
- `water_orientation_angle_distribution.csv`;
- `state_statistics.csv`.

## Remaining Work

The legacy trajectory-facing runner still contains useful implementation ideas:

- atom selection from dump type ids or PLUMED atom expressions;
- center lookup from centroid tables or bubble atom groups;
- hydrogen assignment for water oxygens;
- optional COLVAR alignment;
- publication plots.

Those should be migrated later as smaller adapters that produce the explicit
sample table consumed by this module.
