# Constant-force interface kinematics

`molsimflow postprocess constant-force-kinematics` compares constant-force X/Y
branches with the matching zero-force branch on a common time grid. Restarted
motion tables are stitched with displacement offsets before analysis.

The workflow reports:

- 1 ns block velocities in X and Y;
- force-minus-zero-force paired block response and within-trajectory SEM;
- regularly sampled velocity traces and normalized velocity autocorrelation;
- positive-lobe ACF time diagnostics;
- sustained, intermittent, reversal, and fluctuation response classes;
- overview displacement and ACF figures.

## Contract

```json
{
  "schema_version": 1,
  "time_origin_step": 28200000,
  "timestep_fs": 0.5,
  "block_ps": 1000.0,
  "velocity_sample_ps": 10.0,
  "acf_max_lag_ps": 500.0,
  "minimum_effect_mps": 0.05,
  "classification_sigma_multiplier": 1.0,
  "motion_columns": {
    "step": "TimeStep",
    "x": "v_dxrel",
    "y": "v_dyrel"
  },
  "cases": [
    {
      "case_id": "surface_a",
      "branch_id": "f0_shared",
      "direction": "none",
      "motion_tables": ["segment_1.dat", "segment_2.dat"]
    },
    {
      "case_id": "surface_a",
      "branch_id": "f8e-5_x",
      "direction": "x",
      "motion_tables": ["segment_1_x.dat", "segment_2_x.dat"]
    },
    {
      "case_id": "surface_a",
      "branch_id": "f8e-5_y",
      "direction": "y",
      "motion_tables": ["segment_1_y.dat", "segment_2_y.dat"]
    }
  ]
}
```

Every `case_id` must contain exactly one `direction=none` entry. X/Y branches
are interpolated onto that baseline branch before subtraction.

## Outputs

- `branch_summary.tsv`
- `block_velocity.tsv`
- `velocity_timeseries.tsv`
- `velocity_acf.tsv`
- `input_manifest.tsv`
- `summary.json`
- `REPORT.md`
- `kinematics_overview.{png,pdf}`
- `velocity_acf_overview.{png,pdf}`

The block SEM and autocorrelation are single-trajectory diagnostics. They are
not independent-replicate uncertainty, equilibrium diffusion, mobility, or a
friction coefficient.
