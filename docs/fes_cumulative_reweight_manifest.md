# FES Cumulative Reweight Manifest

`molsimflow postprocess fes-cumulative-reweight-manifest` prepares a command
table for cumulative-prefix FES reweighting.  It replaces hardcoded system lists
with an explicit input manifest and does not run the external reweight driver.

## Input Manifest

Required columns:

- `system`;
- `workdir`;
- `colvar`;
- `sample_size`.

Optional columns:

- `output_dir`;
- `group`.

Relative paths are resolved against the input manifest directory.

## Command

```bash
molsimflow postprocess fes-cumulative-reweight-manifest \
  --manifest cumulative_cases.csv \
  --driver FES_from_Reweighting4Gly_skipfooter.py \
  --output-root cumulative_reweight \
  --output-manifest cumulative_reweight_manifest.csv \
  --fraction 0.6 \
  --fraction 0.8 \
  --fraction 1.0
```

The output CSV includes `skipfoot`, `keep_after_skip`, output paths, input
existence flags, and a ready-to-run reweight command for each system/fraction.

## Boundary

The package only plans commands.  It does not execute the reweight driver,
delete previous outputs, submit scheduler jobs, or embed a fixed list of
systems.
