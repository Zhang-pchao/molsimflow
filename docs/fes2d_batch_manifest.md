# 2D FES Batch Manifest

`molsimflow postprocess fes2d-batch-manifest` prepares a reusable command table
for multiple `fes2d-grid` runs.  It replaces legacy hardcoded batch setup with
an explicit manifest.

## Input Manifest

Required column:

- `bias_dir`: directory containing the source `COLVAR_tmp` and where the FES
  subdirectory should be resolved.

Optional columns:

- `case_label`;
- `family`;
- `dataset_key`;
- `safe_label`;
- `colvar_path`;
- `run_dir`;
- `fes_file`;
- `output_dir`.

Relative paths are resolved against the input manifest directory.

## Command

```bash
molsimflow postprocess fes2d-batch-manifest \
  --case-manifest fes2d_cases.csv \
  --output-manifest fes2d_batch_manifest.csv \
  --x-range 20 52 \
  --y-range 50 380
```

The output CSV includes the resolved FES path, output plot directory, existence
flags, and a ready-to-run `molsimflow postprocess fes2d-grid ...` command per
case.

## Boundary

The public package does not copy legacy reweighting drivers, submit SLURM jobs,
or infer manuscript-specific case labels from private directory layouts.  Those
steps should remain explicit workflow scripts or scheduler templates layered on
top of this manifest.
