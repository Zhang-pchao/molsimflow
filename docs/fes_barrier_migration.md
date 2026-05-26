# FES Barrier Migration

## Migrated Core

`molsimflow.postprocess.fes_analysis` provides the table-oriented core of the
legacy 1D FES comparison workflow:

- read one or more whitespace FES files;
- shift each curve by the minimum in a configurable reference window;
- optionally smooth curves with a dependency-light moving average;
- zero each smoothed curve by a configurable CV window;
- compute max-min barrier summaries for configurable CV windows;
- write processed curve and barrier summary CSV files.

The migrated core intentionally does not include preset case paths, plotting, or
manuscript-level comparison reports.  Those should be rebuilt as separate layers
on top of the generated CSV tables.

## CLI Examples

Run directly from explicit curves:

```bash
molsimflow postprocess fes-barriers \
  --curve fes-rew.dat "case A" tio2 \
  --curve other-fes-rew.dat "case B" bulk \
  --output-dir fes_barrier_results \
  --barrier-window contact:0:5 \
  --barrier-window open:10:20 \
  --smooth-window 5 \
  --smooth-passes 2
```

Run from a manifest:

```bash
molsimflow postprocess fes-barriers \
  --manifest fes_manifest.csv \
  --output-dir fes_barrier_results \
  --reference-low 200 \
  --reference-high 600
```

Manifest columns:

- `path`: FES file path;
- `label`: display or case label;
- `group`: optional grouping label;
- `dataset_key`: optional stable key.

## Outputs

The command writes:

- `fes_processed_curves.csv`;
- `fes_barrier_summary.csv`;
- `fes_input_manifest.csv`.

## Notes

FES files are expected to contain at least two whitespace columns: CV and free
energy.  Lines beginning with `#` are ignored, so PLUMED-style `#! FIELDS`
headers are accepted.
