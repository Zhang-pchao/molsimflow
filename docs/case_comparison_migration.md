# Case Comparison Migration

## Scope

The migrated case-comparison layer keeps the reusable table operations from
legacy mechanism-summary scripts:

- load a case manifest;
- join one or more case-level descriptor scorecards;
- compute target-minus-reference case deltas;
- compute Pearson and Spearman descriptor correlations against a selected
  target column.

It does not migrate project-specific case roots, hardcoded source-file maps,
publication figure layouts, or narrative report text.  Those pieces should be
rebuilt later as explicit examples or plotting utilities on top of this API.

## Inputs

`cases.csv` must contain a `case_label` column by default:

```text
case_label,case_group
caseA,baseline
caseB,variant
```

`descriptor_manifest.csv` lists case-level descriptor tables:

```text
name,path,case_column,columns
barrier,fes_barriers.csv,case_label,barrier_kjmol
bridge,bridge_descriptors.csv,case_label,bridge_waters|net_charge_e
```

Relative descriptor paths are resolved relative to the descriptor manifest.
If `columns` is blank, numeric descriptor columns are inferred after excluding
common metadata columns such as `case_label`, `source_path`, and `output_dir`.

## Command

```bash
molsimflow postprocess case-scorecard \
  --cases cases.csv \
  --descriptor-manifest descriptor_manifest.csv \
  --output-dir case_comparison_results \
  --pair caseA:caseB:caseB_minus_caseA \
  --target-column barrier__barrier_kjmol \
  --correlate bridge__bridge_waters,bridge__net_charge_e
```

The command writes:

- `case_scorecard.csv`;
- `case_descriptor_delta.csv`;
- `case_descriptor_correlation.csv`;
- `case_comparison_input_manifest.csv`.

## API

Use `molsimflow.postprocess.case_comparison.DescriptorTableSpec` when a Python
workflow already knows the descriptor tables:

```python
from molsimflow.postprocess.case_comparison import (
    CasePairSpec,
    DescriptorTableSpec,
    analyze_case_scorecard,
)

outputs = analyze_case_scorecard(
    case_manifest="cases.csv",
    descriptor_specs=[
        DescriptorTableSpec("barrier", "fes_barriers.csv", columns=("barrier_kjmol",)),
        DescriptorTableSpec("bridge", "bridge_descriptors.csv", columns=("bridge_waters",)),
    ],
    output_dir="case_comparison_results",
    pair_specs=[CasePairSpec("caseA", "caseB")],
    target_column="barrier__barrier_kjmol",
)
```

## Migration Notes

The legacy scripts mixed three concerns: path discovery, descriptor synthesis,
and publication plotting.  This migration only stabilizes descriptor synthesis.
Future plotting code should read the scorecard, delta, or correlation CSVs
instead of rediscovering raw case directories.
