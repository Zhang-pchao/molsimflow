# Local Environment Migration

## Scope

`molsimflow.postprocess.local_environment` migrates the reusable table-summary
part of the legacy local water environment workflow.

The module starts from an explicit sample table with:

- frame;
- persistent entity id;
- environment class;
- optional time;
- optional numeric feature columns such as `q`, `lsi`, coordination counts, or
  nearest-neighbor distances.

It writes:

- standardized local-environment samples;
- frame-level class counts/fractions and numeric feature summaries;
- class-level numeric feature summaries;
- environment-class transition count and probability matrices for persistent
  entities;
- run statistics.

The module reuses the generic transition-matrix implementation.  The
trajectory-specific classification logic and reference-library similarity
scoring from the legacy script remain separate migration targets.

## CLI

```bash
molsimflow postprocess local-environment \
  --input local_environment_samples.csv \
  --output-dir local_environment \
  --time-column time_ns \
  --class-order tetrahedral,interfacial,distorted \
  --feature-column q,lsi
```

Generated files:

- `local_environment_samples.csv`;
- `local_environment_frame_summary.csv`;
- `local_environment_class_summary.csv`;
- `local_environment_transition_counts.csv`;
- `local_environment_transition_probabilities.csv`;
- `state_statistics.csv`.

## Python API

```python
from molsimflow.postprocess.local_environment import (
    LocalEnvironmentConfig,
    analyze_local_environment,
    parse_class_order,
)

outputs = analyze_local_environment(
    input_csv="local_environment_samples.csv",
    output_dir="local_environment",
    config=LocalEnvironmentConfig(
        class_order=parse_class_order("tetrahedral,interfacial,distorted"),
    ),
    time_column="time_ns",
    feature_columns=("q", "lsi"),
)
```

## Remaining Work

The residual legacy logic for building the local-environment samples from
trajectory membership tables, scoring similarity to reference environments, and
running sensitivity analyses should be migrated as smaller explicit adapters or
kept in workflow-specific code if tied to one project layout.
