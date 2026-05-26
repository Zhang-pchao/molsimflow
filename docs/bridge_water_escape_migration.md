# Bridge-Water Escape Migration

## Scope

`molsimflow.postprocess.bridge_water_escape` migrates the reusable core of the
legacy seed-water escape-direction workflow.

The migrated layer starts from an explicit seed-position table with one row per
tracked seed atom per frame.  Required columns are:

- a persistent atom/entity id;
- frame and optional time;
- `x`, `y`, `z` position columns;
- an `in_bridge` membership flag.

Optional columns can provide a coalescence state and surface-gap value.  The
workflow classifies each tracked seed as `retained` or `exited`, computes
initial-to-exit and initial-to-destination displacements, and labels escaped
seeds as:

- `toward_bulk_or_zplus`;
- `toward_TiO2_or_zminus`;
- `lateral_xy`;
- `unresolved`.

The public package does not copy legacy case-directory discovery, batch
orchestration, or publication plotting.  Those should remain workflow adapters
that generate the explicit seed-position table.

## CLI

```bash
molsimflow postprocess bridge-water-escape \
  --input seed_positions.csv \
  --output-dir bridge_water_escape \
  --case-label caseA \
  --time-column time_ns \
  --in-bridge-column in_bridge_region \
  --gap-column surface_gap_A \
  --exit-confirm-frames 2 \
  --destination-lag-frames 1
```

Use `--box-lengths LX LY LZ` when wrapped coordinates require a minimum-image
displacement.

Generated files:

- `bridge_water_escape_events.csv`;
- `bridge_water_escape_direction_summary.csv`;
- `bridge_water_escape_gap_summary.csv`;
- `state_statistics.csv`.

## Python API

```python
from molsimflow.postprocess.bridge_water_escape import (
    BridgeWaterEscapeConfig,
    analyze_bridge_water_escape,
)

outputs = analyze_bridge_water_escape(
    input_csv="seed_positions.csv",
    output_dir="bridge_water_escape",
    case_label="caseA",
    config=BridgeWaterEscapeConfig(
        exit_confirm_frames=2,
        destination_lag_frames=1,
    ),
    gap_column="surface_gap_A",
)
```

## Remaining Work

The residual legacy script still contains project-specific helpers for choosing
case windows, discovering trajectory segments, generating seed-position tables
from dump frames, and plotting direction heatmaps.  Those pieces should be
migrated later as explicit adapters around this table API or kept under the
double-bubble workflow namespace if they stay case-specific.
