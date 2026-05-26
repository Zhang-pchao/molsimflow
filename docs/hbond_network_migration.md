# H-Bond Network Migration

## Scope

`molsimflow.postprocess.hbond_network` migrates the reusable network-summary
core of the legacy bridge H-bond workflow.

The first engineered layer starts from an explicit H-bond edge table.  Required
columns are:

- frame;
- donor id;
- acceptor id;
- H-bond type, or donor/acceptor species columns from which the type can be
  inferred.

Optional columns can provide frame time, donor/acceptor bridge-axis coordinates
(`donor_s_A`, `acceptor_s_A`), and a surface-gap value.

The module writes:

- standardized edge rows;
- per-frame graph summaries;
- H-bond lifetime summaries by type;
- gap-binned network summaries;
- run statistics.

The implementation uses small in-package graph routines instead of requiring
`networkx`.  MDAnalysis-based H-bond detection remains a later optional
adapter, not part of this table API.

## CLI

```bash
molsimflow postprocess hbond-network \
  --input hbond_edges.csv \
  --output-dir hbond_network \
  --case-label caseA \
  --donor-s-column donor_s_A \
  --acceptor-s-column acceptor_s_A \
  --gap-column surface_gap_A \
  --bridge-s-min-A -10 \
  --bridge-s-max-A 10 \
  --side-thickness-A 1.0
```

Generated files:

- `hbond_edge_table.csv`;
- `hbond_frame_summary.csv`;
- `hbond_lifetime_summary.csv`;
- `hbond_gap_summary.csv`;
- `state_statistics.csv`.

## Python API

```python
from molsimflow.postprocess.hbond_network import (
    HbondNetworkConfig,
    analyze_hbond_network,
)

outputs = analyze_hbond_network(
    input_csv="hbond_edges.csv",
    output_dir="hbond_network",
    case_label="caseA",
    config=HbondNetworkConfig(
        bridge_s_min_A=-10.0,
        bridge_s_max_A=10.0,
        side_thickness_A=1.0,
    ),
    donor_s_column="donor_s_A",
    acceptor_s_column="acceptor_s_A",
    gap_column="surface_gap_A",
)
```

## Remaining Work

The legacy script also performs trajectory-backed H-bond detection, local water
orientation, hydration-motif classification, optional MDAnalysis execution,
case discovery, and plotting.  Those pieces should be migrated separately as
explicit adapters or kept in a double-bubble workflow layer if they remain
case-layout specific.
