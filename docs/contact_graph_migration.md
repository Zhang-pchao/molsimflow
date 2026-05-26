# Contact Graph Migration

## Scope

`molsimflow.postprocess.contact_graph` migrates the reusable contact-topology
core from the legacy ion/water topology workflow.

The first engineered layer starts from an explicit contact edge table.  Required
columns are:

- frame;
- source id;
- target id.

Optional columns can provide edge type, source/target roles, source/target
regions, bridge-axis coordinates (`source_s_A`, `target_s_A`), and surface gap.

The module computes per-frame graph metrics without requiring `networkx`:

- node, edge, and connected-component counts;
- largest-component fraction and bridge-only largest-component fraction;
- bridge-spanning flag from axial coordinates;
- average degree overall and by role;
- ion/surface-mediated edge fractions;
- articulation-node counts by role;
- cycle rank;
- gap-binned topology summaries.

## CLI

```bash
molsimflow postprocess contact-graph \
  --input contact_edges.csv \
  --output-dir contact_graph \
  --case-label caseA \
  --source-s-column source_s_A \
  --target-s-column target_s_A \
  --gap-column surface_gap_A \
  --bridge-s-min-A -10 \
  --bridge-s-max-A 10
```

Generated files:

- `contact_graph_edges.csv`;
- `contact_graph_frame_summary.csv`;
- `contact_graph_gap_summary.csv`;
- `state_statistics.csv`.

## Remaining Work

The legacy topology script also builds contact edges from atomistic membership
tables, runs detailed cycle-basis analyses, computes local-environment
features, scores environment similarity, and performs sensitivity studies.
Those should be migrated as separate explicit table APIs or workflow adapters.
