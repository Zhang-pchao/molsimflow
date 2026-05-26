# Double-Bubble Merge Workflow

## Purpose

Most algorithms should live in generic namespaces such as `structure`,
`plumed`, `postprocess`, `plotting`, and `config`.  Project-specific orchestration
for the double-bubble merging system belongs under:

```text
src/molsimflow/workflows/double_bubble_merge
```

This namespace is for stage ordering, workflow conventions, and adapters that
are too specific for generic package code but still useful for this family of
systems.  It must not contain private paths or cluster environment assumptions.

## Current Stages

The current stage plan is exposed by:

```python
from molsimflow.workflows.double_bubble_merge import recommended_postprocess_stages

for stage in recommended_postprocess_stages():
    print(stage.name, stage.status, stage.reusable_module)
```

Current migrated or partial stages include:

- structure preparation;
- PLUMED generation;
- coalescence-state assignment;
- bubble geometry;
- ion species;
- bridge descriptors;
- bridge-water dewetting;
- bridge-water dynamics;
- bridge-water escape;
- water orientation;
- H-bond network;
- contact graph topology;
- local environment;
- species transitions;
- transition events;
- bridge film;
- ion-water coupling;
- FES barriers;
- case comparison.

## Residual Adapters

The workflow namespace also exposes the optional double-bubble adapters that are
still represented only by legacy scripts:

```python
from molsimflow.workflows.double_bubble_merge import residual_adapter_plan

for adapter in residual_adapter_plan():
    print(adapter.name, adapter.status, adapter.expected_output)
```

These adapters are not public-package blockers.  They should be added only when
we want direct trajectory-to-table commands for this project family.  Generic
analysis should continue to use explicit CSV inputs handled by the migrated
`molsimflow.postprocess` modules.

Current residual adapter categories:

- seed-position table generation for `bridge-water-escape`;
- water-orientation sample generation for `water-orientation-summary`;
- H-bond edge-table generation for `hbond-network`;
- contact-edge and local-environment sample generation for `contact-graph` and
  `local-environment`;
- microstate / region-QC tables for double-bubble diagnostics;
- publication and case-synthesis scripts, which should not be migrated
  directly.

## Migration Rule

When a legacy feature is reusable across systems, migrate it into
`src/molsimflow/postprocess` or another generic namespace.  Use the
double-bubble workflow namespace only for stage composition or adapters that
encode assumptions specific to this double-bubble coalescence project.
