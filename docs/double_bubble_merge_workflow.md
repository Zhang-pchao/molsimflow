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
- FES barriers;
- case comparison.

## Migration Rule

When a legacy feature is reusable across systems, migrate it into
`src/molsimflow/postprocess` or another generic namespace.  Use the
double-bubble workflow namespace only for stage composition or adapters that
encode assumptions specific to this double-bubble coalescence project.
