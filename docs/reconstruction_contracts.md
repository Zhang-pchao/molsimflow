# Reconstruction provenance and support contracts

`molsimflow.postprocess.reconstruction_contracts` provides small validation
helpers for cached observations, declared structural lineage and strict crossed
parent/site holdouts. It does not read trajectories, infer independence, fit a
model, acquire data or qualify a chemical intervention.

## Records and API

All identity records are frozen dataclasses. Digests are supplied by the caller
from actual artifacts; a nonempty digest field does not independently verify an
artifact. Source hashing and cache/schema parity belong to the project adapter.
No host paths, scheduler policy or study thresholds occur in this module.

```python
SourceKey(source_id, epoch_id, step, cadence_ps, dt_ps)
ObservationRecord(
    key, time_ps, origin_step, origin_time_ps, pbc,
    atom_map_hash, type_map_hash, molecule_map_hash, payload_hash,
    stream_id="core", chemical_mask_valid=True, missing_reason=None,
    physical_frame_id=None, geometry_valid=True, observer_reference_hash=None,
)
validate_observation_contract(
    records, *, requested_keys=None, allow_parity_duplicates=False,
) -> ObservationValidation

ParentKey(scaffold_family, structural_parent_id)
SiteKey(scaffold_family, overlap_family)
ParentRecord(
    sample_id, scaffold_family, structural_parent_id, construction_id,
    treatment_arm, velocity_seed, lineage_verified,
)
validate_parent_lineage(records, *, min_independent_parents=1) -> dict[str, ParentKey]

grouped_holdout_masks(
    parent_keys, site_keys, *, held_out_parents, held_out_sites,
) -> HoldoutMasks
validate_fold_support(
    labels, parent_keys, site_keys, masks, *,
    classes=("WATER_REFILL", "N2_REPLACEMENT"), valid=None, strata=None,
    min_per_class=1, min_parents_per_class=1, min_sites_per_class=1,
) -> FoldSupport
```

`ObservationValidation` has three tuples of indices into the original input:
`canonical_indices`, `duplicate_indices`, and `missing_indices`. Missing rows
remain canonical rows and must be masked explicitly in downstream state clocks.
`HoldoutMasks` has boolean vectors `train`, `test`, and `excluded`.
`FoldSupport` exposes `status` (`READY` or `NOT_FIT`), `reasons`, per-split/class
`counts`, and the `fit_allowed` property. `READY` clears only the declared support
contract. It does not authorize a model or establish statistical power.

## Observation rules

- Time must equal `origin_time_ps + (step - origin_step) * dt_ps`. Steps are
  integers. Integration interval and saved cadence must be positive; saved
  cadence must be an integer multiple of the integration interval.
- Source and epoch remain part of identity. Equal numerical steps are not enough
  to join. A velocity reset requires a new epoch. Time origins and cadence are
  fixed within each source/epoch/stream. Different stream anchors must describe
  the same affine step/time mapping within a source/epoch.
- PBC, atom ID/order, type map, molecular mapping, observer reference and integration interval must
  remain fixed within each source/epoch. A different source may have a different
  gas inventory if it supplies its corresponding identity map. The optional
  `observer_reference_hash` defaults to `None` for legacy records; when supplied
  it must be nonempty, fixed within source/epoch and equal across physical aliases.
  Controlled treatment arms need the same externally qualified immutable scaffold
  reference for registration and local height. Comparing different observer
  references can manufacture geometry changes from exchanged surface O/C atoms.
  The project adapter must enforce a shared reference across those arms; unrelated
  legacy materials are not required to have the same reference.
- The default physical identity is `(source_id, epoch_id, step)`. Raw observations
  repeated within the same stream always fail. The same source/epoch/step must
  retain one physical identity across streams: different explicit aliases, or
  mixed absent/explicit aliases, are rejected even with parity deduplication.
  Core/HF rows in distinct streams
  can deduplicate only when explicitly enabled and time, epoch, integration
  interval, PBC, atom/type/molecule maps, payload and validity agree. Different
  raw sources require the same explicit `physical_frame_id` to declare an alias.
  Saved cadence may differ; extra timestamps add resolution, not replicas.
- Unknown chemistry or geometry requires an explicit `missing_reason`. This is
  retained as missing observation, never coded as no contact or water absence.
  Missing observations cannot establish duplicate payload parity.
- `requested_keys` checks each exact requested selection key. A requested row
  omitted from the collection is an error, even if both range endpoints exist.
  An explicit missing row is present but unknown. Supply a full requested grid
  to detect interior omissions. Without a grid, collection validation makes no
  continuity claim; state reducers must still handle gaps and seams.

The payload digest should cover every field whose equality matters for the join,
with a declared stable serialization. Equal geometry-only digests do not prove
chemistry or molecular-support parity. The helpers validate declared hashes and
cannot replace the caller's byte/numerical comparison of upstream products.

## Parent and site groups

`construction_id` is the independent construction root, not a velocity seed,
treatment arm, source filename, relaxation segment or trajectory restart. All
samples under one structural parent must declare the same root; the same root
cannot be relabeled as several parents. `lineage_verified=True` is an assertion
that supporting provenance was inspected by the caller. It is not inferred from
names or hashes. Identical parent labels with several seeds or arms count once.
An unverified lineage fails. Numerical decorrelation, stationarity and physical
independence require separate evidence even for distinct declared roots.

For sites, callers must resolve registration and spatial overlap before grouping.
`SiteKey` groups the mapped overlap family under the scaffold family. Grouping by
an arm-specific site name can leak the same physical site and is invalid.
The functions accept any hashable group key so existing canonical tuples can be
used without a parallel identity system.

For held-out parent set P and site-family set S:

- `test` contains the intersection: parent in P and site in S;
- `train` contains rows sharing neither: parent outside P and site outside S;
- `excluded` contains both crossing strips.

Every row belongs to exactly one partition. Empty test/train cells are allowed
by the planner and return `NOT_FIT` at support validation. The support validator
also checks leakage in manually supplied masks, including invalid rows, so a
missing outcome does not conceal shared identities. Matching and preprocessing
must occur within the final training partition; the helpers do not perform them.

## Support and limitations

Inputs to the support gate are deduplicated event representatives, not correlated
frames, repeat quadrature estimates or multiple seed labels for one event. The
caller must supply event IDs/provenance and deduplicate them before this gate.
Only valid rows in the requested competing classes count. A missing class,
insufficient distinct parents/sites, unsupported held-out stratum, or parent/site
leakage returns `NOT_FIT` with reasons. `strata` must encode the predeclared
chemical/geometric background, not outcome-selected categories. Every stratum
present among declared test rows needs both classes in both training and test.
Invalid test rows remain in this stratum inventory.

The numeric defaults are minimal software checks. Projects must freeze stronger
scientific support floors and a separate parent-level sample-size plan before
prediction. Satisfying these helpers does not transform correlated observations
into independent trials or demonstrate a causal mechanism.

## Focused verification

```bash
python -m pytest -q -p no:cacheprovider tests/test_reconstruction_contracts.py
```

Counterexamples cover source/epoch collisions, core/HF aliases and parity errors,
missing requested observations, explicit unresolved chemistry, variable inventory,
shared-parent seeds/arms, relabeled roots, crossed-strip exclusion, absent outcome
classes, unsupported test strata, manual leakage and invalid masks. Synthetic
supported folds verify software behavior; they are not observed physical evidence.
