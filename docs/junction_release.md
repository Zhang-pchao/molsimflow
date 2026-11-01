# Incremental chemical-junction release screening

`python -m molsimflow.postprocess.junction_release --contract contract.json --output results/branch`

This adapter reuses `io.lammps_dump` and `nanobubble_attachment` for checked
orthorhombic XYZ-periodic frames, paired N2 centers, largest components, surface
distances and shape tensors. It does not change the global CLI or an installed
package. Select the intended immutable source snapshot explicitly.

## Added modules

- `trajectory_feature_cache`: a transactional SQLite cache identified by an
  analysis key; each source/step retains a canonical frame hash and features.
- `junction_release_metrics`: immutable original material/face identities,
  four-face contact screening and reference-site area weights, fixed support,
  N2-bond checks and water-neighbor descriptors.
- `junction_release_events`: temporal assembly, membership continuity, reseed
  boundaries, missing-frame resets and persistent geometric state episodes.
- `junction_release`: contract validation, bounded streaming, incremental
  execution, artifact hashes and current result publication.

## Contract requirements

A version1 JSON contract specifies a frozen reference NPZ and hash, parameters,
unique nonoverlapping source step intervals, source identity and admitted size,
analysis key, output sampling and explicit reseed boundaries. Reference NPZ
arrays contain expected atom types, nitrogen/solvent-O/fixed-Si IDs, initial fixed
positions, original terminal-site IDs, material/face groups, box lengths and
reference-plane grid weights. No user paths, atom counts or scheduler policy
are hardcoded in the library.

The analysis key must identify all executed code, reference arrays and metric
parameters. The intended caller constructs it from frozen code hashes, reference
hash and parameters. Expanding a source maximum step retains the same key; a
changed definition requires a new cache. Existing result rows are never silently
replaced by different same-step contents. Append-only source identity is checked
by path/inode, admitted size and a stored prefix fingerprint; cached boundaries
are reread and rehashed on reuse. This is not a full immutable-file hash guarantee
for an arbitrary externally rewritten trajectory.

## Physical meaning

- Original surface site identity and material ownership persist across motion.
  Contacts to each material are counted independently: an N2 can touch both.
- Site area is the reference XY area assigned by periodic nearest-site cells to
  gas-contacting terminal sites, checked with1A and2A grids. It is an operational
  gas-covered site area, not a continuum footprint or thermodynamic adhesion area.
- The water-neighbor mask refers to original solvent oxygen IDs within the
  declared cutoff. Neither a neighbor nor a newly wet site proves water entered
  the solid-gas gap. No neutral-water or proton-charge assignment is inferred.
- `DETACHMENT_CANDIDATE_GEOMETRY_ONLY` requires the declared main-cluster count,
  no material/face contacts, gap and elementary integrity screens. Persistence
  resets at reseeds, missing frames and large cluster-identity changes. Final
  acceptance additionally requires film, topology, chemistry and event review.
- A minimum-image center reconstruction can become ambiguous for a truly
  periodic-spanning cluster. Persistent candidates require a graph-topology
  review; this adapter does not claim to resolve that topology.
- Existing hydrogen-assignment WARNs remain external species context. Geometric
  completion does not remove them or certify full framework bond identities.

## Tests

`pytest -q tests/test_junction_release.py tests/test_nanobubble_attachment.py tests/test_planar_motion.py`

Tests include Z-periodic opposite-face attachment, reference area conservation,
identity/nonfinite rejection, temporal gaps/reseeds, cache identity protection,
exact cached replay and true append-only extension.
