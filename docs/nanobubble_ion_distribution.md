# Nanobubble ion distributions

`nanobubble-ion-distribution` samples fixed ions and reactive oxygen-species
candidates relative to a silica surface and the largest N2 molecular cluster.
It is intended for staged comparisons of detached and surface-attached
nanobubbles.

## Interface definitions

- The primary solid coordinate is the periodic mean z position of reference
  Si atoms within a configurable distance of the highest reference Si atom.
  Ion distances are directed along +z modulo the box length, matching the
  fluid-facing side of the slab rather than folding the upper liquid into
  negative minimum-image distances.
- A translated terminal-group plane is also reported using the existing
  dynamic slab reference and the same directed +z convention. The two z
  origins are never treated as identical.
- Consecutive N atom pairs define N2 molecular centers. The gas object is the
  largest PBC-connected cluster of those centers, so disconnected or dissolved
  N2 is excluded.
- `r_minus_bubble_R90_A` is a spherical geometry proxy. For an attached or
  anisotropic bubble, prefer `nearest_main_n2_center_A` together with the
  cylindrical coordinates and solid-relative z coordinate.

Na and Cl retain their explicit atom-type identities. H3O and OH are
geometric candidates reconstructed every frame by assigning H atoms to their
nearest valid O or C owner under PBC. The default O-H and C-H cutoffs match the
surface proton-transfer workflow.

## Example

```bash
molsimflow postprocess nanobubble-ion-distribution \
  --trajectory bubble.lammpstrj \
  --output-dir ion_samples \
  --reference-structure model.initial.xyz \
  --surface-range 1:8730 \
  --nitrogen-range 8731:9330 \
  --solution-range 9457:49056 \
  --terminal-surface-z-A 20.68091239 \
  --stage pre_attachment:0.01:4.28 \
  --stage attachment_transition:4.29:5.29 \
  --stage matched_attached:6.46:8.46 \
  --stage late_absolute:8.00:10.00
```

The output contains `frame_summary.csv`, compressed per-ion samples,
`summary.json`, and `manifest.json`. Stage-specific density, formal-charge,
and plotting choices remain downstream operations so study windows are not
hard-coded into the reusable sampler.

Formal charges are species labels, not atomic partial charges. The resulting
profiles can support a descriptive double-layer organization candidate, but
they are not an electrostatic potential, a dynamic-polarization charge field,
or proof of an equilibrium electric double layer.
