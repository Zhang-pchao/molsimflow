# Surface proton-transfer candidates

`molsimflow postprocess surface-proton-transfer` analyzes reactive LAMMPS dump
segments without assuming fixed water molecules. It rebuilds O-H and C-H
ownership in every frame under orthorhombic periodic boundaries, using stable
atom IDs from an initial extended XYZ structure.

The command distinguishes:

- top SiOH H-loss, H-gain, and H-identity exchange candidates;
- top CH3 C-H loss, gain, and H-identity exchange candidates;
- solution OH-like and H3O-like oxygen coordination;
- identity-resolved solution-ion episodes grouped by stable oxygen atom ID;
- system-level H3O-like/OH-like co-occupancy episodes;
- consecutive sampled episodes that satisfy a configurable persistence gate;
- footprint, TPCL, far-field, and pre-contact/unmapped locations.

Example:

```bash
molsimflow postprocess surface-proton-transfer \
  --trajectory segment_1.lammpstrj \
  --trajectory segment_2.lammpstrj \
  --initial-xyz model.initial.xyz \
  --surface-range 1:8000 \
  --water-range 9001:12000 \
  --contact-line contact_line.csv \
  --contact-line-points contact_line_points.csv \
  --surface-z-A 20.0 \
  --output-dir proton_transfer_results \
  --font-path Arial.ttf
```

Restart duplicates are resolved by timestep, with the later trajectory segment
replacing the earlier frame. The default persistence gate is two consecutive
sampled frames. `observed_span_ps` is the interval between the first and last
observations; it is not an inferred continuous lifetime.

The main outputs are frame counts, individual surface-site candidates,
solution-ion candidates, identity-resolved raw sampled episodes,
persistence-filtered events, a manifest, and an Arial plot. Event rows retain
contact-line radius and lateral displacement when those data are available.
System-level co-occupancy only means that at least one H3O-like and one OH-like
candidate occur in the same frame; it is not a tracked ion-pair identity.

These outputs are geometry candidates, not formal chemical identities.
Coordination cutoffs must be tested, and events faster than the dump interval
cannot be resolved. Assigning SiO-, H3O+, or C-H cleavage requires compatible
sampling cadence plus independent bond-order or electronic-structure evidence.
