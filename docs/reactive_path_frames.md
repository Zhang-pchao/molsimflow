# Reactive-path frame descriptors

`molsimflow postprocess reactive-path-frames` computes manifest-backed protonation,
ion-defect, peroxide, and hydrogen-bonded water-wire descriptors from XYZ frames.
The command writes only to the requested output directory; raw coordinates and
project-specific manifests remain outside the repository.

Required manifest columns are `system`, `state`, `replicate`, and `xyz_path`.
Periodic dimensions may be supplied as one cubic `box_A` value or as all three
orthorhombic fields `box_x_A`, `box_y_A`, and `box_z_A`. `stratum` is optional.
Relative `xyz_path` values are resolved against the manifest directory.

The JSON site configuration uses explicit one-based atom indices. Organic
oxygens do not need to form the first oxygen block in the XYZ file:

```json
{
  "systems": {
    "example": {
      "organic_oxygen_indices": [122, 123, 124, 125],
      "donor_carbon_index": 121,
      "target_oxygen_index": 125,
      "peroxy_proximal_oxygen_index": 124,
      "peroxy_attachment_carbon_index": 120
    }
  },
  "thresholds": {
    "oxygen_h_assignment_A": 1.30,
    "state_bond_A": 1.25,
    "hbond_h_acceptor_A": 2.50,
    "hbond_oxygen_oxygen_A": 3.50,
    "hbond_angle_degree": 130.0
  }
}
```

Example:

```bash
molsimflow postprocess reactive-path-frames \
  --manifest frame_manifest.tsv \
  --site-config reactive_sites.json \
  --output-dir reactive_path_results
```

Outputs are `reactive_path_frames.tsv`, `reactive_path_summary.tsv`, and
`reactive_path_metadata.json`. The geometry classes are structural labels, not
rates, populations, or committor assignments.
