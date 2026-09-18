# Constant-force water structure

molsimflow postprocess constant-force-water-structure computes consistent
water-network and local-order diagnostics for constant-force trajectories with
different morphologies. Input paths, atom ranges, chemistry types, cutoffs,
surface references, and region definitions are supplied by a JSON contract;
the implementation contains no project paths or atom identifiers.

The command streams plain or zstd-compressed LAMMPS custom dumps and treats X
and Y as periodic. Z remains nonperiodic.

## Command

```bash
molsimflow postprocess constant-force-water-structure \
  --contract water-structure.json \
  --output water-structure-results
```

The output directory must not already exist.

## Contract

    {
      "schema_version": 1,
      "time_origin_step": 1000000,
      "timestep_fs": 0.5,
      "surface_atom_range": [1, 8000],
      "water_atom_range": [8001, 11891],
      "oxygen_type": 2,
      "hydrogen_type": 1,
      "oh_cutoff_A": 1.25,
      "oo_cutoff_A": 3.5,
      "hbond_angle_deg": 30.0,
      "lsi_cutoff_A": 3.7,
      "lsi_neighbor_cap": 24,
      "cluster_cutoff_A": 3.5,
      "write_plots": true,
      "cases": [
        {
          "case_id": "finite_drop",
          "branch_id": "force_x",
          "direction": "x",
          "region_mode": "islands",
          "trajectories": [
            "segment_1.lammpstrj.zst",
            "segment_2.lammpstrj.zst"
          ]
        },
        {
          "case_id": "spread_film",
          "branch_id": "force_y",
          "direction": "y",
          "region_mode": "layers",
          "surface_z_A": 18.0,
          "z_edges_A": [0.0, 3.5, 6.5, 10.0, 15.0, 25.0, 100.0],
          "trajectories": [
            "film_segment_1.lammpstrj.zst",
            "film_segment_2.lammpstrj.zst"
          ]
        }
      ]
    }

Trajectory segments must form one strictly increasing, constant-interval time
series. A duplicate restart frame may be present and is de-duplicated by step.
Atom ranges are inclusive.

## Region modes

islands constructs an O-O connectivity graph with cluster_cutoff_A and reports
all water, the largest connected island, and the union of satellite islands.

layers bins water oxygen by height relative to surface_z_A. The interval
z_edges_A[i] <= z - surface_z_A < z_edges_A[i+1] is named layer_i. Water outside
the supplied edges is retained in the all-water row and labeled outside for
residence and exchange accounting.

Region labels follow each water oxygen identity over time. Region changes
produce exchange rows and terminate one residence episode.

## Hydrogen bonds and water order

Water O-H ownership uses the nearest oxygen inside oh_cutoff_A. A hydrogen bond
requires O-O distance within oo_cutoff_A and donor O-H...O angular deviation
within hbond_angle_deg. Water-water and water-surface donor directions are
reported separately.

Each region reports:

- water-water and water-surface H-bond edge counts;
- H-bond network component count and largest-component fraction;
- one-frame edge Jaccard similarity and turnover;
- tetrahedral order q_tet;
- local structure index (LSI);
- O-O coordination;
- the fraction of water oxygen with a geometric hydrogen count other than two.

## Outputs

- water_structure_by_frame.tsv: frame and region observables;
- region_exchange.tsv: identity-resolved region transitions;
- region_residence_summary.tsv: residence distributions and right-censor counts;
- hbond_persistence_summary.tsv: water-water and water-surface edge persistence;
- input_manifest.tsv: input sizes and SHA256 hashes;
- water_structure_overview.png when write_plots is true;
- summary.json and OUTPUT-SHA256SUMS.

Turnover and persistence resolve only the supplied frame interval. Residence
episodes at the final frame are right censored. Results from one trajectory are
descriptive diagnostics; they are not independent-replica uncertainty, free
energies, reaction rates, friction coefficients, or causal mechanism tests.
