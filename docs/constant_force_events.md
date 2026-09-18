# Constant-force event audit

`molsimflow postprocess constant-force-events` audits short windows around
geometric species changes, high-Z oxygen excursions, nonzero Z image flags,
and wall approaches. It also joins restart-segmented displacement tables before
measuring event-aligned lateral motion.

The workflow streams plain or `.zst` LAMMPS custom dumps. It requires full-atom
state trajectories for O-H assignment and may use smaller oxygen-only dumps for
high-Z event discovery. X and Y are periodic; Z is treated as nonperiodic for
nearest-neighbor assignments.

## Contract

Paths may be absolute or relative to the contract file. Each case/branch pair
must be unique.

```json
{
  "schema_version": 2,
  "time_origin_step": 1000000,
  "timestep_fs": 0.5,
  "window_ps": 20.0,
  "merge_gap_ps": 11.0,
  "high_z_threshold_A": 80.0,
  "wall_clearance_threshold_A": 1.0,
  "types": {
    "hydrogen": 1,
    "oxygen": 2,
    "silicon": 8,
    "carbon": 3
  },
  "cutoffs_A": {
    "oh": 1.35,
    "ch": 1.35,
    "si_o": 2.25,
    "oo": 3.5,
    "hbond_angle_deg": 30.0,
    "lsi": 3.7,
    "proton_sharing_delta": 0.2
  },
  "lsi_neighbor_cap": 24,
  "motion_columns": {
    "step": "TimeStep",
    "x": "v_dxrel",
    "y": "v_dyrel",
    "clearance": "v_topclear"
  },
  "cases": [
    {
      "case_id": "surface_a",
      "branch_id": "force_x",
      "state_trajectories": [
        "segments/state_01.lammpstrj.zst",
        "segments/state_02.lammpstrj.zst"
      ],
      "oxygen_audit_trajectories": [
        "segments/oxygen_01.lammpstrj.zst",
        "segments/oxygen_02.lammpstrj.zst"
      ],
      "motion_tables": [
        "segments/motion_01.dat",
        "segments/motion_02.dat"
      ],
      "species_tables": [
        "segments/species_01.tsv",
        "segments/species_02.tsv"
      ]
    }
  ]
}
```

Schema 2 requires an explicit carbon type and C-H cutoff. Hydrogen atoms are
assigned to the closest valid oxygen or carbon under periodic X/Y boundaries;
carbon-owned hydrogen atoms are reported separately and are excluded from
`unassigned_H`. Schema 1 remains readable for systems without carbon.

Species tables use `step`, `time_ps`, `O_solution`, `OH_solution`,
`OH4plus_solution`, and `unassigned_H` by default. Override the first two names
with `species_step_column` and `species_time_column` inside a case entry.

## Outputs

- `events.tsv`: merged episodes, return checks, tracked oxygen identities,
  proton-sharing counts, mean local water order, H-bond coordination, and Z
  image status.
- `event_sources.tsv`: every raw species, high-Z, Z image, and wall sample.
- `frame_species.tsv`: geometric species counts for all frames in event windows.
- `atom_identity.tsv`: O/H identities, nearest and second-nearest O-H distances,
  proton-sharing deltas, `q_tet`, LSI, O-O coordination, and water-water or
  water-surface H-bond counts for tracked or abnormal oxygen atoms.
- `motion_event_summary.tsv`: local displacement and pre/core/post velocities.
- `input_manifest.tsv`: size and SHA256 for the contract and all inputs.
- `summary.json` and `REPORT.md`: machine-readable and review-oriented summaries.

The output directory must not already exist. Conflicting duplicate species rows,
missing atom types, missing solution oxygen, non-increasing motion steps, and
invalid event times fail closed.

Species and proton-sharing labels are geometric diagnostics. The sharing delta
is the second-nearest minus nearest O-H distance and requires the second oxygen
to lie inside the O-H cutoff. `q_tet` is reported only when at least four oxygen
neighbors lie inside the configured O-O cutoff. Event-aligned association does
not by itself
establish formal charge identity, reaction kinetics, wall causality, or a
friction coefficient.
