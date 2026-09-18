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
  "schema_version": 1,
  "time_origin_step": 1000000,
  "timestep_fs": 0.5,
  "window_ps": 20.0,
  "merge_gap_ps": 11.0,
  "high_z_threshold_A": 80.0,
  "wall_clearance_threshold_A": 1.0,
  "types": {
    "hydrogen": 1,
    "oxygen": 2,
    "silicon": 8
  },
  "cutoffs_A": {
    "oh": 1.35,
    "si_o": 2.25
  },
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

Species tables use `step`, `time_ps`, `O_solution`, `OH_solution`,
`OH4plus_solution`, and `unassigned_H` by default. Override the first two names
with `species_step_column` and `species_time_column` inside a case entry.

## Outputs

- `events.tsv`: merged episodes, return checks, tracked oxygen identities, and
  Z image status.
- `event_sources.tsv`: every raw species, high-Z, Z image, and wall sample.
- `frame_species.tsv`: geometric species counts for all frames in event windows.
- `atom_identity.tsv`: O/H identities and O-H distances for tracked or abnormal
  oxygen atoms.
- `motion_event_summary.tsv`: local displacement and pre/core/post velocities.
- `input_manifest.tsv`: size and SHA256 for the contract and all inputs.
- `summary.json` and `REPORT.md`: machine-readable and review-oriented summaries.

The output directory must not already exist. Conflicting duplicate species rows,
missing atom types, missing solution oxygen, non-increasing motion steps, and
invalid event times fail closed.

Species labels are geometric diagnostics. Event-aligned association does not by
itself establish formal charge identity, reaction kinetics, wall causality, or a
friction coefficient.
