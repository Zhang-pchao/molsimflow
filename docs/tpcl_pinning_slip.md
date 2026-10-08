# TPCL Pinning--Slip Analysis

`molsimflow postprocess tpcl-pinning-slip` resolves local three-phase contact-line
(TPCL) dwell--jump candidates from LAMMPS dump segments. It separates object
translation, mean contact-line radius, low-order shape modes, and a remaining
local residual before relating accepted local motion to surface sites, hydration,
hydrogen bonds, curvature, contact angle, and optional gas contact.

The workflow is designed for explicit, externally configured nanobubble and
nanodroplet analyses. It does not contain system paths, atom ranges, surface
compositions, or scheduler settings.

## Case configuration

Pass a JSON file with explicit trajectory and structure metadata:

```json
{
  "kind": "nanobubble",
  "trajectories": ["segment_0.dump", "segment_1.dump"],
  "initial_xyz": "model.xyz",
  "surface_range": "1:8000",
  "phase_range": "8001:8600",
  "water_range": "8601:48000",
  "surface_z_A": 20.0,
  "timestep_fs": 0.5,
  "cluster_cutoff_A": 5.5,
  "contact_cutoff_A": 4.0,
  "arc_bins": 36,
  "minimum_contact_points": 8
}
```

Ranges are inclusive, one-based atom-ID ranges. Nanobubbles use the largest
periodic cluster of diatomic phase-molecule centers; nanodroplets use the largest
periodic water-oxygen cluster. Surface CH3-carbon and SiOH-oxygen sites are
identified from initial C--H and O--H connectivity and then mapped with current
coordinates.

Run the analysis with an explicit output root and font file:

```bash
molsimflow postprocess tpcl-pinning-slip \
  --config case.json \
  --output-dir tpcl_results \
  --font-path Arial.ttf
```

Optional `--start-ns`, `--end-ns`, and `--max-frames` arguments support bounded
geometry checks. A later trajectory segment replaces an earlier duplicate time
step. Step zero is dropped by default. Missing frames, invalid contours, and
segment boundaries remain explicit; the analysis does not interpolate or smooth
across them.

## Candidate definition

The default contour has 36 fixed angular arcs. A contour is valid only when the
phase center lies inside a nondegenerate contact-phase hull and every ray
intersection is finite. Localization noise is estimated from consecutive local
residual differences. Default thresholds are:

- localization noise: `max(0.5 A, 1.4826 * MAD(delta residual) / sqrt(2))`;
- dwell tolerance: `max(1.0 A, 2 * noise)`;
- jump threshold: `max(2.0 A, 4 * noise)`;
- at least four dwell samples and two post-jump confirmation samples;
- a residual change of at least `2 * noise`;
- same-sign support from an adjacent arc;
- no more than 70% of the jump attributed to object-center translation;
- the same nearest surface-site ID in at least 75% of dwell samples.

Adjacent arc records at the same transition are grouped into one independent
event cluster. A chemistry class needs at least two independent clusters before
it is labeled a repeated candidate. These rules identify sampled candidate
stick--slip motion, not a transition-state path or free-energy barrier.

## Per-case outputs

The command writes:

- `frame_metrics.csv` with contour validity, center motion, radius, and shape
  modes;
- `contour_points.csv.gz` and `local_arc_metrics.csv.gz` with compact local
  geometry and environment data;
- `candidate_events.csv` with candidate, insufficient-repetition, and rejected
  records plus rejection reasons;
- `summary.json` and `manifest.json` with thresholds, resolution, and provenance;
- PBC contour, kymograph, motion-decomposition, and event-timeline PNG/PDF
  figures when plotting is enabled.

## Cross-case comparison

`molsimflow postprocess tpcl-pinning-slip-compare` accepts a tab-separated case
manifest. Required columns are `case_id`, `kind`, `output_root`, `ch3_sites`, and
`oh_sites`; nanobubble rows also require `attachment_time_ns`. Each output root
must expose an accepted immutable run through `latest`.

```bash
molsimflow postprocess tpcl-pinning-slip-compare \
  --manifest cases.tsv \
  --output-dir tpcl_comparison \
  --font-path Arial.ttf \
  --block-ps 200 \
  --event-half-window-ps 100 \
  --bootstrap-replicates 2000 \
  --seed 20260830
```

The comparison writes case and event-cluster summaries, explicit contour-validity
intervals, nonoverlapping time blocks, time-block bootstrap summaries,
event-aligned environment traces, stricter-only threshold sensitivity, and all
nonzero circular angular shifts for mixed-surface null tests. It reports a fair
window only when at least 95% of its frames have valid contours; low-coverage
windows remain in the tables but are excluded from comparison curves.

Circular-shift tests use every motion-qualified cluster before applying the
chemistry-repetition label. This avoids selecting on the same chemistry mapping
being tested. Benjamini--Hochberg adjustment is applied across the reported
surface-type, boundary, hydration, hydrogen-bond, site-transition, and gas-contact
spatial tests.

## Interpretation boundary

Repeated dwell--jump clusters support an effective kinetic-landscape proxy only.
They do not establish equilibrium pinning, causality, or a free-energy barrier.
Time-block intervals quantify variation within one trajectory and are not a
substitute for independent initial-condition repeats. Processes shorter than the
trajectory output interval are unresolved.
