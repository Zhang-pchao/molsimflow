# PIMD Quantum-FES Reweighting

`molsimflow postprocess pimd-reweight` reconstructs bead-defined quantum free
energies from a complete ring-polymer trajectory.  The workflow supports three
explicit path-CV bias modes:

- `centroid_coord`: the bias is evaluated on a CV of the Cartesian centroid;
- `bead_mean`: the bias is evaluated on the arithmetic mean of the bead CVs;
- `bead_density_shared`: one shared field is evaluated on every bead and the
  complete-path bias energy is `mean_b B(q_b)`.

All three modes use the same default quantum target: the bead marginal of the
requested coordinate. For frame `n`, every bead uses the same normalized frame
weight `W_n`, and each of its `P` beads contributes `W_n / P` to that marginal.
The bias representation changes how the complete-path energy is assembled; it
does not change the physical observable being reconstructed. No fixed bead count
or particular molecular system is assumed.

This operation requires all bead coordinates or bead CVs. A CV of the centroid
cannot be converted into the quantum bead marginal by rescaling its free energy.
For a nonlinear CV, `Q(mean_b R_b)`, `mean_b Q(R_b)`, and the distribution of
`Q(R_b)` are distinct observables.

## Command

```bash
molsimflow postprocess pimd-reweight \
  --contract analysis-contract.json \
  --output pimd-reweight-output
```

The output directory must not exist.  The command writes selected frame and
bead tables, FES tables and figures, CV time series, block diagnostics, a
machine-readable summary, and SHA-256 provenance.

Set `analysis_profile` explicitly:

- `core` accepts one or two arbitrary CV names and requires only aligned
  sampling/bead COLVAR tables plus a declared frame-weight provider.  It does
  not require OPES kernels, PIMD thermo logs, atom trajectories, or
  water-ionization fields;
- `water_ionization_opes` retains the richer historical water/OPES diagnostics
  and is the backward-compatible default for existing contracts.

## Contract fields

All paths and system-specific column names belong in the JSON contract.  The
reusable code contains no cluster or case paths.  The representation and weight
semantics should be declared explicitly. The following contract excerpt shows
the recommended estimator defaults; add the source paths, manifest, selection,
grid and bandwidth settings for the actual inputs:

```json
{
  "analysis_profile": "core",
  "source": {
    "sampling_label": "Bead mean",
    "sampling_slug": "bead_mean",
    "restart_duplicate_policy": "keep_first"
  },
  "reweight": {
    "bias_mode": "bead_mean",
    "primary_estimator": "probability_mean",
    "temperature_K": 300.0,
    "energy_unit": "eV",
    "weight_kind": "quasi_static_opes",
    "quasi_static": true,
    "bias_column": "opes.bias",
    "extra_bias_columns": [],
    "cv_names": ["cv1", "cv2"],
    "sampling_cv_names": ["mean.cv1", "mean.cv2"],
    "bead_cv_names": ["cv1", "cv2"]
  }
}
```

`temperature_K` is the physical temperature, not `P * T`. It must be finite
and positive. `energy_unit` defaults to `eV`, and currently only `eV` is accepted
for input energy columns; convert other units before analysis. Precomputed log
weights are dimensionless. Output FES tables explicitly use `kcal/mol`.
`primary_estimator` defaults to `probability_mean`; selecting the bead
free-energy mean as the primary estimator is rejected. The summary records
`target_observable: bead_marginal`.

For shared bead density, each bead COLVAR must contain the local field value
and identical shared OPES diagnostics at every selected time:

```json
{
  "analysis_profile": "water_ionization_opes",
  "source": {
    "sampling_label": "Bead-density frame mean",
    "sampling_slug": "bead_density",
    "sampling_colvar": "COLVAR.0",
    "bead_colvars": ["COLVAR.0", "COLVAR.1", "COLVAR.2", "COLVAR.3"],
    "expected_beads": 4
  },
  "reweight": {
    "bias_mode": "bead_density_shared",
    "weight_kind": "quasi_static_opes",
    "quasi_static": true,
    "bias_column": "opes.bias",
    "shared_diagnostic_tolerance": 1e-9
  }
}
```

The frame weight is `exp(beta * mean_b B(q_b))`.  Per-bead weights and
`mean_b exp(beta * B(q_b))` are different ensembles and are rejected by the
backend contract.  The frame-mean CV is emitted only as a diagnostic sampling
coordinate; the quantum target remains the weighted bead distribution.

The JSON boolean `quasi_static: true` is required for `quasi_static_opes`.
Missing values, `false`, strings, and numbers are rejected; legacy contracts
must explicitly declare this assumption before analysis. This declaration does
not prove that an adaptive OPES trajectory has reached that regime. With only
the primary bias declared, the OPES log weight is `+opes.bias / kBT`.
`opes.rct` is retained only as a diagnostic and is never subtracted from the
weight. The implemented quasi-static provider is not a general reconstruction
of an arbitrarily time-dependent bias history; establish a suitable analysis
window or provide separately audited frame weights for another protocol.

Raw time columns are converted explicitly before frame selection or alignment.
The conversion is data-source metadata, not an engine name heuristic:

```json
{
  "source": {
    "sampling_time_scale_to_fs": 0.00025,
    "bead_time_scale_to_fs": 0.00025,
    "kernel_time_scale_to_fs": 0.00025,
    "bead_time_offset_fs": 0.0
  }
}
```

The three scales multiply the corresponding raw `time` columns. They must be
finite and positive and default to `1.0` for existing femtosecond-based
contracts. `kernel_time_scale_to_fs` defaults to the sampling scale.
`bead_time_offset_fs` is applied after bead-time scaling and remains optional.
No missing frame is synthesized: converted timestamps must still map to one
distinct source frame.

The core `molsimflow.postprocess.pimd_fes.frame_log_weights` API also supports:

- `fixed_bias`, using `+total_bias_energy / kBT`;
- `precomputed`, accepting a separately audited log frame weight.

The command exposes the same providers.  For another enhanced-sampling method,
store one audited log weight per complete ring-polymer frame in the sampling
COLVAR and select it without asking molsimflow to guess a method-specific
formula:

```json
{
  "analysis_profile": "core",
  "reweight": {
    "weight_kind": "precomputed",
    "log_weight_column": "log_weight",
    "protocol_label": "WTMetaD reweighting"
  }
}
```

`protocol_label` is presentation metadata.  The supplied log weights remain
the scientific contract and must already include any method-specific offsets,
additional biases, or normalization terms.

## Complete-path weights and quantum estimator

Let `R[n,b]` be bead coordinates, `q[n,b] = Q(R[n,b])`, and
`beta = 1/(kB*T)` at the physical temperature. In the physical-temperature
ring-polymer convention used here, a shared bead-local bias contributes its
bead **average** to the complete-path energy. The energy to remove is:

| Bias mode | Sampling coordinate | Complete-path bias energy |
| --- | --- | --- |
| `centroid_coord` | `Q(mean_b R[n,b])` | `B(Q(mean_b R[n,b]))` |
| `bead_mean` | `mean_b q[n,b]` | `B(mean_b q[n,b])` |
| `bead_density_shared` | Each `q[n,b]` in one shared field | `mean_b B(q[n,b])` |

For fixed bias, or an explicitly selected quasi-static OPES window,

```text
ell_n = beta * U_remove[n]
W_n = exp(ell_n - logsumexp_m ell_m)
```

`U_remove` includes every declared energy term to remove. A bead-local wall
contributes its path average in this convention. A complete-path energy already
printed as an ENSEMBLE mean must not be divided by `P` a second time. The
centroid and bead-mean readers take their declared columns as complete-path
energies; the shared-density reader averages bead-local columns once.
Do not apply `exp(beta*B)` separately to each bead: beads share one path weight.

The common default estimator first averages the weighted bead probabilities:

```text
p_b(q) = sum_n W_n * K(q - q[n,b])
p(q) = mean_b p_b(q)
F_quantum(q) = -kBT * log p(q) + C
```

`q` can be one- or two-dimensional. `K` is the configured normalized KDE kernel;
histograms divide probability mass by bin width. In both cases each complete
frame supplies total weight `W_n`. No extra factor of `P` belongs in `beta` or
in the free-energy conversion. The finite-`P` result approximates the quantum
bead marginal; bead-number convergence still needs independent assessment.

A separate diagnostic averages the individual bead free energies:

```text
F_bead_mean(q) = mean_b [-kBT * log p_b(q)] + C
```

The logarithm and probability average do not commute. At finite sampling this
diagnostic differs from `F_quantum`; with a common additive constant, Jensen's
inequality gives `F_bead_mean >= F_quantum`. Agreement requires matching bead
marginals and adequate support, and alone does not establish convergence.
Both curves use the minimum of the probability-mean FES as their common zero;
independently shifting the diagnostic would hide this difference.

The sampling-coordinate FES is also exported, with its own minimum zero. It
characterizes the explicitly labelled sampling coordinate and should not be
identified with the quantum bead marginal. Cross-run sampling-coordinate
comparisons require matching observable identities. Quantum comparisons across
the three modes remain meaningful only for the same bead CV, physical
Hamiltonian, temperature, and coordinate measure.

### Support and diagnostics

The histogram API `quantum_fes_1d` preserves primary probability support even
when individual bead histograms do not overlap. `probability_support` marks
finite primary bins and `common_support` marks bins where both bead estimators
are finite. Zero-count bins remain infinite. KDE outputs have a separate
relative-density support mask for each estimator; the diagnostic intersection
must not erase supported primary quantum bins. Each plotted curve uses its own
mask; pairwise differences use the corresponding intersection.

Primary support, block-to-full comparisons and bandwidth sensitivity use
`probability_mean`. Neither a Gaussian KDE's positive tails nor support masks
prove adequate sampling. Frame-weight ESS, block sensitivity and independent
replica agreement address different limitations.

Log frame weights are normalized after removing their maximum, so changing a
finite energy zero does not change normalization. A conditioning bin whose
normalized mass underflows to zero contributes zero to the decomposition.

The core API provides a weighted conditional decomposition as an independent
finite-sample regression. Its histogram mass must agree with the direct route
up to floating-point rounding. This algebraic parity does not establish OPES
quasi-static behavior.

The weighted log-space 1D/2D KDE, complete-path weights, bead probability mean,
and bead free-energy-mean diagnostic are implemented inside molsimflow. A
contract may optionally provide `reference.driver` to run the historical
`FES_from_Reweighting.py` as a two-dimensional numerical cross-check. That
external driver is not a runtime dependency and is never the authoritative
estimator.

### Output schema and historical migration

New analyses use summary `schema_version: 2`. `fes.primary_estimator` is
`probability_mean`, `fes.target_observable` is `bead_marginal`, and
`fes.diagnostic_estimator` is `free_energy_mean`. One- and two-dimensional
FES CSV tables use the same names:

| Meaning | Support column | Free-energy column |
| --- | --- | --- |
| Sampling coordinate | `sampling_support` | `F_sampling_kcal_mol` |
| Quantum bead marginal | `probability_mean_support` | `F_quantum_probability_mean_kcal_mol` |
| Mean bead free energy, diagnostic | `free_energy_mean_support` | `F_bead_free_energy_mean_diagnostic_kcal_mol` |

The comparison command accepts these columns and adapts archived fields at the
input boundary. Historical `centroid` columns map to `sampling`; `eq8` maps to
`probability_mean`; `eq10` maps to `free_energy_mean`. The older core writer's
`logmean_support` and `F_bead_logmean_diagnostic_kcal_mol` also map to the last
row. Conflicting canonical and historical columns are rejected. New report
labels and summaries use estimator meanings rather than literature equation
numbers. The library's `quantum_fes_1d` keeps the old `eq8`, `eq10`,
`logmean_diagnostic` and `support` keys as compatibility aliases; new consumers
should use its descriptive keys.

Archived tables remain readable without rewriting their numeric values.
However, archived reports that promoted the mean bead free energy to the
primary quantum result must be regenerated to use the probability mean.
Historical diagnostics may also have been shifted to their own minima. The
column adapter preserves those values and cannot restore the shared zero;
recompute from the original inputs before comparing Jensen gaps or diagnostic
zero conventions. External scripts reading old CSV, summary, block or bandwidth
fields must migrate to schema 2 even though the comparison reader accepts old
analysis tables.
Comparisons read CV names from `contract.cvs` or
`summary.sampling_representation.logical_cv_names`; no water-specific axes are
assumed. Matching column names or bias modes do not establish matching CV
definitions: parameters, transformations and units may differ. Cross-run
sampling surface differences require explicit matching
`config.sampling_observable_id` values, or their recorded equivalents in
`summary.sampling_representation.sampling_observable_id`. Without matching
identities these differences are omitted. Analysis contracts can record this
identity as `source.sampling_observable_id`.

For the bead CV, comparison configs may declare `target_observable_id`;
analysis contracts can record `reweight.target_observable_id`, which is exported
as `summary.fes.target_observable_id`. Explicit matching values produce
`coordinate_definition_gate: DECLARED_COMPATIBLE`; missing identities produce
`NOT_VERIFIED`, and conflicting known identities are rejected. These IDs must
identify the coordinate definitions, parameter values, transformations, units
and target-measure conventions. They are user declarations, not numerical proof
of equivalence. Use consistent target Hamiltonians and temperatures separately.
No identity is inferred from a molecular-system label or a familiar CV name.


## Input integrity

`source.raw_manifest` must be a GNU SHA256SUMS text manifest, with paths
relative to `source.run_root` (absolute paths are also accepted). Text and
binary markers, spaces in names, and GNU escaped filenames are supported.
`source.raw_manifest_sha256` authenticates the manifest against the supplied
contract; it is not an independent signature or trust anchor.

Before analysis, each consumed source-data file must be listed and match its
SHA-256 digest. The core profile checks the sampling and bead COLVAR files.
The diagnostic profile also checks its KERNELS, thermo logs and trajectories.
Missing entries, conflicting or duplicate resolved paths, malformed records,
and changed data are rejected. Unconsumed entries are parsed but their files
are not read or required to be present. Identical input paths shared by
sampling and one bead are hashed once.

The output `provenance/verified-inputs.json` records the verified paths and
digests. This preflight reads the consumed files once for hashing in addition
to analysis reads, so large trajectories incur an extra sequential I/O pass.
Use immutable, completed inputs: preflight verification does not lock files
or protect against changes made during analysis. External reference-driver
identity is recorded separately by the reference cross-check.

Existing contracts with complete SHA256SUMS manifests need no new option.
Placeholder manifests or manifests missing consumed inputs must be replaced
with real checksums and their contract hash updated before use. A successful
manifest check does not establish scientific validity.

## Fail-closed checks

The implementation rejects:

- a frame with a missing or duplicate bead;
- repeated bead input files, including relative-path aliases, symlinks, and hard links;
- a missing or misaligned frame between sampling and bead tables;
- an ambiguous timestamp match or reuse of one source frame for multiple target frames;
- a non-finite or non-positive raw-time conversion scale;
- decreasing frame IDs across a restart seam;
- duplicate restart frames when the policy is `error`;
- non-finite CVs, energies, or weights;
- an undeclared adaptive OPES weight;
- unsupported bias modes, primary estimators, or energy units;
- inconsistent shared `rct`, `zed`, `neff`, or `nker` values across beads.

With `restart_duplicate_policy: keep_first`, the predecessor endpoint is kept
and the repeated successor endpoint is removed.  The number of removed rows is
stored in the summary.

## Numerical regression coverage

The core-profile tests compare exported 1D and 2D FES tables with closed-form
Gaussian mixtures from a biased three-state sample. The 2D cases use unequal
grid sizes and bandwidths and cover all three bias modes with precomputed
frame weights. These cases test estimator and export arithmetic; they do not
validate the dynamics that generated a production trajectory.

Reported weight ESS is `1 / sum_n W_n^2`, calculated over complete frames.
Block diagnostics renormalize weights within each block. This weight ESS does
not account for temporal autocorrelation, and block-to-full FES differences
are diagnostics, not confidence intervals. Correlation-aware uncertainty and
independent-replica convergence require separate assessment.

## Scientific boundary

A successful command establishes input alignment, estimator arithmetic, and
artifact production.  It does not establish bead-number convergence, time-step
convergence, bias convergence, adequate overlap, physical interpretation, or a
scientifically converged quantum FES.  Beads from one frame are correlated and
must not be counted as independent samples for uncertainty estimates.

For adaptive shared bead-density OPES, `quasi_static: true` remains an explicit
analysis assumption.  A successful reconstruction does not prove that the
time-dependent shared field is quasi-static or that the resulting FES has
converged.

Timestamp alignment accepts roundoff on either side of a source timestamp,
including endpoints, within a finite nonnegative tolerance (default `1e-8` in
the supplied time units). Every target must match exactly one distinct source
frame. Missing or ambiguous matches are rejected; no interpolation is performed.

## Block jackknife for one-dimensional histogram FES

The library API `pimd_fes.quantum_fes_block_jackknife_1d` estimates standard
errors of probability-mean FES differences relative to an explicitly selected
`reference_bin`. It accepts the same complete-frame bead CVs, log weights and
bin edges as `quantum_fes_1d`, plus `block_size` in frames.

All beads remain together when a contiguous block is deleted. Blocks must be
equal-sized and cover every frame; no tail is silently discarded. Each
leave-one-block-out estimate renormalizes its retained weights independently,
including when the deleted block carried nearly all the original weight.
Unequal bin widths are included in the density ratio.

For B deleted-block estimates theta_b, the reported standard error is
sqrt((B-1)/B * sum_b (theta_b - mean(theta))^2). The point estimate remains
the full-data FES difference. A fixed reference avoids changing the free-energy
zero separately in every replicate. If the reference loses support on any
deletion, estimation fails. Other bins losing support have a false support
mask and NaN standard error, never an artificial zero.

Choose blocks longer than the relevant correlation scale and examine stability
over block lengths while retaining enough blocks. The mathematical minimum
of two blocks is not evidence of a reliable error estimate. This function
does not determine correlation times, certify independence or generate
confidence intervals. Duplicating beads cannot increase statistical sample
size. Adaptive-bias weighting assumptions still require separate validation.

The current implementation recomputes each deleted-block histogram for stable
normalization, costing O(B*N*P) for B blocks, N frames and P beads. It is an
explicit 1D histogram API; it is not yet wired into the CLI KDE reports or
the 2D estimator. Existing CLI block diagnostics retain their original meaning.


### Statistical regression coverage

A fixed-seed ensemble regression uses independent Bernoulli draws repeated
eight times to create exactly known within-cluster correlation. Both uniform
weights and state-dependent importance weights are checked against the
analytic free-energy difference and its large-sample delta-method variance.
The independent sample count is the number of original draws, not repeated
frames. Ensemble variance, mean jackknife variance and their ratio are checked
with finite-ensemble tolerances; this is not an exact finite-sample identity.

This provides calibration for a controlled stationary two-state model.
It does not validate arbitrary correlation tails, adaptive bias histories,
rare-event support, KDE bandwidth bias or confidence-interval coverage.
Users still need block-length sensitivity and independent-replica comparisons.

## Optional KDE sampling uncertainty in analysis reports

Add an `uncertainty` object under `reweight` to enable whole-frame block
jackknife for the primary probability-mean KDE FES:

```json
"uncertainty": {
  "block_frames": 100,
  "reference_grid_index": [20, 15]
}
```

Indices are zero-based and follow `cv_names` order; use one index for one CV.
The example numbers are illustrative, not recommended block lengths.
Equal blocks must cover all selected frames and leave at least two blocks.
Choose an interior reference supported in the full data and every deletion;
an unsupported reference is an error. Other points losing relative density
support receive NaN standard errors and `support=0`.

The primary bandwidth and grid remain fixed across deletions. All beads of a
frame share its weight and remain together. Weights are normalized afresh for
each retained sample. The API is `quantum_kde_block_jackknife`; it supports
one or two CVs, using the existing two-dimensional (y, x) array layout.

Outputs are `blocks/quantum-fes-uncertainty.csv` and its JSON metadata.
CSV columns contain the CV coordinates, `delta_F_eV`,
`standard_error_eV`, and `support`. Free-energy differences use the fixed
reference, so their zero generally differs from the minimum-zero plotting
tables. JSON records block count, reference coordinates, bandwidth, support
threshold and units. Existing block-difference diagnostics are unchanged.

This is fixed-bandwidth sampling uncertainty, not KDE smoothing bias, a
confidence interval, autocorrelation analysis, or independent-replica validation.
Positive Gaussian density and relative-density support do not prove adequate
rare-event sampling. Compare block lengths and independent replicas separately.
Cost grows with the number of deletion blocks, frames, beads and grid points;
this optional calculation is disabled unless explicitly configured.

### Correlated-process regression and practical block choice

A stationary symmetric two-state Markov-chain regression supplements the
independent-draw calibration. Its autocorrelation is exactly rho^lag, and the
finite-length variance of the state fraction is available by summing that
covariance. The weighted two-state Gaussian KDE density ratio gives an
independent delta-method reference for the FES variance. The test compares
96 independent chains and two block lengths using a fixed random seed;
the shorter blocks must reveal underestimation, while the longer-block
variance must agree with theory and across-chain dispersion within declared
finite-ensemble tolerances.

For an actual analysis, compare several block lengths with sufficient retained
blocks, examine stability of uncertainties at the same reference and supported
coordinates, and compare independent simulation replicas. Do not concatenate
replicas across a deletion block. Record the selected frame spacing as well as
block_frames. A plateau over a narrow range is not proof that slow modes or
rare transitions have been sampled; the regression's block sizes and
tolerances are not automatic production-admission thresholds.

# Bead-count convergence

Use one or more named LAMMPS PIMD estimators to compare bead counts without
fixed paths or fixed column numbers. The standard energy and pressure check is:

- `f_pi[5]`: primitive kinetic-energy estimator (eV)
- `f_pi[6]`: virial energy estimator (eV)
- `f_pi[7]`: centroid-virial energy estimator (eV)
- `f_pi[10]`: centroid-virial pressure estimator (bar)

Prepare a CSV manifest whose paths are absolute or relative to the manifest:

```csv
label,beads,log
P16,16,run/p16/log.lammps.0
P32,32,run/p32/log.lammps.0
P36,36,run/p36/log.lammps.0
```

Then run:

```bash
molsimflow postprocess pimd-bead-convergence \
  --manifest cases.csv --output convergence-v1 \
  --field 'f_pi[5]' --field 'f_pi[6]' --field 'f_pi[7]' --field 'f_pi[10]' \
  --burn-in-ps 2 --blocks 5 --write-plot
```

The command writes long-form summaries, per-estimator comparisons to the largest
bead count, and one multi-panel plot labelled with physical quantities and
units. A single `--field` also writes the legacy `bead_summary.csv` and
`reference_comparison.csv` files. `within_sigma` means only that the sampled
means are not distinguishable at the requested block-error threshold; it is not
proof of equilibrium or scientific convergence.


## Coordinate joins and additional bias energies

The piecewise log-distance helpers validate the join before inversion or
Jacobian conversion. The archived defaults remain `switch=1`, `offset=0.03`,
`linear_shift=0.9704412`, `log_scale=1`, `log_reference=1`; the eight-digit
historical shift has a small rounding discrepancy (accepted up to 5e-8).
Other joins must satisfy
`log_scale*log((switch+offset)/log_reference) = switch-linear_shift`.
Printed-coordinate validation checks both forward and inverse errors against
the requested tolerance. It rejects an offset change that retains an
incompatible old shift, even if all sampled points avoid the join.

For the C1 map `1.1*log((x+0.1)/1.1)` below 1 and `x-1` above it, specify
`offset=0.1`, `linear_shift=1`, `log_scale=1.1`, `log_reference=1.1` explicitly
in the optional `derived_coordinate` contract. This coordinate differs from
`log(x+0.1)` with a continuity-only shift. Do not reuse a history under a changed
coordinate. Direct KDE of a printed distance does not invoke this map.
For nonlinear maps, transforming the bead mean is not the same operation as
averaging transformed bead coordinates; do not declare them interchangeable.

By default, reweighting removes only `reweight.bias_column`. To remove a wall
as well, declare `"extra_bias_columns": ["iwall.bias"]`. Each named energy is
summed before exponentiation. Centroid and bead-mean modes require complete-path
energies (for a bead-local wall in bead-mean mode, print its ENSEMBLE mean).
Shared bead-density mode uses the bead average of the sum of local energies.
Duplicate columns and combinations with precomputed weights are rejected.
Summary metadata records the selected columns; the OPES-only diagnostic colors
continue to use the primary bias. A restraint intentionally retained defines
part of the target Hamiltonian and must not be included in `U_remove`.
Conversely, obtaining a wall-free target requires declaring the wall energy in
addition to the enhanced-sampling bias. Removing a wall by reweighting cannot
recover regions the restrained simulation never visited. Undeclared biases
remain in the target ensemble; this choice must be consistent across compared
methods.

Correct postprocessing weights do not certify online OPES deposition weights or
time-dependent equilibration. Ordinary WALKERS_MPI uses local walker weights.
A synchronized shared-path probability estimator needs the same complete-path
weight on every bead at a deposition frame. Its kernel count is not independent
frame ESS. Existing local-weight histories must retain their method label.


### Complete-path inputs and stable diagnostics

Set `source.expected_beads` to the simulated bead count, as a positive integer.
The reader checks this against the distinct bead files before processing data;
this catches a file omitted from a contract or glob. There is no fixed bead
count. The field is optional for archived contracts, but without it the input
list itself defines the path size and cannot prove that the path is complete.
The water-ionization profile also requires one thermo log and one trajectory
per listed bead. Frame alignment and duplicate-file checks apply separately.

KDE inputs require one finite log weight per frame and finite, positive
bandwidths. A single weight is not broadcast to multiple frames. OPES plots
calculate cumulative ESS in log space, so full-trajectory normalization cannot
turn early finite weights into zero and abort the report. Beads within a frame
still share one path weight and do not increase the independent frame count.
