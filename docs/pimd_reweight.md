# PIMD Quantum-FES Reweighting

`molsimflow postprocess pimd-reweight` reconstructs bead-defined quantum free
energies from a complete ring-polymer trajectory.  The workflow supports three
explicit path-CV bias modes:

- `centroid_coord`: the bias is evaluated on a CV of the Cartesian centroid;
- `bead_mean`: the bias is evaluated on the arithmetic mean of the bead CVs;
- `bead_density_shared`: one shared field is evaluated on every bead and the
  complete-path bias energy is `mean_b B(q_b)`.

The target observable is the bead distribution, not the sampled path CV.  For
frame `n`, every bead uses the same normalized frame weight `W_n`, and each bead
contributes `W_n / P` to the target distribution.

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
semantics should be declared explicitly:

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
    "weight_kind": "quasi_static_opes",
    "quasi_static": true,
    "bias_column": "opes.bias",
    "rct_column": "opes.rct",
    "sampling_cv_names": ["mean.cv1", "mean.cv2"],
    "bead_cv_names": ["cv1", "cv2"]
  }
}
```

For shared bead density, each bead COLVAR must contain the local field value
and identical shared OPES diagnostics at every selected time:

```json
{
  "analysis_profile": "water_ionization_opes",
  "source": {
    "sampling_label": "Bead-density frame mean",
    "sampling_slug": "bead_density",
    "sampling_colvar": "COLVAR.0",
    "bead_colvars": ["COLVAR.0", "COLVAR.1", "COLVAR.2", "COLVAR.3"]
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
not prove that an adaptive OPES trajectory has reached that regime.  The OPES log weight is
`+opes.bias / kBT`.  `opes.rct` is retained only as a diagnostic and is never
subtracted from the weight.

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

## Estimators

For bead CV values `q[n, b]`, the primary probability-mean estimator is:

```text
p(q) = sum_n W_n * mean_b K(q - q[n, b])
F_q(q) = -kBT log p(q) + C
```

The second output averages the individual bead free energies and is reported as
a same-zero bead-logmean finite-sampling diagnostic.  It is not a replacement
for convergence or overlap checks.  These two bead aggregations are available
after valid full-path reweighting for all three bias modes.  Plot labels use
`Quantum FES` and `Bead-logmean diagnostic`; only `centroid_coord` appends the
literature identifiers `(Lamaire Eq. 8)` and `(Lamaire Eq. 10)`.

The complete-path energy, not the final bead aggregation, distinguishes the
three modes:

```text
centroid_coord:       U = B(Q(R_centroid))
bead_mean:            U = B(mean_b q_b)
bead_density_shared:  U = mean_b B(q_b)
```

The histogram API `quantum_fes_1d` preserves primary probability support even
when individual bead histograms do not overlap. Its `probability_support` mask
marks finite primary bins; the backward-compatible `support` mask marks bins
where both estimators are finite. Zero-count bins stay infinite, and both
curves use the primary minimum as their common zero. An empty primary support
is rejected. KDE plots have their own density-support masks.

Log frame weights are normalized after removing their maximum, so changing a
finite energy zero does not change normalization. A conditioning bin whose
normalized mass underflows to zero contributes zero to the decomposition.

The core API provides a weighted conditional decomposition as an independent
finite-sample regression.  Its histogram mass must agree with the direct route
up to floating-point rounding.  This algebraic parity does not establish OPES
quasi-static behavior.

The weighted log-space 1D/2D KDE, full-path weights, bead probability mean, and
bead-logmean diagnostic are implemented inside molsimflow.  A contract may
optionally provide `reference.driver` to run the historical
`FES_from_Reweighting.py` as a two-dimensional numerical cross-check.  That
external driver is not a runtime dependency and is never the authoritative
estimator.


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
- decreasing frame IDs across a restart seam;
- duplicate restart frames when the policy is `error`;
- non-finite CVs, energies, or weights;
- an undeclared adaptive OPES weight;
- unsupported bias modes;
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
