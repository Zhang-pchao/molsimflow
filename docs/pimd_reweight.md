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

`quasi_static: true` records an analysis assumption; it does not prove that an
adaptive OPES trajectory has reached that regime.  The OPES log weight is
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

## Fail-closed checks

The implementation rejects:

- a frame with a missing or duplicate bead;
- a missing or misaligned frame between sampling and bead tables;
- decreasing frame IDs across a restart seam;
- duplicate restart frames when the policy is `error`;
- non-finite CVs, energies, or weights;
- an undeclared adaptive OPES weight;
- unsupported bias modes;
- inconsistent shared `rct`, `zed`, `neff`, or `nker` values across beads.

With `restart_duplicate_policy: keep_first`, the predecessor endpoint is kept
and the repeated successor endpoint is removed.  The number of removed rows is
stored in the summary.

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
