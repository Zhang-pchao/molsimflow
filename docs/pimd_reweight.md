# PIMD Quantum-FES Reweighting

`molsimflow postprocess pimd-reweight` reconstructs bead-defined quantum free
energies from a complete ring-polymer trajectory.  The workflow supports the
two path-CV bias modes that have an implemented runtime contract:

- `centroid_coord`: the bias is evaluated on a CV of the Cartesian centroid;
- `bead_mean`: the bias is evaluated on the arithmetic mean of the bead CVs.

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
bead tables, one- and two-dimensional FES tables, CV/bias and PIMD diagnostic
figures, block diagnostics, a machine-readable summary, and SHA-256 provenance.

## Contract fields

All paths and system-specific column names belong in the JSON contract.  The
reusable code contains no cluster or case paths.  The representation and weight
semantics should be declared explicitly:

```json
{
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

`quasi_static: true` records an analysis assumption; it does not prove that an
adaptive OPES trajectory has reached that regime.  The OPES log weight is
`+opes.bias / kBT`.  `opes.rct` is retained only as a diagnostic and is never
subtracted from the weight.

The core `molsimflow.postprocess.pimd_fes.frame_log_weights` API also supports:

- `fixed_bias`, using `+total_bias_energy / kBT`;
- `precomputed`, accepting a separately audited log frame weight.

The full diagnostic workflow currently expects the OPES and PIMD columns listed
by its contract.  Use the core API for a fixed-bias estimator that does not have
OPES diagnostics.

## Estimators

For bead CV values `q[n, b]`, Eq. 8 is the primary estimator:

```text
p(q) = sum_n W_n * mean_b K(q - q[n, b])
F_8(q) = -kBT log p(q) + C
```

Eq. 10 averages the individual bead free energies and is reported as a
same-zero finite-sampling diagnostic.  It is not a replacement for convergence
or overlap checks.

The core API provides a weighted conditional decomposition as an independent
finite-sample regression.  Its histogram mass must agree with the direct route
up to floating-point rounding.  This algebraic parity does not establish OPES
quasi-static behavior.

## Fail-closed checks

The implementation rejects:

- a frame with a missing or duplicate bead;
- a missing or misaligned frame between sampling and bead tables;
- decreasing frame IDs across a restart seam;
- duplicate restart frames when the policy is `error`;
- non-finite CVs, energies, or weights;
- an undeclared adaptive OPES weight;
- unsupported bias modes.

With `restart_duplicate_policy: keep_first`, the predecessor endpoint is kept
and the repeated successor endpoint is removed.  The number of removed rows is
stored in the summary.

## Scientific boundary

A successful command establishes input alignment, estimator arithmetic, and
artifact production.  It does not establish bead-number convergence, time-step
convergence, bias convergence, adequate overlap, physical interpretation, or a
scientifically converged quantum FES.  Beads from one frame are correlated and
must not be counted as independent samples for uncertainty estimates.

Shared bead-density bias is intentionally outside this first public contract.
It requires a distinct total-bias and adaptive-history validation.
