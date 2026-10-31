# Fixed-bias umbrella PMF analysis

`molsimflow.postprocess.umbrella_wham` implements binless WHAM for pooled
same-temperature umbrella samples. It reuses the strict PLUMED table reader in
`fes_analysis`. It is separate from adaptive-bias reweighting: one cannot obtain
a multi-window PMF by pooling `exp(beta * bias)` weights without solving window
normalizations.

## Statistical contract

All windows must share the same underlying Hamiltonian, atom identity,
constraints, and temperature. Every differing bias potential must be included.
The harmonic wrapper assumes the only differing bias is

```
U_k(q) = 0.5 * kappa_k * (q - center_k)**2
```

Coordinates, kappa, and kBT must use compatible units. Examples below use
angstrom and kJ/mol. Kappa acts on the supplied collective coordinate, not on
each constituent atom independently. Neither a radial Jacobian nor a per-atom
normalization is applied. A planar Cartesian height coordinate has no automatic
`4*pi*q**2` correction.

Given reduced biases `u_kn = U_k(q_n)/kBT` and retained sample counts `N_k`, solve

```
D_n = sum_k N_k * exp(f_k - u_kn)
f_k = -log(sum_n exp(-u_kn) / D_n)       (up to a common constant)
w_n proportional to 1 / D_n
```

The implementation minimizes the convex equivalent likelihood with a log-space
warmup and damped Newton iterations, fixing `f[0]=0`. It reports a normalized
score residual and numerical convergence, not thermodynamic equilibration.
Actual counts are used, including unequal lengths after filtering/subsampling.
The state-overlap matrix is `O_ij = N_j * sum_n W_ni * W_nj` with
`W_ni = exp(f_i-u_in)/D_n`; rows sum to one at convergence. This is a model-based
state-overlap diagnostic, distinct from an observed adjacent-histogram overlap.

Histogram probabilities use the normalized frame weights and bin widths. Zero
support remains zero density / infinite PMF, and the requested reference bin
must have observations. There is no smoothing or tail extrapolation.
`weight_ess_iid` measures weight concentration only; it ignores time correlation
and must not be presented as an independent-sample count or a PMF error bar.

## Python use

```python
import numpy as np
from molsimflow.postprocess.umbrella_wham import (
    fit_harmonic_umbrellas,
    histogram_pmf,
    load_window,
)

centers = [1.0, 2.0, 3.0]
kappas = [2.0, 2.0, 2.0]
samples = [
    load_window(path, center, kappa, discard_time=10, end_time=110)[1]
    for path, center, kappa in zip(paths, centers, kappas)
]
kbt = 0.00831446261815324 * 330
result = fit_harmonic_umbrellas(samples, centers, kappas, kbt)
if not result.converged:
    raise RuntimeError(result.score_residual)
profile = histogram_pmf(
    np.concatenate(samples), result.log_weights, edges, kbt,
    reference_coordinate=2.0,
)
```

`load_window` retains `discard_time < time <= end_time`, rejects malformed or
nonfinite PLUMED input, requires increasing selected timestamps, and checks the
recorded bias against the supplied harmonic potential. Duplicate production
times are rejected; they are not silently dropped. The bias field defaults to
`umb.bias`, and the CV field to `dz.z`; both can be overridden. Presence of
additional unmodeled biases remains the caller's responsibility.

The lower-level `solve_wham(reduced_biases, sample_counts)` supports arbitrary
known fixed biases. Its matrix has shape `(windows, pooled samples)`.

## Manifest entry point

Create a portable JSON file with relative COLVAR paths:

```json
{
  "kbt": 2.743772664,
  "discard_time": 10,
  "end_time": 110,
  "cv_field": "dz.z",
  "bias_field": "umb.bias",
  "windows": [
    {"colvar": "window_0/COLVAR", "center": 1.0, "kappa": 2.0},
    {"colvar": "window_1/COLVAR", "center": 2.0, "kappa": 2.0}
  ]
}
```

```
python -m molsimflow.postprocess.umbrella_wham manifest.json \
  --output pmf_output --bin-width 0.2 --reference 2.0
```

The output directory must be new. Outputs are `pmf.csv`, `overlap.csv`, and
`solver.json`. The API permits explicit bin edges when a particular reference
center or shared grid is required across temporal subsets.

## Validation and interpretation

The focused tests cover an analytic Gaussian reference PMF and normalization
constants, unequal counts, energy-unit rescaling, large common bias shifts,
single-window bias removal, empty support, nonuniform bins, invalid inputs,
reported nonconvergence, and strict production selection/bias checking.

Before interpreting a PMF, inspect actual sampled CV ranges, overlap, drift,
autocorrelation, independent blocks/replicas and finite-size boundaries. A
numerically converged WHAM solution can fit biased nonstationary data precisely
and still yield an invalid equilibrium PMF. Temporal-subset curves are sensitivity
checks, not confidence intervals. A PMF is defined up to a constant; a detachment
free energy additionally needs a justified reference state, sufficient range,
and appropriate standard-state treatment.

## Migration notes

The legacy pattern `bias matrix -> WHAM weights -> weighted histogram` is
preserved. This module adds explicit sample counts, vectorized log-space
arithmetic, convergence reporting, input/bias validation and support-preserving
histograms. It does not adopt the legacy quick independent-frame error proxy,
a default thinning factor, hidden wall parameters or a hard-coded CV range.
Legacy-source comparisons belong in consuming-project records; no private
trajectory, model, host path or case result belongs in this repository.
