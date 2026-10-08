# Frozen conditional path bias

This experimental one-dimensional workflow prepares a fixed path potential,
not a new physical potential or a claim of improved sampling. It reuses
`quantum_path_stats` for conditional diagnostics and `pimd_fes` for whole-frame
weights. The online correction requires PLUMED's optional `pathbias` module.

For a positive score `a=epsilon+mean_b(h_b)`, a frozen normalizer `m(c)` and
`0 <= lambda < 1`, the total physical bias is

```text
V = B_c(c) - kBT*log[(1-lambda)+lambda*a/m(c)]
log(w) = V/kBT
```

`c` is the declared centroid CV, and each `h_b` lies in `[0,1]`. The sampled
object is the complete path. Differentiating the total bias must include
both the score derivative and the derivative of the frozen normalizer.
Changing the normalizer changes the potential and requires re-equilibration.

## Fit and freeze

Prepare a stationary pilot NPZ with `conditioning`, `region_fraction`,
`log_weights`, and `block_ids` of shape `(N,)`, plus `bin_edges` of shape
`(K+1,)`. `region_fraction` is the smooth score averaged over all beads of a
frame. `block_ids` describe contiguous, disjoint, equal-sized whole-frame
blocks. Explicit log weights must target the desired conditional law.
A stationary bias depending only on the exact conditioning coordinate leaves
its conditional law unchanged, but finite conditioning bins still introduce
approximation. Do not use adaptive production or shuffle beads into train/test
sets. Hold out independent runs or whole contiguous blocks for validation.

```bash
python -m molsimflow.postprocess.conditional_path fit \
  --input pilot.npz --output normalizer.json --epsilon 0.1 \
  --degree 2 --min-frame-ess 20 --stationary-pilot
```

The numerical values above are examples, not sampling thresholds. The command
prints the new model SHA256 and refuses to overwrite an existing file.
Every cell must retain whole-block support and the declared minimum frame
Kish ESS. Fit log conditional bin means using a degree-0-to-5 polynomial on a
scaled coordinate. The result is smooth and positive after exponentiation.
Its fit diagnostics, finite domain and provenance are retained in the model.
Kish ESS is not a time-correlation correction or evidence of independence.
No held-out scientific qualification is inferred by the fitter.

```bash
python -m molsimflow.postprocess.conditional_path export \
  --model normalizer.json --model-sha256 "$normalizer_sha256" \
  --centroid-label c --score-label a --prefix conditional \
  --coupling 0.5 --kbt 0.02585 --output correction.dat
```

Use the actual physical `kBT` in the energy units of the simulation. Define
`c`, the positive score `a` and the fixed `B_c` separately in PLUMED before
including this fragment. It uses CUSTOM for log-polynomial evaluation and
its exact derivative, CONDITIONAL_PATH for stable log-mixture evaluation and
domain checking, and BIASVALUE for the correction energy. The new function
fails outside the frozen model domain. For the one-dimensional component
construction in LAMMPS PIMD use the existing `path_integral bead_mean` adapter;
this interface name does not relabel the statistical method as bead-mean bias.
Average Cartesian components before computing nonlinear centroid functions.
This is not a general automatic Cartesian-centroid molecular CV interface.

Verify the same model and complete input hashes before restart preparation.
`load_model(path, expected_sha256=...)` rejects changed model bytes. The online
function is stateless and does not itself serialize or enforce external model
identity. Freeze all parameters and bias fields through production.

## Analyze an admitted fixed-bias record

The existing core reweighting workflow accepts explicit
`bias_mode: centroid_conditioned`, `weight_kind: fixed_bias`, a single total
`bias_column`, and explicit source `sampling_slug: centroid_conditioned` and
`sampling_label`. Use a one-dimensional core profile. The `conditional_path`
object under `reweight` must contain:

```json
{
  "frozen": true,
  "model_file": "normalizer.json",
  "model_sha256": "<SHA256 from fit>",
  "coupling": 0.5,
  "bead_region_column": "h",
  "log_normalizer_column": "conditional_logm",
  "centroid_bias_column": "base.bias",
  "energy_atol_eV": 1e-9,
  "log_normalizer_atol": 1e-10
}
```

As in the existing core workflow, declare temperature, physical `kbt_eV`,
units, time alignment, complete bead files, selection, grids and plotting
parameters. Include the model in the raw-input checksum manifest. `model_file`
is relative to the run root. The workflow reconstructs `a` from complete bead
rows, verifies printed log normalizers against the hashed model, and checks
recorded total bias against its components before computing any weights.
The audit is written to `qc/conditional-path.json`. Extra bias columns are
rejected for this route: include all fixed terms once in the audited total
and corresponding baseline column. Defaults intended for adaptive OPES are
not valid here.

`probability_mean` remains the primary bead-marginal FES estimator.
Independent block/replica, local support and time-stability diagnostics remain
necessary. The library algebra works with any explicitly frozen potential;
exact centroid-marginal preservation requires the exact conditional normalizer.

Multiple independently normalized windows require a validated multistate
estimator, evaluating all states on all frames. Do not concatenate single-state
`exp(beta*V)` weights. The new fixed-record CLI does not combine windows or
learn online OPES. Existing project MBAR analyses remain separate until their
state matrices and input/model identities have been explicitly audited.
