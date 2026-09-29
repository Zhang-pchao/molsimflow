# Partial path contraction

The optional coupling maps a complete, consistently lifted Cartesian path to
`R_virtual = mean(R_real) + lambda * (R_real - mean(R_real))`, with fixed
`0 <= lambda <= 1`. All physical beads and PES evaluations remain unchanged.
The bias is one scalar `B(mean(CV(R_virtual)))`. The endpoint potentials are
centroid and bead-mean on the same coordinate lift. This is not a physical-PES
reduced-bead approximation and does not promise an efficiency improvement.

`contract_coordinates` and `pullback_bias_forces` in
`molsimflow.postprocess.path_contraction` are independent NumPy reference
helpers. Input shape has beads on axis zero, followed by atom/component axes.
Callers must establish a consistent periodic lift and atom identity first.
They do not add the MD engine factor P, infer images, or mutate input arrays.

The force pullback is `lambda*f_virtual + (1-lambda)*mean(f_virtual)`, where
`f_virtual` is already the derivative of the single physical path bias. Apply
it only to the bias increment. Do not smooth the physical force.

## Existing FES workflow integration

Use `bias_mode` and `sampling_slug` equal to `contracted_bead_mean`, an explicit
sampling label, a stationary `fixed_bias` record and the actual complete-path
bias column. Declare the original real-bead observable columns with
`bead_cv_names`. The reweight section also requires:

```json
"path_contraction": {
  "lambda": 0.5,
  "coordinate_lift": "pimd_unwrapped",
  "observable_coordinates": "real_beads"
}
```

This metadata is copied to `qc/path-contraction.json`; metadata validation is
not proof of energy reconstruction or sufficient sampling. The data provider
must verify the actual applied total bias and generate real-bead CVs from the
original trajectory. Virtual-coordinate PLUMED output must not be relabelled
as real-bead observations. The generic estimator reuses one `exp(beta*V)`
weight per complete frame, shared by all its correlated beads. No coordinate
Jacobian or extra lambda factor is appropriate. Adaptive OPES reweighting is
not admitted through this new record type.
