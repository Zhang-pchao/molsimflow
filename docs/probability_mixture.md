# Frozen probability-mixture path biases

`molsimflow.postprocess.probability_mixture` supplies a deterministic oracle,
frozen-manifest validation, record auditing and PLUMED graph composition.
The opt-in modes are `bead_probability_mixture` and
`centroid_probability_mixture`. Existing modes retain their semantics.

For a common frozen energy field `v(s)`, define `ell_b=-v(s_b)/kBT` and
`V_A=-kBT log(mean_b exp(ell_b))`. The force coefficient is the bead softmax
of `ell`, with no extra `1/P`. For OPES this averages the effective
target/unbiased ratio, including OPES regularization and its existing bias
factor exactly once. It does not average the estimated unbiased density.

The optional centroid mixture is
`V_eta=-kBT log[(1-eta)exp(-V_c/kBT)+eta exp(-V_A/kBT)/C]`.
`C=Z_A/Z_c` is a frozen global constant. An estimated C gives a conservative
potential, but the effective mixture fraction is uncertain. C is not the
coordinate-dependent conditional normalizer of `conditional_path`.

## Runtime and records

Use identical frozen fields on every bead and the PLUMED `PATH_LOGMEANEXP`
action. The existing LAMMPS `bead_mean` adapter converts physical path forces
to the integrator convention and reports the energy once per path. Do not
select `bead_density`, which would introduce an extra scaling.

`export_plumed` composes already defined scalar fields. For active frozen
OPES, set `active_field_bias=True` to apply `V_path-v_b`, cancelling its
direct local bias. Verify native update suppression and identical immutable
STATE files: a large deposition pace does not freeze a learner. A centroid
input must be a non-bias scalar. At eta=0 generate the centroid-only graph
and omit all unused field actions. Online OPES is not supported by this API.

The manifest schema is `probability-mixture-v1`, with `frozen: true`, `mode`,
positive integer `expected_beads`, positive `kbt`, explicit `energy_unit`,
and SHA256 `field_sha256`. Mixed mode also requires `coupling` in [0,1),
finite `log_normalizer`, and `centroid_field_sha256`. Field hashes identify
the actual immutable field/state files, including parameters and units in
the surrounding source manifest. A bare hash cannot prove identical runtime
evaluation; installed force/state checks are a separate gate.

The existing one-dimensional core FES workflow accepts these modes only
with `weight_kind: fixed_bias`, one complete-path `bias_column`, no extra
bias columns, and an explicit matching source sampling slug and label.
`reweight.probability_mixture` requires `manifest_file`, `manifest_sha256`,
`field_file`, `bead_bias_column`, and `energy_atol_eV`. Mixed mode also supplies
`centroid_field_file` and `centroid_bias_column`. At eta=0 the inactive
`field_file` and `bead_bias_column` are not consumed; the source bead
coordinate records must still contain complete paths. Include all these input
files in the existing raw SHA256 manifest. The audit checks field identity,
bead count, kBT, energy units and total energy before reporting FES.

Use one frame weight `exp(V_path/kBT)` and the uniform bead observable.
Softmax coefficients distribute forces; they do not replace uniform bead
weights in the physical observable. Do not subtract an arbitrary OPES rct.
Blocks and independent replicas, not correlated beads, determine uncertainty.
The first workflow is single stationary state; multi-window normalization
requires an explicit cross-state contract. Passing engineering tests does
not establish convergence, target-region support or improved efficiency.
