# Offline quantum-path descriptors

`molsimflow.postprocess.quantum_path` computes geometric path moments for
complete, compact ring polymers in fully periodic orthorhombic cells. It uses
NumPy and existing soft-Voronoi helpers. It does not apply bias forces,
estimate uncertainty from beads, infer equilibrium or compute reaction rates.

## Input and geometry

`describe_quantum_path(positions, types, box, *, center_type, assigned_type,
kappa, distance_kappa, reference, environment_r0)` takes positions with shape
`(beads, atoms, 3)`, matching integer atom types, and three box lengths.
Coordinates, box and `environment_r0` must use one length unit; the two kappa
values have its inverse unit. At least two centers and one assigned atom are
required. Every center uses the same supplied reference occupation.

The compact-path convention aligns each atom to its image in bead zero and
requires its span along every box axis to be strictly less than half the box.
Extended/ambiguous paths, invalid geometry and coincident center-assigned or
center-center pairs fail explicitly. This is a stated supported domain, not
a general triclinic or winding-path treatment. It may reject valid extended
paths rather than choosing an ambiguous centroid. Integer image shifts do
not change periodic observables. Centroid coordinates themselves retain the
images of bead zero.

## Outputs

The smooth assignment uses `exp(-kappa * distance)`, normalized over centers
for each assigned atom. Site defects are occupancy minus `reference`.

| Keys | Definition |
|---|---|
| `defect_beads`, `defect_centroid`, `defect_bead_mean` | Per-site defects on each bead, on Cartesian-centroid coordinates, and their bead average |
| `q_beads`, `q_centroid`, `q_bead_mean`, `q_bead_variance` | Q=sum of squared site defects; population bead moments |
| `v_occ`, `mean_defect_square` | Sum of site bead variances; sum of squared mean site defects |
| `variance_identity_residual` | Q_bead_mean - mean_defect_square - v_occ |
| `distance_beads`, `distance_centroid`, `distance_bead_mean`, `distance_bead_variance` | D=-sum over unique center pairs of distance times both defects, using distance_kappa |
| `oo_coordination_centroid`, `oo_coordination_centroid_mean` | Per-center and mean ordinary geometric coordination, sum_j 1/(1+(r_ij/r0)^6) |
| `centroid_positions` | PBC-consistent Cartesian-centroid positions |

The identity is `Q_bead_mean = mean_defect_square + v_occ`.
In general `mean_defect_square != q_centroid`, so `v_occ` must not be replaced
by `q_bead_mean - q_centroid`. Distance is a defect-weighted pair sum, not a
normalized geometric ion separation. It is not clipped. There is no cutoff.

The global coordination mean is only an initial ordinary-environment
descriptor. It may dilute a local reaction environment and does not exclude
unmeasured ordinary slow coordinates. Path moments do not retain bead
adjacency; their permutation invariance is not a Hamiltonian symmetry claim.

## Streaming complete physical frames

```python
from molsimflow.postprocess.pimd_path_io import iter_pimd_path_frames
from molsimflow.postprocess.quantum_path import describe_quantum_path

# Supply these from a verified immutable input manifest, not filename guessing.
for frame in iter_pimd_path_frames(
    bead_paths, bead_order=bead_order, expected_identity=atom_id_to_type,
    selected_steps=selected_steps,
):
    descriptor = describe_quantum_path(
        frame.positions, frame.atom_types,
        frame.bounds[:, 1] - frame.bounds[:, 0],
        center_type=center_type, assigned_type=assigned_type,
        kappa=kappa, distance_kappa=distance_kappa,
        reference=reference, environment_r0=environment_r0,
    )
```

`bead_paths` maps explicit bead IDs to distinct dump paths. The reader reuses
`io.lammps_dump.iter_lammps_dump_records`, sorts atom rows by validated tags,
and rejects missing/extra beads, mismatched steps or boxes, changing atom
identity, duplicates and incomplete tails. Image columns, if present, must
be a complete integer triple. Exhaust the iterator to validate the whole
input; stopping early does not inspect the unread tail. Selecting early
steps does not silently admit a malformed later tail.

Files cannot prove their own bead labels: preserve a hash-locked mapping in
the analysis contract. File integrity, numerical qualification, matching
bias/state timestamps and scientific selection remain caller responsibilities.
For later FES analysis reuse `pimd_fes` frame weights and estimators, retaining
physical-frame and block grouping. A frame contributes one weight; its beads
split that contribution by their count.

Run focused tests with `PYTHONPATH=src python -m pytest -q
tests/test_quantum_path.py tests/test_pimd_path_io.py`. Tests cover analytic
assignments, the nonlinear centroid distinction, units, periodic images,
symmetries, malformed input, missing tails and frame-weight preservation.
Independent installed-PLUMED value comparisons are a separate integration
gate. Atomic/box derivatives and online bias are not implemented by this API.

## Contract-driven command

```bash
molsimflow postprocess quantum-path \
  --contract quantum-path-contract.json \
  --output quantum-path-results
```

One invocation handles one declared run. The output directory must not exist.
The [portable example](../examples/quantum-path/README.md) generates small
synthetic dumps and fills their SHA256 identities without any research data.
Its [JSON template](../examples/quantum-path/contract.json) is not executable
until its placeholder hashes have been replaced with actual file hashes.

The schema version is `1`. Paths are resolved relative to the contract file.
Every listed bead file must be unique and match its declared SHA256 before
analysis. The list order in `beads` is the bead topology; atom tags and types
come from `atom_identity`, not from position or filename conventions.

| Contract field | Required meaning |
|---|---|
| `run` | `run_id`, `seed_id`, `initial_path_id`, nullable `parent_restart_id`, `bias_mode`, and `data_role` |
| `run.bias_mode` | `classical`, `centroid_coord`, `bead_mean`, or `bead_density_shared` |
| `run.data_role` | `engineering`, `discovery`, or `validation` |
| `atom_identity` | List of unique positive integer `{id, type}` pairs |
| `beads` | Ordered list of `{bead_id, path, sha256}` records; explicit bead IDs |
| `steps` | Integer `first`, `last`, `stride`, plus positive finite `timestep_fs` |
| `geometry` | The descriptor parameters above; the CLI requires `length_unit: angstrom` and does no implicit conversion |
| `weights` | One explicit weight provider described below |
| `analysis` | Optional fixed blocks, conditioning cells and same-bead target regions |

Physical time is `step * timestep_fs`. The required selected sequence is the
inclusive range `first..last` at `stride`; missing steps, duplicate identities,
misaligned beads and malformed unread tails are errors. Declaring a narrow
selection does not bypass validation of the remaining trajectory. Run, seed,
initial-path and restart labels preserve lineage. Different labels alone do not
prove statistical independence. This command neither combines runs nor joins
restart segments, and blocks never cross an invocation's run boundary.

The output files are `frames.csv` (scalar path moments, lineage, step/time and
raw frame log weight), `beads.csv` (reconstructed bead Q, D and log-distance),
optional `conditional.json`, and `result.json` (status and provenance). Input,
contract and implementation identities support auditing; hashes are checked
again after analysis. Failures retain a receipt and any partial outputs. A
successful engineering result proves only the checks explicitly reported.

## Frame weights and conditional summaries

The default descriptive provider must still be explicit:

```json
{"weights": {"kind": "uniform_sampler"}}
```

It summarizes the supplied sampler, including any sampling bias. It does not
produce equilibrium probabilities or a quantum free-energy surface. External
weights can instead be supplied through:

```json
{
  "weights": {
    "kind": "precomputed",
    "path": "frame-weights.csv",
    "sha256": "REPLACE_WITH_THE_ACTUAL_FILE_SHA256",
    "target_id": "declared-target",
    "admission_reference": "external-weight-validation-record"
  }
}
```

The CSV has exactly the columns `step,log_weight`, with one finite log weight
for every selected physical frame in exact step order. Steps are integers.
There are no bead-specific weights. `target_id` identifies the intended target;
`admission_reference` identifies the external validation, which this command
does not reproduce or automatically accept as a scientific qualification.
Changing a declaration cannot make invalid weights valid.

Optional `analysis` supplies `block_frames`, `conditioning_fields`, parallel
`bin_edges`, and `regions`. The available conditioning scalars are
`q_centroid`, `distance_centroid`, `logdistance_centroid`, and
`oo_coordination_centroid_mean`. Bin edges must be finite and strictly
increasing. Fix the selection, block size, edges and target regions before
examining their effects. `block_frames` must divide the selected frame count
exactly and produce at least two contiguous blocks.

A target region contains a name and bounds on one or more of `q`, `distance`
and `logdistance`. All target-region bounds are half-open `[lo, hi)`.
Conditioning bins are also half-open except that the final upper edge in each
dimension is included, following the histogram convention. Multiple dimensions
in a region are AND conditions on the same bead. Regions
may overlap, so their probabilities need not sum to one. For a physical frame,
the target observable is the fraction of its beads inside the region. That
fraction receives the single frame weight in its conditioning cell. Averaging
bead-conditioned samples separately would change the estimator and overcount
correlated observations.

Block diagnostics describe the supplied sequence. Empty or poorly supported
cells, unequal weights, shared initial paths and correlation between blocks
limit interpretation; no bead count or label repairs these limitations. The
workflow does not fit a model, calculate a committor, identify a physical rate,
or establish that a path descriptor is necessary for enhanced sampling.

## Native log-distance convention

The CLI requires `geometry.length_unit` to be `angstrom`: coordinates, box
lengths and `environment_r0` must be in angstrom, with the kappas in inverse
angstrom. It always reports the established piecewise distance transform using
D from the separately parameterized `distance_kappa`:

```text
log(D + 0.03) * step(1.0 - D) + (D - 0.9704412) * step(D - 1.0)
```

Here D is expressed numerically in angstrom. Native `step(x)` is zero for
`x < 0` and one otherwise. At exactly D = 1 angstrom, both terms contribute.
The logarithm requires D > -0.03 angstrom; values outside that domain fail
instead of being clipped. The CLI rejects other length-unit labels because
retaining the constants after a unit change would silently change this
observable. The raw `describe_quantum_path` API remains unit-generic when all
coordinates, box lengths, r0 and kappas use consistent units; it does not apply
this log-distance transform. The transform is a chosen geometric coordinate,
not a normalized ion separation or free energy, and its reuse does not make
an arbitrary system a water-ionization model.
