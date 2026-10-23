# PIMD postprocessing architecture

`pimd_reweight`, `pimd_fes` and the four existing CLI commands remain compatibility
facades. Existing contracts, default artifact names, CSV columns, JSON keys,
error messages and render hooks retain their meanings.

## Scientific boundaries

- `EstimatorKind` selects probability mean or free-energy mean of bead densities.
- `ObservableKind` identifies real-bead CVs, arithmetic bead-mean CVs, contracted
  CVs or centroid CVs. An observable is not an estimator choice.
- `BiasMode` identifies the sampling Hamiltonian. Its `probability_mean` alias is
  resolved only in the bias field and does not select an estimator.
- `WeightProvider` supplies one raw log weight per complete aligned path. Fixed
  bias, explicitly declared quasi-static OPES and supplied weights remain
  distinct admissions. OPES rct is diagnostic and is not subtracted.

For a nonlinear CV, evaluate the CV on contracted Cartesian coordinates and
then average over beads. Alpha one gives the arithmetic bead-mean CV; alpha zero
is the CV of the Cartesian centroid. Do not interpolate endpoint CVs or infer a
contracted CV from real-bead CV values. External contraction metadata retains
its existing `lambda` key.

Two-dimensional observations must pair the same frame and bead. Whole frames
and independent replicas, not individual beads, are the uncertainty units.
Density support, estimator gauges and raw free-energy-mean normalization remain
explicit and separate. Passing tests does not establish equilibrium sampling.

## Ownership

| Module under `postprocess/pimd` | Responsibility |
| --- | --- |
| `types`, `contract` | Independent concepts, typed stage handoffs, legacy field adapters |
| `io` | Strict readers, consumed-input verification and input preparation |
| `alignment` | Restart ownership, complete-frame assembly and matched observations |
| `weights` | Path-weight providers and weight diagnostics |
| `kde`, `estimators`, `uncertainty` | Numerical kernels, density estimators and whole-frame errors |
| `observables` | Geometric CVs and coordinate transforms |
| `reporting`, `core_reporting`, `derived_reporting` | Schemas and core/derived output assembly |
| `water_ionization` | Optional water trajectory and ionization diagnostics |
| `plotting`, `reference` | Lazy rendering and explicitly requested legacy-driver comparison |
| `pipeline`, `cli` | Stage order and command registration |

The pipeline passes `PreparedInputs`, `AlignedPaths`, `WeightedPaths` and
`JointResults` explicitly. It does not implement a kernel or reconstruct a bias
history. Rendering callbacks preserve the existing facade's `save_figure` hook
without changing module globals or importing the facade from internal modules.

File-byte SHA256 lives in the established multi-domain `io.hashes` module.
Manifest parsing policies, strict PIMD readers and kcal/kJ CSV expansion remain
separate. Do not introduce a catch-all common module or silently substitute a
permissive reader.

## Qualification and release

Run the repository audit, compileall and the full pytest suite on Python 3.9 and
3.12. Preserve the existing PIMD tests and synthetic checks for path weights,
restart seams, 2D pairing, nonlinear contraction and legacy APIs. Compare all
output schemas and rendering hooks when moving orchestration.

Build an immutable versioned wheel from an exact clean commit outside the
worktree. Record its hash and dependency environment, test the installed wheel
from outside source directories, and distribute the same artifact. Deployment
must not copy a Git worktree or install editable source. Retain the previous
installation for rollback. Private cached-data qualification and host-specific
installation receipts belong outside this public repository.
