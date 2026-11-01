# Molecular placement under explicit geometric conventions

`molsimflow.postprocess.molecular_placement` screens fixed molecular shapes in
one frozen union of Cartesian cells. It uses complete source-frame arrays and
returns continuous single- and two-pose distance scores with inspectable limiting
atom pairs. It does not read trajectories, select chemical backgrounds, infer
species, set physical radii, define site observers, or run molecular dynamics.

## Inputs and completeness

Construct inputs with `complete_frame`, `molecular_shape`, `pair_convention`,
and `cell_domain`. The frozen data classes returned by these factories contain
immutable tuples; direct data-class construction is not a validated input path.

- `complete_frame` requires all source atom IDs, species, Cartesian positions,
  an independently established expected full atom-ID roster, the explicitly
  selected obstacle IDs, and orthorhombic lower/upper bounds. Actual and expected
  full rosters must match exactly. Missing atoms, unknown selected IDs, duplicate
  IDs, invalid coordinates, or missing pair conventions fail closed.
- The caller must establish the expected roster from source identity and atom
  inventory. Supplying the same cropped array as both actual and expected IDs
  is not an independent completeness proof. The library verifies the declared
  contract; it cannot authenticate provenance or chemical selection.
- Explicit obstacle subsets are analytical backgrounds. Every omitted species
  or molecule class must be recorded by the caller. No spatial halo crop is
  supported. All selected obstacle atoms from the full source frame are indexed.
- Geometry is Cartesian, orthorhombic, periodic in X and Y, and nonperiodic in Z.
  The caller must verify these boundary conditions against source metadata.
  Triclinic and other boundary-condition models require a different implementation.
- `molecular_shape` requires species, positive masses, and fixed atom offsets
  centered on their mass-weighted center of mass. No default bond length or shape
  is provided. Proper 3-by-3 rotations and explicit COM samples are supplied by
  the caller; the module does not certify angular or translational resolution.
- `cell_domain` defines a frozen union of half-open boxes. XY boxes must lie in
  the primary periodic box; split cells at a periodic seam. Coordinates are
  wrapped only in XY for domain membership.

Unknown completeness is a rejected input, not a blocked site. A full source
roster does not by itself prove a scientifically appropriate obstacle selection.

## Two separate score conventions

For a pose with atom positions (x_i), obstacle positions (y_j), and an explicit
species-pair table (r_{ab}), the single score is

```text
s(p) = min over all probe-obstacle pairs [MIC_XY_distance(x_i, y_j) - r_ab].
```

`mode="center_distance"` requires every supplied table value to be zero. It
measures nuclear-center distance capacity. It never reports a binary exclusion
witness or physical nonoverlap.

`mode="exclusion_clearance"` requires positive pair values. These can encode a
separately frozen sum of radii and a signed gap, but the caller must supply and
justify the final positive pair distances. The table is symmetric; missing
entries and conflicting reversed entries are rejected. A nonnegative score is
only a witness under this explicitly chosen exclusion convention.

No score establishes a potential of mean force, insertion probability, exchange
rate, persistence, connected transport path, or physical accessibility. A
negative score at an observed configuration does not invalidate that trajectory.
Do not tune pair distances after observing a control merely to make it pass.

For two distinct sampled poses in the same frozen cell union,

```text
s2(p, q) = min(s(p), s(q), every intermolecular atom-pair clearance).
```

The two COMs may be in different cells. There is no requirement to fit two
molecules in each cell. Every intermolecular atom pair is included; two poses
that separately clear the background can still fail the two-pose convention.

`PoseResult.all_probe_nuclei_in_domain` tests only atom-center membership in the
union. It does not test bond segments, finite atomic envelopes, or containment
of the whole molecule. A COM inside the domain does not require every atom to
be inside it. No continuous translation or rotation path is inferred between
sampled poses.

## Exact nearest neighbors and bounded pair search

The optional `analysis` dependency SciPy supplies `cKDTree`. Selected obstacles
are grouped by species and normalized in XY. Each group is indexed with nine XY
images (shifts -L, 0, +L in each periodic direction), with no Z images. Exact
`eps=0` nearest queries are equivalent to full-array XY minimum-image distances
because both query and base coordinates are in the primary XY box. A constant
species-pair subtraction preserves the nearest member of that species.

Candidate poses and scores use compact NumPy arrays. Species queries and
intermolecular pair distances use bounded internal chunks. No pose-by-all-atoms
or all-pairs pose matrix is retained. Memory scales linearly with obstacle and
evaluated-pose counts; the nine-image index increases the obstacle constant.

`max_pose_evaluations` and `max_pair_evaluations` are required positive budgets.
Optional `max_obstacle_queries` bounds individual probe-atom/species nearest
queries. If omitted, its bound is derived explicitly as the pose budget times
probe-atom count times obstacle-species count. A budget smaller than one complete
pose yields no pose and an incomplete result.

Pair candidates are sorted by decreasing single score; ties retain the original
COM/rotation order. Since `s2(p,q) <= min(s(p),s(q))`, remaining sorted pairs can
be pruned only when their upper bound cannot improve the best evaluated pair.
The implementation reports:

- `poses_evaluated`, `poses_planned`, `obstacle_queries`, and query budget;
- evaluated, planned, and upper-bound-pruned pair counts;
- `best_single` and `best_two`, including limiting pairs and component;
- `best_two_upper_bound_A` and `best_two_gap_A`;
- `finite_grid_optimum`, which applies only to the supplied finite pose grid.

`EXHAUSTIVE_DISCRETE_SEARCH` means all required discrete pairs were evaluated.
`CERTIFIED_DISCRETE_OPTIMUM` means a remaining-pair upper bound proves the same
finite-grid optimum. `BUDGET_EXHAUSTED` means only a best-tested lower bound is
available. If any planned pose was not evaluated, the unsearched full-grid upper
bound is infinity, not an extrapolated certificate.

In exclusion mode, `two_pose_threshold_witness` is true whenever a tested
nonnegative joint score is available, even in an incomplete search. It is false
only after a complete finite-grid search has no such witness, and otherwise
null. In center-distance mode it is always null. A complete finite negative is
still not proof that continuous geometry has no suitable pose.

## Rescoring frozen witness coordinates

`evaluate_pose_positions(frame, species, positions_A, rule)` evaluates one
supplied pose. `evaluate_pose_pair_positions(frame, first_species,
first_positions_A, second_species, second_positions_A, rule)` evaluates exactly
two supplied poses and all their intermolecular pairs. Neither function performs
a pose search, changes coordinates, or accepts arbitrary self-exclusions.
They support applying a predeclared secondary convention to a frozen primary
witness without allowing the secondary convention to select another pose.

The caller remains responsible for coordinate provenance and molecular
membership. Per-element obstacle minima can be computed with explicitly recorded
species-specific background selections and the same evaluator; no new geometry
convention is introduced by such a diagnostic partition.

## Observed-pose control

`occupied_pose_control(frame, occupied_atom_ids, rule)` uses only the actual
coordinates of the named source atoms. The caller must obtain complete molecule
membership from an admitted molecular roster. Atom existence does not prove
bonded membership. No nominal shape, fitted bond length, or arbitrary replacement
coordinates are accepted.

Only these exact atom IDs are removed from the selected obstacle set; returned
`excluded_self_ids` makes that removal inspectable. Ordinary placement searches
have no self-exclusion argument. Analytical omission of an entire molecule class
must instead be declared in the background selection and applies consistently
to all candidate poses.

## Minimal example

```python
import numpy as np
from molsimflow.postprocess.molecular_placement import (
    cell_domain, complete_frame, molecular_shape, pair_convention,
    search_placements,
)

frame = complete_frame(
    [10], ["O"], [[8., 8., 8.]], expected_atom_ids=[10],
    obstacle_atom_ids=[10], bounds_A=[[0., 10.], [0., 10.], [-5., 20.]])
probe = molecular_shape(["X", "X"], [1., 1.], [[-.5, 0., 0.], [.5, 0., 0.]])
rule = pair_convention({("X", "O"): 0., ("X", "X"): 0.}, mode="center_distance")
cells = cell_domain([1, 2], [[0., 0., 0.], [5., 0., 0.]],
                    [[5., 10., 5.], [10., 10., 5.]])
result = search_placements(
    probe, frame, rule, cells, [[2., 2., 2.], [7., 2., 2.]], [np.eye(3)],
    max_pose_evaluations=2, max_pair_evaluations=1, max_obstacle_queries=4)
assert result.finite_grid_optimum
assert result.two_pose_threshold_witness is None
```

The example dimensions are synthetic and carry no species-specific physical
calibration. Synthetic tests cover exact full-array parity, shape orientation,
species tables, periodic seams, nonperiodic Z, completeness, independent
single/two-pose limits, upper-bound certificates, budget truncation, and control
self-exclusion. They establish implementation behavior, not scientific acceptance
of any molecular system or parameter convention.
