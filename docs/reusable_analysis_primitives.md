# Reusable analysis primitives

These NumPy-first helpers take explicit arrays and scientific parameters. Use
project configuration to select inputs, time windows, atom identities, styles
and material conventions. The module catalog lists their public definitions.

## Time series and observed window overlap

```python
from molsimflow.postprocess.timeseries import autocorrelation_fft, statistical_inefficiency
from molsimflow.postprocess.umbrella_wham import histogram_overlap
acf = autocorrelation_fft(values)
g = statistical_inefficiency(acf, method="initial_positive_pair")
overlap = histogram_overlap(left_cv, right_cv, common_edges)
```

FFT input is a finite equally spaced sequence. Keep saved-frame gaps explicit
upstream. The positive-lag and positive-pair truncations are separate methods.
Constant-series behavior and missing-row filtering remain available under the
legacy normalized_autocorrelation/positive_acf_time_ps imports. A g estimate
and within-path overlap require separate stationarity and replica review.
The empirical histogram uses caller-supplied edges covering both samples.

## Planar mechanical observables and chunk input

```python
from molsimflow.postprocess.planar_interface import mechanical_surface_tension
from molsimflow.io.lammps_table import iter_fix_ave_chunk_blocks
gamma = mechanical_surface_tension(
    pressure_xyz, normal_length, normal_axis=2, interfaces=2,
    pressure_unit="bar", length_unit="angstrom",
)
for block in iter_fix_ave_chunk_blocks(profile_path):
    timestep = block.metadata["Timestep"]
    profile = block.data
```

Gamma is in mN/m. Each row contains diagonal total pressures in x/y/z order.
The formula uses the declared normal length and interface multiplicity. Choose
an appropriate total-pressure/kinetic/virial convention upstream. The chunk
reader accepts Number-of-rows and Number-of-chunks metadata, named columns,
and complete finite blocks. It rejects repeated headers and incomplete tails.
Density normalization, temporal blocks and plateau selection belong to callers.

## Reference frames and occupancy probes

```python
from molsimflow.postprocess.reference_registration import anchor_translation
from molsimflow.postprocess.spatial_probes import cuboid_counts
shift, residuals = anchor_translation(
    reference_ids, reference_xyz, frame_ids, frame_xyz,
    lengths=lengths, periodic=[True, True, False],
)
counts = cuboid_counts(
    selected_positions - shift, probe_centers, widths,
    lengths=lengths, periodic=[True, True, False],
)
```

Matched anchor IDs are unique integers. Translation uses componentwise medians
of per-anchor nearest-image displacements. Inspect residuals under the project's
deformation criterion. The existing Si/O scaffold observer additionally owns
anchor-relative image consensus, immutable elements and deformation limits.
Choose the observer contract before migration. All widths are full cuboid
extents in the coordinate unit. Defaults use [lower, upper) intervals on each
axis; explicit lower_closed/upper_closed flags express historical boundaries.
The supplied positions determine whether counts represent atoms or molecules.
Chemical owner assignment and accessible-volume quadrature are separate inputs.

## Resampling and multiplicity

```python
from molsimflow.postprocess.resampling import block_bootstrap_partial_spearman, bh_qvalues
result = block_bootstrap_partial_spearman(
    predictor, outcome, controls, block_ids, samples=2000, seed=42,
)
qvalues = bh_qvalues(pvalues)
```

Residuals are fitted once before resampling entire caller-defined blocks,
matching the maintained fixed-residual convention. Unequal block lengths and
repeated blocks preserve their rows. Explicit controls and finite-pair ranking
retain missing observations in their original row positions. A refitted-control
bootstrap needs a distinct statistical contract. Supply actual dependence or
replica IDs. The returned percentile interval alone cannot certify independence.
BH adjustment retains original hypothesis order, including tied p-values.

## Structure drawing

```python
from molsimflow.plotting.structures import (
    project_coordinates, find_bond_pairs, draw_atoms, draw_bonds,
)
xy, depth = project_coordinates(coordinates, view_basis)
pairs = find_bond_pairs(coordinates, elements, pair_cutoffs)
draw_bonds(ax, xy, pairs, color="gray")
draw_atoms(ax, xy, elements, radii=radii, colors=colors, depth=depth)
```

Use explicit periodic images and an equal-aspect axis. Radii share projected
coordinate units; colors, fonts and styles are supplied by the caller.
The neighbor candidate radius defaults to the largest element-pair cutoff.
An explicit candidate_radius can preserve a historical chart's smaller scope.
Distance cutoffs describe a drawing rule. They require an independent chemical
criterion before interpretation as molecular bonding. Optional SciPy and
Matplotlib dependencies load only in the operations that need them.

Project adapters can compose these helpers while preserving frozen original
scripts and output identities. Record migration parity before adopting a new
observer, resampling convention or neighbor-search scope.
