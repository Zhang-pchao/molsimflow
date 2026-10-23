# Optional PIMD postprocessing features

Existing contracts retain their defaults. Numerical and display extensions are
opt-in and do not select a new estimator/observable combination implicitly.

## Global weighted Scott bandwidth

Set `reweight.bandwidth_rule` to `weighted_scott` to calculate missing
`bandwidth_variants`. Explicit variants always win. The selected name is
`primary_bandwidth`, or `weighted_scott` when omitted. A private analysis output
receipt records the resolved widths and whether they were explicit or computed.

The rule uses weighted bead dispersion, complete-frame Kish ESS to the power
`-1/(dimensions + 4)`, and a floor of 1.5 median grid cells. Replicating beads
does not increase ESS. This is a global heuristic, not a locally adaptive KDE,
a correlation-adjusted rule, or evidence that a trajectory is converged.

## Unsmoothed direct-count companion

Set `reweight.histogram` to an object containing only `edges`, a mapping from
each declared CV name to explicit increasing bin edges (at least two bins).
This writes `tables/unsmoothed-path-histogram.csv` and a separate QC JSON file.
It does not replace the existing KDE outputs.

The companion reports the probability-mean bead marginal using one normalized
weight per path and a 1/P bead contribution. It applies no pseudocount, kernel,
interpolation or in-grid renormalization. Empty bins have false support and NaN
FES; out-of-grid weight mass is reported. Two-dimensional arrays use (y, x)
orientation and same-frame/same-bead pairs. Observed bead counts are distinct
from observed complete-frame counts. Explicit bin widths define the density
measure. The low-level helper accepts singleton-bead path observables too.

## Discrete 2D display style

Set `plots.fes2d_style` to `discrete` and optionally set
`plots.fes2d_energy_unit` to `kJ/mol` (default) or `kcal/mol`. This changes only
the 2D FES panels. The legacy style and all numerical table units remain
unchanged. Support masks and unsupported-panel labels are retained. The lower
level `plot_fes2d_discrete` helper accepts an automatic supported-value display
cap when `max_kcal` is None; this cap is a display choice, not an FES cutoff.

Ionization time-series plots show up to the first four beads. P1/P2 paths are
accepted without indexing nonexistent beads.
