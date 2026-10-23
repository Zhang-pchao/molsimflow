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
