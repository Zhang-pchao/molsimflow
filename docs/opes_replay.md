# Recorded OPES bias replay

The postprocess.opes_replay module replays nonperiodic OPES_METAD kernels in
deposition order. NonperiodicOPESReplay accepts centers, widths and heights,
the sum of deposition weights, bias factor, epsilon, Gaussian cutoff,
compression threshold and kBT in the original energy units.

Initialize from an earlier saved state, or from an empty kernel matrix and the
original initial weight. Add each KERNELS record with its recorded log weight.
The replay preserves recursive merging and centered second moments.
evaluate returns bias energy and its CV gradient; normalization recomputes
the full-kernel normalization at the compressed kernel centers.

Force evaluation precedes the same-step deposition. A saved post-update state
therefore includes that deposition while the corresponding bias observation
may use the preceding state. Qualify this ordering using native energy,
normalization, kernel count and saved-state comparisons before interpreting
historical forces. Coordinate gradients and engine normalization belong to
the caller; a CV gradient alone is not an atomic force.

The implementation excludes periodic CVs, OPES_METAD_EXPLORE, neighbor-list
approximations and inference of missing depositions. Recorded adaptive widths
are consumed directly. The kernel normalization is quadratic in kernel count;
use it for bounded diagnostic replay. A caller-supplied normalization must
refer to the same reconstructed state.

Tests cover the analytic force, cutoff behavior, recursive moment conservation,
initialization, disabled compression and rejection of invalid depositions.
