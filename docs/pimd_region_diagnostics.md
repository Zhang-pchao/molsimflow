# Spatial reaction-region diagnostics

`molsimflow.postprocess.pimd.region_diagnostics` provides small NumPy helpers
for explicit spatial participation, signed squared coordination activity, fixed
bead coverage and sampled atom membership crossings. No helper changes nuclear
resolution, biases a trajectory or confines a chemical defect.

Use `sphere_weights` on coordinates already evaluated in the intended convention:
centroids, individual complete force configurations, or classical coordinates.
It returns C2 spatial weights; a PBC box must be orthorhombic and the support must
lie below half the shortest box width. Do not silently average wrapped paths.

`regional_activity` accepts complete-pool occupancies and keeps all frame/bead
axes. Positive and negative squared activity are reported separately. The local
and outside values use complementary weights with the SAME occupancies and sum
to the global result. These values are not integer ion counts or electronic
charges. Existing `observables.soft_voronoi_occupancies` supplies the full
exponential reference; it must not be relabelled as the compact PLUMED kernel.

`fixed_bead_coverage` reports low-resolution atoms/weight and whether any support
is present. Include H as well as O. Spatial coverage alone does not qualify
normalization competitors, PES receptive fields, quantum buffer adequacy or
long-time coverage. It does not accept continuous resolution amplitudes as counts.

`membership_crossings` records atom-ID entries/exits between adjacent saved
frames and skips gaps. It does not track a proton defect through host exchange,
infer exact first-passage times, or turn biased PIMD paths into physical rates.
A local/global scalar change alone cannot prove recombination rather than
transport. Preserve geometric occupancy/host information when interpreting it.

The synthetic tests cover an escaping but unrecombined pair, sign cancellation,
PBC images, center motion, permutation, frame/bead axes, an incoming classical
atom and missing saved frames.

`activity_change_terms` gives an exact symmetric finite-interval balance of
occupancy-change and spatial-weight-change contributions. A proton relay between
fixed centers belongs to the occupancy term; it is deliberately NOT called
creation or recombination. Compare its global balance and retain host information.


`adaptive_amplitude_coverage` handles continuous path amplitudes separately from
integer bead counts. Pass matching arrays with the atom axis last; explicitly
broadcast centroid amplitudes if spatial CV weights have a bead axis. It reports
support presence, reduced-amplitude atom count/weight and the weighted amplitude
deficit. The tolerance affects only classification near amplitude one. These
are geometric diagnostics, not effective bead counts or quantum-error estimates.
Evaluate oxygen and hydrogen coverage separately when the CV depends on both.


`region_weights` extends the same coordinate contract to `sphere`, `slab_x`,
`slab_y` and `slab_z`. Slabs use a fixed Cartesian normal and core/outer
half-widths. `sphere_weights` retains its previous API and numerical definition.
Centers can move with matching frame/bead axes; selecting a reference object
and reconstructing its periodic images remains explicit upstream work.

`voronoi_dependency_support` audits complete compact-kernel candidate pools.
It first finds assigned atoms touching a nonzero-weight center and then includes
all competing centers of those atoms. It preserves frame/bead axes and reports
retained assignments without an unswitched center. Feed its boolean masks into
`adaptive_amplitude_coverage` separately for centers and assigned atoms. This can
expose insufficient quantum support even when every active CV center has unit
amplitude. It is a bounded-memory offline audit, not an online acceleration or
a many-body PES neighborhood proof. The caller must include the region anchor
and independently assess buffer convergence and model receptive fields.
