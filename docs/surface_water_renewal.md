# Contact-water renewal helpers

The surface_water_renewal module supplies small array/graph helpers for
postprocessing existing trajectories. It assumes that the caller has already
validated atom identity, source coverage, coordinate registration and chemistry.

- cohort_position_states uses periodic XY distance to a fixed patch and a
  caller-supplied height relative to the surface at the molecule's current XY.
  Lateral departure within the layer is distinct from vertical layer exit.
- observed_cohort_history separates sampled endpoint retention from observed
  departure/return history. Missing time or a source change censors the trace.
- site_restoration compares water-oxygen-to-substrate-site edges. A different
  water can restore a site connection even when the original identity edge is
  lost. Channels with different physical meanings should be passed separately.
- periodic_grid_components labels an XY-periodic grid mask. Connectivity is
  geometric; a connected component does not itself prove cooperative motion or
  reveal whether the object winds around the periodic box.
- sampled_support_runs records consecutive sampled support, with explicit
  boundaries and gaps. Its duration is last observed time minus first observed
  time; it is not a continuous-time lifetime.

All lengths and times must use a consistent caller-defined unit convention.
No species, hydrogen-bond, adsorption-shell or material-specific threshold is
chosen by these helpers. A material-dependent local height should be supplied
explicitly rather than reusing an old patch height after lateral motion.

Oxygen IDs are tracking instruments, not a physical force or memory variable.
Use structure/connection restoration alongside identity overlap. Do not
interpret snapshot connection persistence as an uninterrupted bond lifetime.

Focused validation:
python -m pytest -q tests/test_surface_water_renewal.py
