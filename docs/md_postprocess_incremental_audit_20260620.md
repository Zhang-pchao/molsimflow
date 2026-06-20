# md_postprocess Incremental Audit: 2026-06-20

This audit covers legacy source files modified after the previous migration
cutoff on 2026-06-07.

## Legacy Files Reviewed

- `scripts/render_toc_concept_graphic.py` (modified 2026-06-19);
- `scripts/plot_figure7_titled_20260618.py` (modified 2026-06-18);
- `scripts/plot_figure6_titled_20260618.py` (modified 2026-06-18);
- `src/md_postprocess/analysis/jacs_barrier_mechanism_2x2_replot.py`
  (modified 2026-06-18).

## Migration Decision

No new reusable analysis kernel was found in these four files.

- The TOC script is a manuscript-specific SVG composition and external renderer
  wrapper.
- The Figure 6 and Figure 7 scripts use fixed source tables, fixed panel
  composition, and manuscript-specific labels.
- The JACS 2 x 2 replot script joins fixed case tables and produces a specific
  publication figure.

These files remain publication-layer scripts and are not copied into
`src/molsimflow`. Their underlying reusable inputs are already covered by the
package's FES, bridge electrostatics, coupling, case-comparison, and generic
plotting modules.

## Reusable Repository Work Completed

The existing uncommitted reusable work accumulated after the previous audit was
reviewed and completed:

- direct OPES/COLVAR 1D and 2D FES reweighting;
- PLUMED diagnostic phase-plane plots;
- DeepMD dataset descriptor sketching;
- multi-case printed-CV comparison with restart-segment time reconstruction.

These features are path-explicit and do not contain private default paths.
