# Plotting Migration

## Scope

The migrated plotting layer provides generic CSV-driven figure helpers:

- line plots for processed FES curves and time/gap series;
- scatter plots for descriptor-barrier relationships;
- heatmaps for descriptor-delta or correlation matrices.
- energy/force parity panels with density rendering and adaptive absolute-error histograms.

The legacy scripts contained many publication-specific panel layouts, color
palettes, annotations, and case-root discovery rules.  Those are not migrated
as defaults.  They can be rebuilt later as project examples that consume the
generic CSV outputs.

## Commands

```bash
molsimflow plot line \
  --input fes_processed_curves.csv \
  --x-column cv \
  --y-column free_energy_smooth_zeroed_kj_mol \
  --group-column label \
  --output fes_curves.png
```

```bash
molsimflow plot scatter \
  --input case_scorecard.csv \
  --x-column bridge__bridge_waters \
  --y-column barrier__barrier_kjmol \
  --label-column case_label \
  --fit-line \
  --output descriptor_vs_barrier.png
```

```bash
molsimflow plot heatmap \
  --input case_descriptor_delta.csv \
  --row-column descriptor \
  --column-column case_pair_label \
  --value-column delta_target_minus_reference \
  --output descriptor_delta_heatmap.png
```

Use repeated `--format` values when a figure stem should be saved in several
formats:

```bash
molsimflow plot scatter \
  --input case_scorecard.csv \
  --x-column bridge__bridge_waters \
  --y-column barrier__barrier_kjmol \
  --output descriptor_vs_barrier \
  --format png \
  --format pdf \
  --format svg
```

## API

The non-plotting helpers can be imported without Matplotlib:

```python
from molsimflow.plotting.table_plots import build_heatmap_grid, output_paths

grid = build_heatmap_grid(rows, row_column="descriptor", column_column="case", value_column="delta")
paths = output_paths("figure", ["png", "pdf"])
```

Matplotlib is imported lazily only when `plot_line_table`,
`plot_scatter_table`, or `plot_heatmap_table` is called.

Model-validation figures can reuse the parity-panel API:

```python
from molsimflow.plotting.parity import plot_parity_panel

plot_parity_panel(
    ax,
    reference,
    model,
    metrics,
    kind="force",
    title="Compressed DPA4C vs SCAN DFT",
    reference_symbol=r"\mathrm{SCAN\ DFT}",
    model_symbol=r"\mathrm{DPA4C}",
    dense=True,
)
```

The inset histogram displays the central 99.5% error range. Its bin count
scales with the square root of the displayed sample count and is bounded to
remain legible in a compact panel.

## Migration Notes

This layer is a foundation for future figure assembly.  It should stay free of
private case names, hardcoded paths, scheduler assumptions, and manuscript text.
Project-specific style presets or multi-panel figures should be added as
optional examples after the underlying analysis outputs are stable.
