# Electronic path profiles

`molsimflow postprocess electronic-path-profiles` builds path-ordered Hirshfeld
charge and spin summaries from an atom-level table and a frame-level table. A
JSON configuration defines atom labels, route order, path states, and optional
expected geometry classes.

Frames remain visible in the output when their observed geometry does not match
the configured chemical state, but they are excluded from admitted path means.
This prevents directory names or historical state labels from silently becoming
chemical assignments.

Required atom-table columns are `system`, `stratum`, `replicate`,
`atom_index_one_based`, `charge`, and `spin`. The index is intentionally named
for the atom it identifies rather than for a particular electronic-structure
program. Required frame-table columns are `system`, `stratum`, `replicate`,
`geometry_state`, `organic_charge`, and `organic_spin`.

Configuration shape:

```json
{
  "systems": {
    "example": {
      "atom_labels": {
        "donor_C": 10,
        "target_O": 12
      },
      "routes": {
        "water-mediated": {
          "states": [
            {
              "label": "N",
              "stratum": "reactant_box",
              "expected_geometry_state": "reactant-like C-H"
            },
            {
              "label": "A",
              "stratum": "ion_pair_box",
              "expected_geometry_state": "anion/hydronium-like"
            }
          ]
        }
      }
    }
  }
}
```

Example:

```bash
molsimflow postprocess electronic-path-profiles \
  --atom-table atom_assignments.tsv \
  --frame-table path_frames.tsv \
  --config electronic_path_config.json \
  --output-dir electronic_path_results
```

Outputs include long-form frame data, geometry-gated state summaries, an atomic
charge plot, an organic-fragment charge/spin plot, and metadata. Use `--no-plots`
for table-only execution. These profiles are structural/electronic descriptors,
not populations, rates, or committor assignments.
