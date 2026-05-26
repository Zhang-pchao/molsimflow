# Real-Case Smoke Checks

This document records smoke checks against private legacy generated cases.  The
exact legacy filesystem paths are intentionally omitted from the public
repository; they are kept in the local migration note beside the original
analysis directory.

## PACKMOL Input Equivalence

Date: 2026-05-26.

Environment:

```bash
<activate a Python environment with ASE and molsimflow dependencies>
```

Scope:

- Regenerated PACKMOL inputs with `molsimflow structure tio2-double-bubble`.
- Compared against two private legacy generated cases.
- Compared normalized structure basenames, molecule counts, `inside box`,
  `outside sphere`, and gas-bubble `inside sphere` constraints.
- Ignored comments and absolute structure path prefixes, because the new
  workflow writes into temporary output directories.

Result: both cases matched.

### Unequal-Radius NaOH Case

Command shape:

```bash
PYTHONPATH=src python -m molsimflow.cli structure tio2-double-bubble \
  --bulk-structure <private-bulk-cif> \
  --molecule-dir <private-molecule-template-dir> \
  --output-dir <tmp-output-dir> \
  --gas-radii 19 14 \
  --bubble-spacing 50 \
  --target-ph 13.4 \
  --output-system-suffix tio2
```

Matched output:

- `output`: `17371h2o_302n2_sphere_ph13.4_tio2.xyz`
- block molecule counts: `[1, 546, 546, 16279, 78, 78, 216, 86]`
- upper water and ion exclusions:
  - `outside sphere 31.828 37.825 145.571 19.000`
  - `outside sphere 81.828 37.825 145.571 14.000`
- gas bubbles:
  - `N2.xyz`, `216`, `inside sphere 31.828 37.825 145.571 19.000`
  - `N2.xyz`, `86`, `inside sphere 81.828 37.825 145.571 14.000`

### Equal-Volume NaOH Case

Command shape:

```bash
PYTHONPATH=src python -m molsimflow.cli structure tio2-double-bubble \
  --bulk-structure <private-bulk-cif> \
  --molecule-dir <private-molecule-template-dir> \
  --output-dir <tmp-output-dir> \
  --gas-radii 16.87 16.87 \
  --bubble-spacing 50 \
  --water-height-radius 19 \
  --bubble-center-z-radius 19 \
  --target-ph 13.4 \
  --output-system-suffix tio2
```

Matched output:

- `output`: `17371h2o_302n2_sphere_ph13.4_tio2.xyz`
- block molecule counts: `[1, 546, 546, 16279, 78, 78, 151, 151]`
- upper water and ion exclusions:
  - `outside sphere 31.828 37.825 145.571 16.870`
  - `outside sphere 81.828 37.825 145.571 16.870`
- gas bubbles:
  - `N2.xyz`, `151`, `inside sphere 31.828 37.825 145.571 16.870`
  - `N2.xyz`, `151`, `inside sphere 81.828 37.825 145.571 16.870`

## Interpretation

The reusable TiO2 adapter now reproduces the legacy PACKMOL scientific content
for both unequal-radius and equal-volume NaOH double-bubble cases.  This clears
the structure-preparation path for starting post-processing migration.
