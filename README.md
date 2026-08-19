# molsimflow

`molsimflow` is a Python package and command-line interface for reusable
molecular-simulation workflows. It turns one-off structure preparation,
PLUMED setup, trajectory analysis, and plotting scripts into explicit,
testable commands.

The project follows three rules:

- inputs and outputs are supplied by the user, never embedded as private paths;
- command and module names describe the operation they perform;
- reusable code, small synthetic tests, and essential documentation are kept in
  Git, while trajectories and generated results stay outside the repository.

## Requirements

- Python 3.9 or newer;
- NumPy for the core package;
- optional dependencies only for workflows that need plotting, structure
  handling, DeepMD data, or video output.

## Installation

Install the core package from a checkout:

```bash
python -m pip install -e .
```

Install only the optional groups you need:

```bash
python -m pip install -e ".[analysis]"
python -m pip install -e ".[structure]"
python -m pip install -e ".[deepmd-sketch]"
python -m pip install -e ".[media]"
python -m pip install -e ".[dev]"
```

## Discovering commands

The CLI is organized by task. Start from the built-in help instead of copying a
project-specific script:

```bash
molsimflow --help
molsimflow structure --help
molsimflow plumed --help
molsimflow postprocess --help
molsimflow plot --help
molsimflow media --help
```

For direct execution from an uninstalled source checkout:

```bash
PYTHONPATH=src python -m molsimflow.cli --help
```

## Quick examples

Convert an extended XYZ structure to a LAMMPS data file:

```bash
molsimflow structure extxyz-to-lammps-data \
  --xyz model.xyz \
  --output model_atomic.data
```

Run generic PLUMED diagnostics for selected collective variables:

```bash
molsimflow postprocess plumed-cv-diagnostics \
  --run-dir run \
  --output-dir diagnostics \
  --cv-kind generic \
  --target-cv coordination \
  --plot-column coordination \
  --plot-column distance
```

Describe configured reactive-path frames without embedding case paths in code:

```bash
molsimflow postprocess reactive-path-frames \
  --manifest frames.tsv \
  --site-config reactive_sites.json \
  --output-dir reactive_path_results
```

Evaluate transition-state rate sensitivity from a table whose required columns
are `label` and `barrier_kj_mol`:

```bash
molsimflow postprocess reaction-kinetics \
  --input pathway_barriers.tsv \
  --output-dir kinetics_results \
  --temperature-K 300 \
  --competitor-rates-s-inv 0.1 1 10
```

Select and unwrap a LAMMPS dump without material-specific defaults:

```bash
molsimflow postprocess prepare-trajectory \
  --input segment_1.lammpstrj \
  --input segment_2.lammpstrj \
  --output prepared.lammpstrj \
  --stride 5 \
  --unwrap-z
```

Every command accepts `--help` and writes only to paths supplied through its
arguments or configuration.

## Capabilities

| Area | Examples | Documentation |
| --- | --- | --- |
| Structure and I/O | extended XYZ, LAMMPS data, double-bubble slabs | [Configuration](docs/configuration.md) |
| PLUMED generation | double-bubble and nanobubble inputs | [Nanobubble PLUMED](docs/nanobubble_plumed.md) |
| Trajectory analysis | interfaces, hydrogen bonds, ion species, transition events | [Post-processing](docs/postprocess_migration.md) |
| Reactive paths | geometry, water-wire, charge, and spin profiles | [Frame descriptors](docs/reactive_path_frames.md), [electronic profiles](docs/electronic_path_profiles.md) |
| Model validation and kinetics | CP2K parsing, force errors, coordinate-neighbor checks, Eyring sensitivity | [Validation and media utilities](docs/model_validation_trajectory_media.md) |
| Trajectory and media preparation | dump selection, Z unwrapping, reference-layer alignment, image-sequence video | [Validation and media utilities](docs/model_validation_trajectory_media.md) |
| Plotting | line, scatter, and heatmap figures from tabular data | `molsimflow plot --help` |

The package is still migrating from research-specific scripts. The public API
contains only workflows that have explicit inputs, reusable names, and focused
tests. Local migration snapshots belong in the ignored `legacy_sources/`
directory and must never be imported by package code.

## Repository layout

```text
src/molsimflow/
  config/        External workflow configuration.
  io/            Simulation file readers and writers.
  media/         Optional image-sequence and video tools.
  plotting/      Table-driven plotting helpers.
  plumed/        PLUMED input generators.
  postprocess/   Reusable analysis workflows.
  structure/     Geometry and structure preparation.
  workflows/     Composed multi-step workflows.
docs/             Workflow and migration documentation.
templates/        Generic configuration and scheduler templates.
tests/            Small synthetic tests.
legacy_sources/   Ignored local migration references.
```

## Development and public-safety checks

Before committing a migration batch:

```bash
python scripts/audit_public_repo.py
python -m compileall -q src scripts tests
PYTHONPATH=src python -m pytest -q
```

The audit checks tracked files and unignored new files for private paths,
credentials, host aliases, addresses, generated results, logs, and backups.
Keep scheduler settings, environments, real trajectories, and case outputs in
external configuration or private working directories.

Contributions should keep source code, comments, docstrings, CLI help, tests,
and commit messages in English. Add a focused synthetic test for non-trivial
logic and avoid introducing a dependency when the standard library is enough.
Development is integrated on `devel`; validated changes are then merged into
`main`. See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE).
