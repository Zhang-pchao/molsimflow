# Workflow Configuration

## Purpose

Reusable package code should not contain private data paths, module loads,
conda activation commands, queue names, or account names.  Keep those settings
in external config files or private setup scripts, then pass them into
`molsimflow` commands or scheduler templates.

The repository includes a public example:

```text
templates/config/workflow.example.ini
```

It is safe to copy for a private run and replace placeholder paths.

## Config Helpers

Inspect sections:

```bash
molsimflow config summary --config workflow.ini
```

Export selected sections as shell variables:

```bash
molsimflow config env \
  --config workflow.ini \
  --section scheduler \
  --section structure
```

Use the output with a scheduler template:

```bash
eval "$(molsimflow config env --config workflow.ini --section scheduler --section structure)"
sbatch templates/slurm/double_bubble_preprocess.slurm
```

Resolve paths relative to the config file:

```bash
molsimflow config resolve-path \
  --config workflow.ini \
  --section structure \
  --key OUTPUT_DIR
```

## Sections

Suggested public sections:

- `scheduler`: `ENV_SETUP_SCRIPT`, `MOLSIMFLOW_SOURCE_DIR`, `PYTHON_BIN`,
  `PACKMOL_COMMAND`.
- `structure`: `STRUCTURE_MODE`, `INTERFACE_STRUCTURE`, `MOLECULE_DIR`,
  `OUTPUT_DIR`, bubble geometry, and run toggles.
- `conversion`: `POSCAR_PATH`, `EXTXYZ_OUTPUT`, `LAMMPS_DATA_OUTPUT`.
- `postprocess`: trajectory, topology, COLVAR, and output-root settings.

The config helper only prints valid shell variable names.  Descriptive notes or
non-exportable keys can stay in the file without becoming environment
variables.

## Private Setup Scripts

Cluster-specific setup belongs in a private script referenced by
`ENV_SETUP_SCRIPT`.  That script can load modules, activate environments, or
set scheduler-local paths.  Do not commit it unless it contains no private
cluster details.
