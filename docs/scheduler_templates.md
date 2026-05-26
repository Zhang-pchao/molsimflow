# Scheduler Templates

The repository includes generic scheduler templates for running generated
structure-preparation workflows on clusters without hardcoding any private
environment paths.

## Double-Bubble Preprocessing

Template:

```text
templates/slurm/double_bubble_preprocess.slurm
```

The template can run this chain:

1. Generate interface files and `packmol.in` with either `slab-double-bubble`
   or `tio2-double-bubble`.
2. Optionally run Packmol.
3. Add extended-XYZ PBC metadata.
4. Convert the extended XYZ file to LAMMPS `atom_style atomic` data.

Cluster-specific setup stays outside the template.  Put module loads, conda
activation, Packmol PATH changes, or project-local variables in a private setup
script and pass it with `ENV_SETUP_SCRIPT`.

The variables can be exported manually, through `sbatch --export`, or from a
private INI file:

```bash
eval "$(molsimflow config env --config workflow.ini --section scheduler --section structure)"
sbatch templates/slurm/double_bubble_preprocess.slurm
```

### Prebuilt Interface Example

```bash
sbatch \
  --export=ALL,ENV_SETUP_SCRIPT=/path/to/setup_env.sh,\
MOLSIMFLOW_SOURCE_DIR=/path/to/molsimflow,\
STRUCTURE_MODE=prebuilt,\
INTERFACE_STRUCTURE=/path/to/interface.xyz,\
MOLECULE_DIR=/path/to/molecule_templates,\
OUTPUT_DIR=/path/to/generated_case,\
GAS_RADIUS_1=19,GAS_RADIUS_2=14,BUBBLE_SPACING=50,\
RUN_PACKMOL=1,RUN_CONVERSION=1 \
  templates/slurm/double_bubble_preprocess.slurm
```

### TiO2 Builder Example

```bash
sbatch \
  --export=ALL,ENV_SETUP_SCRIPT=/path/to/setup_env.sh,\
MOLSIMFLOW_SOURCE_DIR=/path/to/molsimflow,\
STRUCTURE_MODE=tio2,\
BULK_STRUCTURE=/path/to/bulk.cif,\
MOLECULE_DIR=/path/to/molecule_templates,\
OUTPUT_DIR=/path/to/generated_case,\
GAS_RADIUS_1=16.87,GAS_RADIUS_2=16.87,BUBBLE_SPACING=50,\
WATER_HEIGHT_RADIUS=19,BUBBLE_CENTER_Z_RADIUS=19,\
RUN_PACKMOL=1,RUN_CONVERSION=1 \
  templates/slurm/double_bubble_preprocess.slurm
```

### Existing `packmol.in` Example

Use this when `packmol.in` and `POSCAR` already exist in `OUTPUT_DIR`:

```bash
sbatch \
  --export=ALL,ENV_SETUP_SCRIPT=/path/to/setup_env.sh,\
MOLSIMFLOW_SOURCE_DIR=/path/to/molsimflow,\
STRUCTURE_MODE=existing,\
OUTPUT_DIR=/path/to/generated_case,\
RUN_PACKMOL=1,RUN_CONVERSION=1 \
  templates/slurm/double_bubble_preprocess.slurm
```

## Important Variables

- `STRUCTURE_MODE`: `prebuilt`, `tio2`, or `existing`.
- `RUN_PACKMOL`: `1` to run Packmol, `0` to skip.
- `RUN_CONVERSION`: `1` to run extXYZ and LAMMPS-data conversion, `0` to skip.
- `PACKMOL_COMMAND`: Packmol executable or command.
- `MOLSIMFLOW_SOURCE_DIR`: source checkout used to set `PYTHONPATH`.
- `OUTPUT_DIR`: generated-case directory.
- `MOLECULE_DIR`: directory with `H2O.xyz`, `N2.xyz`, `Na.xyz`, and `OH-.xyz`.
- `NO_PH=1`: skip pH-control ion templates.
- `NO_INTERLAYER=1`: skip interlayer water.

Generated files such as `packmol.out`, packed XYZ, `model.xyz`, and
`model_atomic.data` are run artifacts and should not be committed.
