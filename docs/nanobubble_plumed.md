# N2 Nanobubble PLUMED Generation

`molsimflow.plumed.nanobubble` migrates the legacy N2 COM PLUMED scripts into a
reusable generator.  It supports two modes:

- `cluster-size`: pair N atoms into N2 dimers, define COM labels, build
  `reps_center`, and bias `sum_cn.sum`.
- `surface-distance`: additionally select the top layer of a surface element,
  build a surface COM, and bias `dz.z` between the N2 COM and surface COM.

The command can work in legacy range mode:

```bash
PYTHONPATH=/path/to/molsimflow/src python -m molsimflow.cli plumed n2-com \
  --start 9451 \
  --stop 9854 \
  --output plumed_n2_com.dat
```

It can also infer N2 dimers directly from a structure:

```bash
PYTHONPATH=/path/to/molsimflow/src python -m molsimflow.cli plumed n2-com \
  --structure model.xyz \
  --format extxyz \
  --output plumed_n2_com.dat
```

For a SiO2 interface, use silicon as the top-layer surface element.  The top
layer is all Si atoms within `--surface-z-tolerance-A` of the maximum Si
coordinate, then downsampled by `--surface-stride` after sorting by atom id:

```bash
PYTHONPATH=/path/to/molsimflow/src python -m molsimflow.cli plumed n2-com \
  --structure model.xyz \
  --format extxyz \
  --with-surface \
  --surface-element Si \
  --surface-z-tolerance-A 0.5 \
  --surface-stride 10 \
  --output plumed_n2_com_sio2_surface.dat
```

LAMMPS `atom_style atomic` data files are supported when the `Masses` section has
element comments.  If comments are missing or need overriding, pass a type map:

```bash
PYTHONPATH=/path/to/molsimflow/src python -m molsimflow.cli plumed n2-com \
  --structure model.atomic.data \
  --format lammps-data \
  --type-map 2=O 3=N 8=Si \
  --with-surface \
  --surface-element Si \
  --surface-stride 10 \
  --output plumed_n2_com_sio2_surface.dat
```

Generated PLUMED files, scheduler scripts, and run logs should stay in case
directories.  The repository should only track the reusable Python code, tests,
templates, and documentation.
