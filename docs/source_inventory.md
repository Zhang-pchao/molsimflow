# Source Inventory

Inventory date: 2026-05-25.

Host: `ssh 11`.

## Main Post-Processing Project

Source:

- private legacy MD post-processing project path, omitted from public docs;

Observed contents:

- about 18 MB;
- 982 files total;
- existing package scaffold with `pyproject.toml`, `README.md`, `pytest.ini`,
  `src/md_postprocess`, `scripts`, `docs`, `slurm_templates`, and `smoke_tests`;
- about 95,740 Python source lines under `src/md_postprocess`;
- largest category is `src/md_postprocess/analysis` with more than 100 modules.

Publicization risks:

- many scripts and docs still contain personal absolute paths;
- several scripts encode case-specific output directories;
- scheduler templates include fixed conda/module choices;
- backup and smoke-test output material should not be published as package code.

## PLUMED Generators

Paths:

- `gen_large_v3_plumed_dif_size.py`
- `gen_large_v3_plumed_same_size_16.87.py`

Observed contents:

- both files are 846 lines;
- they are functionally identical except for hardcoded default paths and output names;
- reusable logic includes PACKMOL parsing, LAMMPS atom-count validation, atom-range
  inference, bubble geometry inference, and PLUMED text generation.

Migration action:

- consolidate into `molsimflow.plumed.double_bubble`;
- remove private defaults and make data, PACKMOL, build script, and output paths explicit.

## Structure Preparation Scripts

Paths:

- `generated_slab_v3_dif_size_naoh/org`
- `generated_slab_v4_same_size_16.87_naoh/org`

Observed contents in each directory:

- `ase_tio2_cif2geo_slab2_2bubble_2d3d_ph.py`
- `add_exyz_pbc/add_exyz_pbc.py`
- `convert_exyz_to_lmp_data/convert_exyz_to_lmp_data.py`
- `run_simulation.sh`

Publicization risks:

- CIF and PACKMOL molecule paths are hardcoded;
- run scripts hardcode a conda environment and Packmol installation path;
- comments in some helper scripts are not consistently English;
- two versions differ mainly in bubble radius settings and equal-volume-control logic.

Migration action:

- keep raw scripts only in `legacy_sources`;
- migrate reusable file conversion first;
- migrate TiO2 slab and PACKMOL builder after defining an explicit configuration model.
