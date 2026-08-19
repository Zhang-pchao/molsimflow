# Model Validation, Trajectory, and Media Utilities

This utility set extracts reusable operations from case-specific analysis and
rendering scripts. It contains no fixed molecule labels, atom types, frame
counts, box lengths, filesystem paths, or publication conclusions.

## CP2K Energy and Force Parsing

The parser reads the final complete `ENERGY_FORCE` result and converts atomic
forces from Hartree/Bohr to eV/Angstrom:

```python
from pathlib import Path

from molsimflow.io import parse_cp2k_energy_forces

result = parse_cp2k_energy_forces(Path("cp2k.out"), atom_count=128)
print(result.energy_hartree)
print(result.forces_eV_A.shape)
```

It parses existing output only; it does not run CP2K or infer a computational
method from a directory name.

## Model-Validation Helpers

`molsimflow.postprocess.model_validation` provides:

- atom-wise vector force-error statistics for all atoms or explicit zero-based
  atom indices;
- relative-energy errors after independent minimum shifts within user-supplied
  groups;
- chunked nearest-coordinate RMSD with optional orthorhombic periodic boxes;
- explicit exact, near-neighbor, and non-exact coordinate labels.

Nearest-coordinate RMSD compares matching atom indices directly. It does not
rotate, translate, or permute structures. An exact coordinate row in arrays
demonstrates a coordinate match in those arrays only; a near neighbor does not
establish training membership, model lineage, or force accuracy.

## Reaction Kinetics

Create a CSV or TSV file:

```text
label	barrier_kj_mol
pathway A	72.5
pathway B	81.0
```

Then run:

```bash
molsimflow postprocess reaction-kinetics \
  --input barriers.tsv \
  --output-dir kinetics \
  --temperature-K 300 \
  --barrier-shifts-kj-mol -10 -5 0 5 10 \
  --competitor-rates-s-inv 0.1 1 10
```

The command writes `barrier_sensitivity.tsv`, optional
`two_channel_competition.tsv`, and compact metadata. Eyring rates use the
requested temperature and transmission coefficient. Competition fractions are
conditional two-channel results based on the supplied effective first-order
rates; they are not measured branching ratios.

## Reactive-Path Descriptors

Proton-defect and hydrogen-bonded water-wire analysis is already available
through the generic manifest-driven command:

```bash
molsimflow postprocess reactive-path-frames \
  --manifest frames.tsv \
  --site-config reactive_sites.json \
  --output-dir reactive_path_results
```

Atom indices and geometric thresholds are explicit in the site configuration.
This avoids a second trajectory analyzer with material-specific PLUMED group
names or atom-order assumptions.

## LAMMPS Trajectory Preparation

Concatenate and select frames, with optional periodic Z unwrapping and a global
minimum-Z shift:

```bash
molsimflow postprocess prepare-trajectory \
  --input segment_1.lammpstrj \
  --input segment_2.lammpstrj \
  --output prepared.lammpstrj \
  --frame-start 0 \
  --stride 5 \
  --unwrap-z \
  --shift-min-z-A 2.0
```

Defaults keep every frame, do not drop segment boundaries, do not unwrap, and
do not shift. All custom atom columns are preserved. The parser currently
supports orthorhombic boxes and `z` or `zu` coordinates for transformations.

Align a first-frame reference layer by explicit atom IDs or a configurable atom
type:

```bash
molsimflow postprocess align-reference-layer \
  --input prepared.lammpstrj \
  --output aligned.lammpstrj \
  --reference-atom-type 7 \
  --layer-edge lowest \
  --layer-tolerance-A 0.5
```

No element, atom type, layer size, or target height is assumed. Use
`--expected-atom-count` as an optional fail-closed check. Unwrap Z first when
the tracked layer crosses a periodic boundary.

## Image Sequences and Video

Install the optional media dependencies and render any naturally numbered
sequence:

```bash
python -m pip install -e ".[media]"
molsimflow media image-sequence-video \
  --image-dir frames \
  --pattern "render_*.png" \
  --output trajectory.mp4 \
  --crop-white-border \
  --fps 10 \
  --time-step 0.5 \
  --time-unit ps
```

The first selected image is the crop reference unless `--reference-image` is
provided. Time annotation is omitted unless `--time-step` or `--time-end` is
given. Fonts use an explicit `--font` path or a portable fallback; no
user-specific font location is embedded.
