# Portable synthetic quantum-path example

This example exercises the contract, streaming reader, descriptors and optional
conditional summaries using two beads, four atoms and eight physical frames.
It is constructed geometry, not a water simulation or a model of dynamics.
Two centers and two assigned atoms use reference occupancy 1. No PES, GPU,
scheduler, trained model or scientific trajectory is required.

`contract.json` is a template. Its zero SHA256 placeholders are deliberately
invalid input identities. Generate the dumps and fill their real hashes before
running the analysis. Do not relax hash checks to run the template directly.

From the repository root, use a Python environment with NumPy installed:

```bash
quantum_path_demo=$(mktemp -d "${TMPDIR:-/tmp}/molsimflow-quantum-path.XXXXXX")
python - "$quantum_path_demo" <<'PY_FIXTURE'
import hashlib
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
contract = json.loads(Path("examples/quantum-path/contract.json").read_text())
for bead in contract["beads"]:
    bead_id = bead["bead_id"]
    rows = []
    for frame in range(8):
        shift = 0.25 * math.sin(0.7 * frame) + (2 * bead_id - 1) * 0.12
        atoms = [(-1.2, 0.0, 0.0), (shift, 0.2, 0.0),
                 (1.2, 0.0, 0.0), (-0.65, -0.25, 0.0)]
        rows += ["ITEM: TIMESTEP", str(frame * 10), "ITEM: NUMBER OF ATOMS", "4",
                 "ITEM: BOX BOUNDS pp pp pp", "-10 10", "-10 10", "-10 10",
                 "ITEM: ATOMS id type x y z"]
        for identity, xyz in zip(contract["atom_identity"], atoms):
            rows.append(f"{identity['id']} {identity['type']} "
                        + " ".join(f"{value:.16g}" for value in xyz))
    path = root / bead["path"]
    with path.open("x") as handle:
        handle.write("\n".join(rows) + "\n")
    bead["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
with (root / "contract.json").open("x") as handle:
    json.dump(contract, handle, indent=2)
    handle.write("\n")
print(root / "contract.json")
PY_FIXTURE
PYTHONPATH=src python -m molsimflow.cli postprocess quantum-path \
  --contract "$quantum_path_demo/contract.json" \
  --output "$quantum_path_demo/results"
```

The installed entry point is equivalent:

```bash
molsimflow postprocess quantum-path \
  --contract "$quantum_path_demo/contract.json" \
  --output "$quantum_path_demo/another-results"
```

Each output directory must be new. Expect eight rows in `frames.csv`, sixteen
rows in `beads.csv`, `conditional.json` for the declared analysis, and provenance
and status in `result.json`. Counts exclude the header row. Existing outputs
are preserved. A failed check leaves a failure receipt and any partial outputs;
those outputs are not admitted results.

The example uses one conditioning cell and two contiguous blocks of four
physical frames. Its region requires both Q in `[0.4, 2.0)` and D in `[0.0, 5.0)`
on the same bead. These arbitrary, fixed bounds illustrate input semantics.
They are not chemical-state definitions or a test of a hidden slow coordinate.

`uniform_sampler` describes the supplied sampler. Beads are averaged within a
physical frame before applying the single frame weight; they are never
independent replicas. The two blocks and run/seed labels do not establish
independent sampling, equilibrium or an uncertainty estimate for real science.
Consult [the workflow reference](../../docs/quantum_path.md) before using real data.
