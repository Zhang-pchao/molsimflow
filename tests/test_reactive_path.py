import csv
import json

import pytest

from molsimflow.postprocess.reactive_path import (
    ReactivePathConfig,
    ReactiveSiteConfig,
    _distance,
    analyze_reactive_path,
    describe_reactive_frame,
)


def _anion_hydronium_atoms():
    return [
        {"element": "H", "xyz": (1.70, 0.00, 0.00)},
        {"element": "H", "xyz": (2.70, 1.00, 0.00)},
        {"element": "H", "xyz": (2.70, 0.00, 1.00)},
        {"element": "C", "xyz": (-3.00, 0.00, 0.00)},
        {"element": "O", "xyz": (2.70, 0.00, 0.00)},
        {"element": "O", "xyz": (0.00, 0.00, 0.00)},
        {"element": "O", "xyz": (-1.40, 0.00, 0.00)},
    ]


def _write_xyz(path):
    atoms = _anion_hydronium_atoms()
    with path.open("w") as handle:
        handle.write(f"{len(atoms)}\nsynthetic reactive-path frame\n")
        for atom in atoms:
            x, y, z = atom["xyz"]
            handle.write(f"{atom['element']} {x} {y} {z}\n")


def test_describe_reactive_frame_detects_hydronium_wire():
    result = describe_reactive_frame(
        _anion_hydronium_atoms(),
        box_A=20.0,
        site_config=ReactiveSiteConfig(
            organic_oxygen_indices=(6, 7),
            donor_carbon_index=4,
            target_oxygen_index=6,
            peroxy_proximal_oxygen_index=7,
            peroxy_attachment_carbon_index=4,
        ),
        path_config=ReactivePathConfig(),
    )

    assert result["geometry_state"] == "anion/hydronium-like"
    assert result["hydronium_count"] == 1
    assert result["directed_wire_bonds"] == 1
    assert result["peroxy_O_O_A"] == 1.4


def test_distance_uses_orthorhombic_periodic_lengths():
    assert _distance((0.0, 0.0, 0.2), (0.0, 0.0, 48.1), (16.1, 16.1, 48.3)) == pytest.approx(0.4)


def test_analyze_reactive_path_writes_code_only_outputs_to_requested_dir(tmp_path):
    xyz_path = tmp_path / "frame.xyz"
    _write_xyz(xyz_path)
    manifest = tmp_path / "manifest.tsv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["system", "state", "stratum", "replicate", "xyz_path", "box_A"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "system": "demo",
                "state": "A",
                "stratum": "seed",
                "replicate": "1",
                "xyz_path": xyz_path,
                "box_A": 20.0,
            }
        )
    config = tmp_path / "site_config.json"
    config.write_text(
        json.dumps(
            {
                "systems": {
                    "demo": {
                        "organic_oxygen_indices": [6, 7],
                        "donor_carbon_index": 4,
                        "target_oxygen_index": 6,
                        "peroxy_proximal_oxygen_index": 7,
                        "peroxy_attachment_carbon_index": 4,
                    }
                }
            }
        )
    )

    outputs = analyze_reactive_path(manifest, config, tmp_path / "outputs")

    assert outputs["frames"].exists()
    assert outputs["summary"].exists()
    assert outputs["metadata"].exists()
    with outputs["frames"].open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows[0]["geometry_state"] == "anion/hydronium-like"
