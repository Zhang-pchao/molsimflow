import json

import numpy as np
import pytest

from molsimflow.cli import build_parser
from molsimflow.structure.molecular_relocation import (
    relocate_lammps_atomic_data,
    relocate_selected_oxygen_species,
)


def test_relocates_selected_oxygen_and_pbc_assigned_hydrogens_together():
    atom_ids = [1, 2, 3, 4, 5]
    atom_types = [2, 1, 1, 2, 8]
    coordinates = np.asarray(
        [
            [1.0, 1.0, 0.2],
            [1.0, 1.0, 0.8],
            [1.0, 1.0, 9.8],
            [5.0, 5.0, 4.0],
            [5.0, 5.0, 4.5],
        ]
    )
    bounds = np.asarray([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])

    result = relocate_selected_oxygen_species(
        atom_ids,
        atom_types,
        coordinates,
        bounds,
        [1],
        allowed_hydrogen_counts=(2,),
        stationary_buffer_A=1.0,
        high_boundary_buffer_A=2.0,
        image_flags=np.ones((5, 3), dtype=int),
    )

    assert result.moved_atom_ids == (1, 2, 3)
    assert result.parent_oxygen_by_atom_id == {1: 1, 2: 1, 3: 1}
    assert result.hydrogen_count_by_oxygen_id == {1: 2}
    assert np.allclose(result.coordinates[:3, 2], [5.9, 6.5, 5.5])
    assert np.isclose(result.stationary_buffer_A, 1.0)
    assert np.isclose(result.high_boundary_clearance_A, 3.5)
    assert result.maximum_oh_vector_change_A < 1.0e-12
    assert np.all(result.image_flags[:, 2] == 0)
    assert np.all(result.image_flags[:, :2] == 1)


def test_fails_closed_when_high_boundary_buffer_cannot_be_met():
    with pytest.raises(ValueError, match="high-boundary clearance"):
        relocate_selected_oxygen_species(
            [1, 2, 3],
            [2, 1, 8],
            np.asarray([[0.0, 0.0, 0.2], [0.0, 0.0, 0.8], [0.0, 0.0, 8.0]]),
            np.asarray([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]]),
            [1],
            allowed_hydrogen_counts=(1,),
            stationary_buffer_A=1.0,
            high_boundary_buffer_A=2.0,
        )


def test_lammps_atomic_data_relocation_preserves_velocities(tmp_path):
    source = tmp_path / "source.data"
    source.write_text(
        """synthetic atomic data

4 atoms
3 atom types

0 10 xlo xhi
0 10 ylo yhi
0 12 zlo zhi

Atoms # atomic

1 2 1 1 0.2 0 0 1
2 1 1 1 0.8 0 0 1
3 1 1 1 11.8 0 0 -1
4 8 5 5 4.0 0 0 0

Velocities

1 0.1 0.2 0.3
2 0.4 0.5 0.6
3 0.7 0.8 0.9
4 1.0 1.1 1.2
""",
        encoding="utf-8",
    )
    selected = tmp_path / "selected.ids"
    selected.write_text("1\n", encoding="utf-8")
    output = tmp_path / "relocated.data"
    mapping = tmp_path / "mapping.tsv"
    report = tmp_path / "report.json"

    metadata = relocate_lammps_atomic_data(
        source,
        output,
        selected,
        mapping_path=mapping,
        report_path=report,
        allowed_hydrogen_counts=(2,),
        stationary_buffer_A=1.0,
        high_boundary_buffer_A=2.0,
    )

    text = output.read_text(encoding="utf-8")
    assert "1 0.1 0.2 0.3\n2 0.4 0.5 0.6" in text
    assert metadata["status"] == "PASS"
    assert metadata["moved_hydrogen_count"] == 2
    assert metadata["report"] == str(report)
    assert metadata["maximum_oh_vector_change_A"] < 1.0e-12
    assert json.loads(report.read_text(encoding="utf-8"))["output_sha256"]
    assert mapping.read_text(encoding="utf-8").count("assigned_hydrogen") == 2
    atom_lines = [line.split() for line in text.splitlines() if len(line.split()) == 8]
    assert all(fields[7] == "0" for fields in atom_lines)


def test_cli_exposes_relocation_command():
    args = build_parser().parse_args(
        [
            "structure",
            "relocate-oxygen-species",
            "--input",
            "source.data",
            "--output",
            "relocated.data",
            "--selected-oxygen-ids",
            "selected.ids",
        ]
    )

    assert args.axis == "z"
    assert args.periodic_axes == "xyz"
    assert args.allowed_hydrogen_counts == [1, 2, 3]
