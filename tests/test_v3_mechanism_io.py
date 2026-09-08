import shutil
import subprocess

import pytest

from molsimflow.io.lammps_dump import iter_lammps_dump_records
from molsimflow.postprocess.v3_mechanism_io import validate_v3_mechanism_io


def _write_dump(path, fields, rows_by_step):
    path.write_text(
        "".join(
            f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n2\n"
            "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
            f"ITEM: ATOMS {' '.join(fields)}\n" + "\n".join(rows) + "\n"
            for step, rows in rows_by_step.items()
        ),
        encoding="utf-8",
    )


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd executable is required")
def test_validate_v3_mechanism_io_streams_zstd(tmp_path):
    coordinate_text = tmp_path / "coordinates.lammpstrj"
    velocity_text = tmp_path / "velocity.lammpstrj"
    coordinate_rows = {
        0: ("1 1 1 2 3", "2 2 4 5 6"),
        10: ("1 1 1.1 2 3", "2 2 4.1 5 6"),
    }
    velocity_rows = {
        0: ("1 1 0 0 0", "2 2 0 0 0"),
        10: ("1 1 0 0 0", "2 2 0 0 0"),
    }
    _write_dump(coordinate_text, ("id", "type", "x", "y", "z"), coordinate_rows)
    _write_dump(velocity_text, ("id", "type", "vx", "vy", "vz"), velocity_rows)
    coordinate_zst = tmp_path / "coordinates.lammpstrj.zst"
    velocity_zst = tmp_path / "velocity.lammpstrj.zst"
    for source, target in ((coordinate_text, coordinate_zst), (velocity_text, velocity_zst)):
        subprocess.run(["zstd", "-q", "-f", "-o", str(target), str(source)], check=True)
    roi_kinetic = tmp_path / "roi_kinetic.dat"
    roi_kinetic.write_text("0 2 0 0 0 0 0 0\n10 2 0 0 0 0 0 0\n", encoding="utf-8")
    global_stress = tmp_path / "global_stress.dat"
    global_stress.write_text("0 0 0 0 0 0 0\n10 0 0 0 0 0 0\n", encoding="utf-8")
    model_data = tmp_path / "model.data"
    model_data.write_text("Masses\n\n1 1.0\n2 16.0\n\nAtoms # atomic\n", encoding="utf-8")
    final_data = tmp_path / "final.data"
    final_data.write_text(
        "Atoms # atomic\n\n1 1 1.1 2 3\n2 2 4.1 5 6\n", encoding="utf-8"
    )

    result = validate_v3_mechanism_io(
        coordinates=coordinate_zst,
        velocity_roi=velocity_zst,
        roi_kinetic=roi_kinetic,
        global_stress=global_stress,
        model_data=model_data,
        final_data=final_data,
        start_step=0,
        end_step=10,
        natoms=2,
        roi_types=(1, 2),
        coordinate_every=10,
        velocity_every=10,
        thermo_every=10,
    )

    assert result["status"] == "PASS"
    assert result["coordinate_frames"] == 2
    assert result["roi_atom_count_min"] == 2
    assert next(iter_lammps_dump_records(coordinate_zst)).timestep == 0
