from pathlib import Path

import pytest

from molsimflow.postprocess.constant_force_energy import (
    analyze_energy_balance,
    read_lammps_thermo_block,
)


def _write_motion(path: Path, steps, work, bath):
    header = (
        "# Time-averaged data for fix MOTION\n"
        "# TimeStep v_drivework v_drivepower f_BATH v_pxx_box v_pyy_box "
        "v_pzz_box v_pxy_box v_pxz_box v_pyz_box\n"
    )
    rows = [
        f"{step} {w} 1.0 {q} 1 2 3 0 0 0"
        for step, w, q in zip(steps, work, bath)
    ]
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def _write_thermo(path: Path, steps, total, bath):
    production = [
        f"{step} 300 -10 1 {energy} {q}"
        for step, energy, q in zip(steps, total, bath)
    ]
    path.write_text(
        "noise\n"
        "Step Temp PotEng KinEng TotEng f_BATH\n"
        f"{steps[0]} 300 -10 1 {total[0]} {bath[0]}\n"
        "Loop time noise\n"
        "Step Temp PotEng KinEng TotEng f_BATH\n"
        + "\n".join(production)
        + "\nLoop time done\n",
        encoding="utf-8",
    )


def test_energy_balance_stitches_restart_and_closes(tmp_path):
    motion_1, motion_2 = tmp_path / "motion-1.dat", tmp_path / "motion-2.dat"
    thermo_1, thermo_2 = tmp_path / "lmp-1.out", tmp_path / "lmp-2.out"
    _write_motion(motion_1, [0, 1000, 2000], [0, 1, 2], [0, 0.3, 0.6])
    _write_motion(motion_2, [2000, 3000, 4000], [0, 1, 2], [0.6, 0.9, 1.2])
    _write_thermo(thermo_1, [0, 1000, 2000], [0, 0.7, 1.4], [0, 0.3, 0.6])
    _write_thermo(thermo_2, [2000, 3000, 4000], [1.4, 2.1, 2.8], [0.6, 0.9, 1.2])

    header, selected = read_lammps_thermo_block(thermo_1)
    assert header[0] == "Step"
    assert len(selected) == 3
    series, blocks, summary = analyze_energy_balance(
        [motion_1, motion_2],
        [thermo_1, thermo_2],
        timestep_fs=1.0,
        block_ns=0.002,
    )
    assert len(series) == 5
    assert len(blocks) == 2
    assert summary["drive_work_eV"] == pytest.approx(4.0)
    assert summary["thermostat_removed_eV"] == pytest.approx(1.2)
    assert summary["delta_total_energy_eV"] == pytest.approx(2.8)
    assert summary["closure_residual_eV"] == pytest.approx(0.0)
