import csv
import json
from pathlib import Path

import numpy as np
import pytest

from molsimflow.cli import build_parser as build_cli_parser
from molsimflow.postprocess.constant_force_kinematics import (
    classify_response,
    normalized_autocorrelation,
    run_contract,
)


def _write_motion(path: Path, time_ps: np.ndarray, x_A: np.ndarray, y_A: np.ndarray) -> None:
    path.write_text(
        "# synthetic motion\n"
        "# TimeStep v_dxrel v_dyrel\n"
        + "\n".join(
            f"{int(time)} {x_value:.8f} {y_value:.8f}"
            for time, x_value, y_value in zip(time_ps, x_A, y_A)
        )
        + "\n",
        encoding="utf-8",
    )


def _piecewise_displacement(time_ps: np.ndarray, velocities_mps: list[float]) -> np.ndarray:
    result = np.zeros_like(time_ps, dtype=float)
    for index in range(1, len(time_ps)):
        midpoint = 0.5 * (time_ps[index] + time_ps[index - 1])
        block = min(int(midpoint // 1000.0), len(velocities_mps) - 1)
        result[index] = result[index - 1] + (
            velocities_mps[block] / 100.0 * (time_ps[index] - time_ps[index - 1])
        )
    return result


def _contract(tmp_path: Path) -> Path:
    time_ps = np.arange(0.0, 4000.0 + 250.0, 250.0)
    baseline_x = _piecewise_displacement(time_ps, [0.1, -0.1, 0.1, -0.1])
    baseline_y = _piecewise_displacement(time_ps, [0.05, -0.05, 0.05, -0.05])
    sustained_x = baseline_x + _piecewise_displacement(time_ps, [1.0, 1.0, 1.0, 1.0])
    reversal_y = baseline_y + _piecewise_displacement(time_ps, [1.0, 1.0, -1.0, -1.0])
    _write_motion(tmp_path / "f0.dat", time_ps, baseline_x, baseline_y)
    _write_motion(tmp_path / "fx.dat", time_ps, sustained_x, baseline_y)
    _write_motion(tmp_path / "fy.dat", time_ps, baseline_x, reversal_y)
    raw = {
        "schema_version": 1,
        "time_origin_step": 0,
        "timestep_fs": 1000.0,
        "block_ps": 1000.0,
        "velocity_sample_ps": 250.0,
        "acf_max_lag_ps": 1000.0,
        "minimum_effect_mps": 0.05,
        "classification_sigma_multiplier": 1.0,
        "write_plots": False,
        "cases": [
            {
                "case_id": "surface",
                "branch_id": "f0",
                "direction": "none",
                "motion_tables": ["f0.dat"],
            },
            {
                "case_id": "surface",
                "branch_id": "fx",
                "direction": "x",
                "motion_tables": ["fx.dat"],
            },
            {
                "case_id": "surface",
                "branch_id": "fy",
                "direction": "y",
                "motion_tables": ["fy.dat"],
            },
        ],
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return path


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_classify_response_distinguishes_sustained_reversal_and_fluctuation():
    arguments = {"minimum_effect_mps": 0.05, "sigma_multiplier": 1}
    assert classify_response([1, 1, 1, 1], 1, **arguments)[0] == "SUSTAINED"
    assert classify_response([1, 1, -1, -1], 0, **arguments)[0] == "REVERSAL"
    assert classify_response([0.01] * 4, 0.01, **arguments)[0] == "FLUCTUATION"


def test_normalized_autocorrelation_has_unit_zero_lag():
    result = normalized_autocorrelation(np.array([1.0, 2.0, 1.0, 2.0]), 3)
    assert result[0] == pytest.approx(1.0)
    assert len(result) == 4


def test_run_contract_writes_paired_blocks_and_classes(tmp_path):
    output = tmp_path / "output"
    summary = run_contract(_contract(tmp_path), output)

    assert summary["status"] == "PASS"
    assert summary["case_branches"] == 3
    rows = {row["branch_id"]: row for row in _read_tsv(output / "branch_summary.tsv")}
    assert rows["f0"]["response_class"] == "CONTROL"
    assert rows["fx"]["response_class"] == "SUSTAINED"
    assert float(rows["fx"]["excess_axis_velocity_full_mps"]) == pytest.approx(1.0)
    assert rows["fy"]["response_class"] == "REVERSAL"
    blocks = _read_tsv(output / "block_velocity.tsv")
    assert len([row for row in blocks if row["branch_id"] == "fx"]) == 4
    fx_velocities = {
        round(float(row["excess_axis_velocity_mps"]), 6)
        for row in blocks
        if row["branch_id"] == "fx"
    }
    assert fx_velocities == {1.0}
    assert (output / "velocity_acf.tsv").is_file()
    assert (output / "input_manifest.tsv").is_file()


def test_run_contract_requires_one_baseline_per_case(tmp_path):
    contract = _contract(tmp_path)
    raw = json.loads(contract.read_text(encoding="utf-8"))
    raw["cases"][0]["direction"] = "x"
    contract.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        run_contract(contract, tmp_path / "output")


def test_run_contract_refuses_existing_output(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        run_contract(_contract(tmp_path), output)


def test_main_cli_registers_constant_force_kinematics(tmp_path):
    args = build_cli_parser().parse_args(
        [
            "postprocess",
            "constant-force-kinematics",
            "--contract",
            str(tmp_path / "contract.json"),
            "--output",
            str(tmp_path / "output"),
        ]
    )
    assert args.func.__name__ == "_cmd_postprocess_constant_force_kinematics"
