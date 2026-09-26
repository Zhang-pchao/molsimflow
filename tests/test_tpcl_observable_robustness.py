from pathlib import Path

import pytest

from molsimflow.postprocess.tpcl_observable_robustness import (
    aggregate_paired_sensitivity,
    extended_detector_grid,
    membership_fraction_rows,
    paired_block_slip,
)


def test_extended_detector_grid_uses_requested_small_grid(tmp_path: Path):
    kinematics = tmp_path / "02_kinematics"
    kinematics.mkdir()
    for case_id in ("ch3_only", "mixed291"):
        path = kinematics / f"{case_id}__f0_shared.tsv"
        path.write_text(
            "case_id\tstep\ttime_ps\tleading_x_A\ttrailing_x_A\n"
            + "\n".join(
                f"{case_id}\t{index}\t{index * 0.01}\t0\t0" for index in range(2001)
            )
            + "\n",
            encoding="utf-8",
        )
    trials, envelope = extended_detector_grid(
        tmp_path,
        amplitudes=(0.0, 2.0),
        durations=(0.5,),
        centers=(10.0,),
        modes=("coherent",),
        cases=("ch3_only",),
        axes=("x",),
    )
    assert len(trials) == 2
    assert len(envelope) == 2
    assert envelope[1]["expected_positive_center_advance"]


def test_membership_fraction_is_explicit():
    rows = [
        {
            "case_id": "mixed291",
            "branch_id": "f8e-5_x",
            "axis": "x",
            "edge": "leading",
            "water_hbond_formed_persistent_rate_per_ps": 3.0,
            "water_hbond_broken_persistent_rate_per_ps": 1.0,
            "water_hbond_formed_membership_rate_per_ps": 2.0,
            "water_hbond_broken_membership_rate_per_ps": 2.0,
        }
    ]
    result = membership_fraction_rows("primary", rows)
    assert result[0]["membership_fraction_of_apparent_turnover"] == pytest.approx(0.5)


def test_paired_sensitivity_reports_sign_changes():
    template = {
        "case_id": "mixed291",
        "axis": "x",
        "edge": "leading",
    }
    values = {}
    for configuration, slip in (("primary", 0.1), ("variant", -0.1)):
        row = dict(template)
        row.update(
            {
                metric: (slip if metric == "paired_delta_persistent_mean_relative_slip_rate_A_per_ps" else 1.0)
                for metric in (
                    "paired_delta_persistent_mean_relative_slip_rate_A_per_ps",
                    "paired_delta_mean_water_hbond_count",
                    "paired_delta_mean_surface_hbond_count",
                    "paired_delta_mean_sioh_surface_hbond_count",
                    "paired_delta_water_hbond_formed_persistent_rate_per_ps",
                    "paired_delta_water_hbond_broken_persistent_rate_per_ps",
                    "paired_delta_water_hbond_formed_membership_rate_per_ps",
                    "paired_delta_water_hbond_broken_membership_rate_per_ps",
                    "paired_delta_surface_hbond_formed_persistent_rate_per_ps",
                    "paired_delta_surface_hbond_broken_persistent_rate_per_ps",
                )
            }
        )
        values[configuration] = [row]
    _, summary = aggregate_paired_sensitivity(values)
    slip = next(
        row
        for row in summary
        if row["metric"] == "paired_delta_persistent_mean_relative_slip_rate_A_per_ps"
    )
    assert slip["positive_count"] == 1
    assert slip["negative_count"] == 1
    assert not slip["same_nonzero_sign_across_configurations"]


def test_paired_blocks_require_every_block_to_share_sign():
    rows = []
    for branch, delta in (("f0_shared", 0.0), ("f8e-5_x", 0.01)):
        for index in range(21):
            rows.append(
                {
                    "case_id": "ch3_only",
                    "branch_id": branch,
                    "axis": "x",
                    "edge": "leading",
                    "time_ps": float(index),
                    "persistent_mean_relative_delta_A": delta,
                }
            )
    blocks, summary = paired_block_slip(
        "primary", rows, window_ps=20.0, block_ps=5.0
    )
    assert len(blocks) == 4
    assert summary[0]["positive_blocks"] == 4
    assert summary[0]["same_nonzero_sign_all_blocks"]
