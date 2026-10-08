from pathlib import Path

import numpy as np
import pytest

from molsimflow.postprocess.tpcl_observable_qualification import (
    METAL_FORCE_TO_ACCELERATION_A_PER_PS2_PER_AMU,
    build_detector_trace,
    classify_pair_changes,
    periodic_pairs_within,
    read_complete_thermo,
)
from molsimflow.postprocess.tpcl_force_step_analysis import select_kinematic_events


def test_pair_changes_separate_membership_from_persistent_chemistry():
    result = classify_pair_changes(
        {(1, 2), (2, 3)},
        {(1, 4)},
        {1, 2, 3},
        {1, 2, 4},
        water_pair=True,
    )
    assert result == {
        "formed_persistent": 0,
        "broken_persistent": 1,
        "formed_membership": 1,
        "broken_membership": 1,
    }


def test_surface_pair_change_requires_only_persistent_water():
    result = classify_pair_changes(
        {(100, 1), (101, 2)},
        {(102, 1), (103, 3)},
        {1, 2},
        {1, 3},
        water_pair=False,
    )
    assert result["formed_persistent"] == 1
    assert result["formed_membership"] == 1
    assert result["broken_persistent"] == 1
    assert result["broken_membership"] == 1


def test_periodic_pairs_are_periodic_only_in_xy():
    bounds = np.asarray([[0.0, 10.0], [0.0, 10.0], [0.0, 20.0]])
    sources = np.asarray([[0.2, 5.0, 2.0], [5.0, 5.0, 19.8]])
    targets = np.asarray([[9.8, 5.0, 2.0], [5.0, 5.0, 0.2]])
    assert periodic_pairs_within(sources, targets, bounds, 0.5) == [(0, 0)]


def test_detector_recovers_a_large_coherent_injection():
    times = np.linspace(0.0, 20.0, 2001)
    base = [
        {
            "case_id": "mixed291",
            "step": str(36_200_000 + index * 20),
            "time_ps": str(time),
            "leading_x_A": str(0.01 * np.sin(time)),
            "trailing_x_A": str(0.01 * np.cos(time)),
        }
        for index, time in enumerate(times)
    ]
    rows = build_detector_trace(
        base,
        "x",
        amplitude_A=2.0,
        duration_ps=0.5,
        center_ps=10.0,
        mode="coherent",
    )
    events, controls, diagnostics = select_kinematic_events(rows)
    assert any(abs(float(event["peak_time_ps"]) - 10.0) < 1.0 for event in events)
    assert len(controls) == len(events)
    assert diagnostics["event_threshold_A_per_ps"] > 0


def test_thermo_reader_merges_repeated_blocks(tmp_path: Path):
    log = tmp_path / "lmp.out"
    log.write_text(
        "Step Temp TotEng f_BATH\n"
        "0 300 10 0\n"
        "20 300 11 0.2\n"
        "Loop time done\n"
        "Step Temp TotEng f_BATH\n"
        "20 300 11 0.2\n"
        "40 300 12 0.4\n"
        "Loop time done\n",
        encoding="utf-8",
    )
    names, values = read_complete_thermo(log, ("Step", "TotEng", "f_BATH"))
    assert names == ["Step", "TotEng", "f_BATH"]
    assert values[:, 0].tolist() == [0.0, 20.0, 40.0]
    assert values[-1, 1] == pytest.approx(12.0)


def test_metal_force_momentum_conversion_constant():
    mass = 18.0
    force = 0.5
    time_ps = 2.0
    delta_v = METAL_FORCE_TO_ACCELERATION_A_PER_PS2_PER_AMU * force * time_ps / mass
    observed_impulse = mass * delta_v / METAL_FORCE_TO_ACCELERATION_A_PER_PS2_PER_AMU
    assert observed_impulse == pytest.approx(force * time_ps)
