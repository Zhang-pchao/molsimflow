from __future__ import annotations

import pytest

from molsimflow.postprocess.constant_force_stage_b_flux_synthesis import (
    _aggregate_interval_windows,
    _category_summary,
    _response_rows,
    _size_class,
)


def _interval(branch: str, direction: str, total: float, transfer: float) -> dict[str, str]:
    row = {
        "case_id": "case",
        "branch_id": branch,
        "direction": direction,
        "start_step": "0",
        "end_step": "1",
        "mid_time_ps": "25",
        "interval_ps": "50",
        "oxygen_count": "10",
    }
    for axis in ("x", "y"):
        row[f"total_{axis}_displacement_A"] = str(total if axis == "x" else 0.0)
        row[f"unchanged_track_{axis}_displacement_A"] = str(
            total - transfer if axis == "x" else 0.0
        )
        row[f"persistent_island_transfer_{axis}_displacement_A"] = str(
            transfer if axis == "x" else 0.0
        )
        row[f"lineage_reassignment_{axis}_displacement_A"] = "0"
        row[f"untracked_transition_{axis}_displacement_A"] = "0"
        row[f"net_{axis}_crossings"] = str(total / 10.0 if axis == "x" else 0.0)
    return row


def test_response_and_fraction_denominators_remain_explicit() -> None:
    intervals = [
        _interval("f0", "none", 10.0, 2.0),
        _interval("fx", "x", 30.0, 12.0),
    ]
    windows = _aggregate_interval_windows(intervals, 4000.0)
    response = _response_rows(windows)
    categories = _category_summary(response)

    persistent = next(row for row in categories if row["category"] == "PERSISTENT_ISLAND_TRANSFER")
    assert persistent["response_velocity_mps"] == pytest.approx(2.0)
    assert persistent["total_response_velocity_mps"] == pytest.approx(4.0)
    assert persistent["signed_fraction_of_total_response"] == pytest.approx(0.5)
    assert persistent["absolute_component_l1_denominator_mps"] == pytest.approx(4.0)
    assert persistent["absolute_fraction_of_component_l1"] == pytest.approx(0.5)


def test_size_class_uses_previous_membership_and_largest_track() -> None:
    assert _size_class(100, 1, 1) == "MAIN_CONDENSED_ISLAND"
    assert _size_class(2, 2, 1) == "OTHER_MULTI_ISLAND"
    assert _size_class(1, 3, 1) == "SINGLETON_VAPOR"
    assert _size_class(None, None, 1) == "UNTRACKED"
