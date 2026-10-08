import csv
import math

import pytest

from molsimflow.postprocess.event_aligned_response_susceptibility import run_analysis


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _inputs(tmp_path, *, drop_null=False, drop_event=False):
    event_rows = []
    null_rows = []
    for case_id, fast_near, fast_far, slow_far in (
        ("a", 2.0, 1.0, 1.0),
        ("b", 1.0, 3.0, 0.0),
    ):
        values = {(1.0, 1): fast_near, (1.0, 2): fast_far, (10.0, 1): 0.0, (10.0, 2): slow_far}
        for event_id, block in ((1, 0), (2, 1)):
            for (lag, distance), value in values.items():
                for offset in (-distance, distance):
                    if (
                        drop_event
                        and case_id == "b"
                        and event_id == 2
                        and lag == 10.0
                        and offset == 2
                    ):
                        continue
                    event_rows.append(
                        {
                            "case_id": case_id,
                            "event_id": event_id,
                            "time_block_200ps": block,
                            "field": "residual",
                            "lag_ps": lag,
                            "arc_offset_signed": offset,
                            "arc_distance": distance,
                            "aligned_change": value,
                        }
                    )
        for lag, distance in values:
            for offset in (-distance, distance):
                if not (drop_null and case_id == "b" and lag == 10.0 and offset == 2):
                    null_rows.append(
                        {
                            "case_id": case_id,
                            "field": "residual",
                            "lag_ps": lag,
                            "arc_offset_signed": offset,
                            "arc_distance": distance,
                            "null_mean": 0.0,
                        }
                    )
    event_path, map_path = tmp_path / "events.csv", tmp_path / "map.csv"
    _write(event_path, event_rows)
    _write(map_path, null_rows)
    return event_path, map_path


def test_response_shape_metrics_and_contrasts(tmp_path):
    event_path, map_path = _inputs(tmp_path)
    output = tmp_path / "out"
    summary = run_analysis(
        event_path,
        map_path,
        output,
        field="residual",
        fast_window=(0.0, 5.0),
        slow_window=(5.0, 50.0),
        minimum_distance=1,
        far_minimum_distance=2,
        bootstrap_samples=40,
        random_seed=7,
    )
    with (output / "response_shape_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = {(row["case_id"], row["metric"]): row for row in csv.DictReader(handle)}
    assert summary["case_count"] == 2
    assert summary["event_count"] == 4
    assert math.isclose(float(rows[("a", "fast_spatial_extent_arcs")]["value"]), 4.0 / 3.0)
    assert math.isclose(float(rows[("a", "fast_far_response_fraction")]["value"]), 1.0 / 3.0)
    assert math.isclose(float(rows[("a", "far_slow_response_fraction")]["value"]), 0.5)
    assert math.isclose(float(rows[("b", "fast_far_response_fraction")]["value"]), 0.75)
    with (output / "response_shape_contrasts.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        contrasts = {row["metric"]: row for row in csv.DictReader(handle)}
    assert math.isclose(
        float(contrasts["fast_far_response_fraction"]["case_a_minus_case_b"]),
        1.0 / 3.0 - 0.75,
    )


def test_response_shape_rejects_missing_null_cell(tmp_path):
    event_path, map_path = _inputs(tmp_path, drop_null=True)
    with pytest.raises(ValueError, match="cell sets differ"):
        run_analysis(
            event_path,
            map_path,
            tmp_path / "out",
            field="residual",
            fast_window=(0.0, 5.0),
            slow_window=(5.0, 50.0),
            minimum_distance=1,
            far_minimum_distance=2,
            bootstrap_samples=20,
            random_seed=7,
        )


def test_response_shape_rejects_incomplete_event_cells(tmp_path):
    event_path, map_path = _inputs(tmp_path, drop_event=True)
    with pytest.raises(ValueError, match="incomplete event-cell coverage"):
        run_analysis(
            event_path,
            map_path,
            tmp_path / "out",
            field="residual",
            fast_window=(0.0, 5.0),
            slow_window=(5.0, 50.0),
            minimum_distance=1,
            far_minimum_distance=2,
            bootstrap_samples=20,
            random_seed=7,
        )
