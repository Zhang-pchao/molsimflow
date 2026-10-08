import csv
import math

import pytest

from molsimflow.postprocess.event_aligned_mode_attribution import Bin, run_analysis

FIELDS = ("total", "residual", "low", "mean")


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _inputs(tmp_path, *, break_closure=False):
    event_rows = []
    for event_id, block, scale in ((1, 0, 1.0), (2, 1, 2.0)):
        values = {"total": 2.0 * scale, "residual": scale, "low": 0.5 * scale, "mean": 0.5 * scale}
        for lag in (1.0, 2.0):
            for distance in (1, 2):
                for field in FIELDS:
                    value = values[field]
                    if break_closure and event_id == 1 and lag == 1.0 and distance == 1 and field == "total":
                        value += 1.0
                    event_rows.append(
                        {
                            "case_id": "x",
                            "event_id": event_id,
                            "time_block_200ps": block,
                            "field": field,
                            "lag_ps": lag,
                            "arc_distance": distance,
                            "aligned_change": value,
                        }
                    )
    null_rows = []
    null_values = {"total": 0.2, "residual": 0.1, "low": 0.05, "mean": 0.05}
    for lag in (1.0, 2.0):
        for distance in (1, 2):
            for field in FIELDS:
                null_rows.append(
                    {
                        "case_id": "x",
                        "field": field,
                        "lag_ps": lag,
                        "arc_distance": distance,
                        "null_mean": null_values[field],
                    }
                )
    event_path, map_path = tmp_path / "events.csv", tmp_path / "map.csv"
    _write(event_path, event_rows)
    _write(map_path, null_rows)
    return event_path, map_path


def test_mode_attribution_closes_and_partitions_primary_effect(tmp_path):
    event_path, map_path = _inputs(tmp_path)
    output = tmp_path / "out"
    summary = run_analysis(
        event_path,
        map_path,
        output,
        total_field="total",
        component_fields=("residual", "low", "mean"),
        regions=(Bin("near", 1.0, 3.0),),
        windows=(Bin("fast", 0.0, 2.0),),
        primary_region="near",
        primary_window="fast",
        bootstrap_samples=40,
        random_seed=7,
    )
    with (output / "primary_mode_attribution.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["field"]: row for row in csv.DictReader(handle)}
    assert summary["event_count"] == 2
    assert summary["primary_row_count"] == 3
    assert summary["maximum_additive_closure_error_A"] < 1.0e-12
    assert math.isclose(float(rows["residual"]["signed_fraction_of_total_effect"]), 0.5)
    assert math.isclose(float(rows["low"]["signed_fraction_of_total_effect"]), 0.25)
    assert math.isclose(float(rows["mean"]["signed_fraction_of_total_effect"]), 0.25)


def test_mode_attribution_rejects_nonadditive_source(tmp_path):
    event_path, map_path = _inputs(tmp_path, break_closure=True)
    with pytest.raises(ValueError, match="closure"):
        run_analysis(
            event_path,
            map_path,
            tmp_path / "out",
            total_field="total",
            component_fields=("residual", "low", "mean"),
            regions=(Bin("near", 1.0, 3.0),),
            windows=(Bin("fast", 0.0, 2.0),),
            primary_region="near",
            primary_window="fast",
            bootstrap_samples=20,
            random_seed=7,
        )
