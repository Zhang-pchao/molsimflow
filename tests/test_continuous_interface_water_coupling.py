import csv
import gzip
import json
import math
from argparse import Namespace

from molsimflow.postprocess.continuous_interface_water_coupling import run_analysis


def _write(path, rows, *, compressed=False):
    if compressed:
        with gzip.open(path, "wt", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_continuous_increment_coupling_and_residence(tmp_path):
    driver_values = (0, 1, 0, 2, 1, 4, 2, 5, 3, 7, 4, 8)
    response_values = (0, *driver_values[:-1])
    drivers = [
        {"step": index, "time_ns": index * 0.0005, "shape": value}
        for index, value in enumerate(driver_values)
    ]
    responses = []
    for index, value in enumerate(response_values):
        for arc in (0, 1):
            site = "A" if arc == 0 and index < 3 else "B" if arc == 0 else "C"
            responses.append(
                {
                    "step": index,
                    "time_ns": index * 0.0005,
                    "arc_index": arc,
                    "nearest_site_id": site,
                    "nearest_site_type": "SiOH",
                    "water_order": value + (0.1 if arc == 0 else -0.1),
                }
            )
    driver_path, response_path = tmp_path / "driver.csv", tmp_path / "response.csv.gz"
    _write(driver_path, drivers)
    _write(response_path, responses, compressed=True)
    output = tmp_path / "out"
    summary = run_analysis(
        Namespace(
            case_id="case",
            system_kind="nanodroplet",
            driver_table=driver_path,
            response_table=response_path,
            driver_metrics="shape",
            response_metrics="water_order",
            lag_frames="-1,0,1",
            block_frames=2,
            null_samples=20,
            random_seed=7,
            residence_group="arc_index",
            residence_id="nearest_site_id",
            residence_type="nearest_site_type",
            output_dir=output,
        )
    )

    with (output / "lagged_increment_coupling.csv").open() as handle:
        coupling = list(csv.DictReader(handle))
    lag_one = next(row for row in coupling if row["lag_frames"] == "1")
    assert math.isclose(float(lag_one["observed_pearson_r"]), 1.0)
    with (output / "nearest_site_residence_segments.csv").open() as handle:
        segments = list(csv.DictReader(handle))
    assert len(segments) == 3
    assert sorted(float(row["residence_ps"]) for row in segments) == [1.5, 4.5, 6.0]
    assert summary["frame_count"] == 12
    assert summary["residence_segment_count"] == 3
    assert json.loads((output / "manifest.json").read_text())["transform"] == "first_difference"
