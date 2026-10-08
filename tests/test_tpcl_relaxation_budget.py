import csv
from pathlib import Path

from molsimflow.postprocess.tpcl_relaxation_budget import main


def test_relaxation_budget_writes_block_summary_and_figures(tmp_path: Path):
    source = tmp_path / "event_modes.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case_id",
                "time_block_200ps",
                "region",
                "window",
                "mean_radius_component_A",
            ),
        )
        writer.writeheader()
        writer.writerows(
            (
                {
                    "case_id": "oh",
                    "time_block_200ps": 0,
                    "region": "far",
                    "window": "fast",
                    "mean_radius_component_A": -1.0,
                },
                {
                    "case_id": "oh",
                    "time_block_200ps": 1,
                    "region": "far",
                    "window": "fast",
                    "mean_radius_component_A": 2.0,
                },
                {
                    "case_id": "ch3",
                    "time_block_200ps": 0,
                    "region": "far",
                    "window": "fast",
                    "mean_radius_component_A": 0.5,
                },
                {
                    "case_id": "ch3",
                    "time_block_200ps": 1,
                    "region": "far",
                    "window": "fast",
                    "mean_radius_component_A": -0.5,
                },
            )
        )
    output = tmp_path / "output"
    assert (
        main(
            [
                "--event-mode-attribution",
                str(source),
                "--output-dir",
                str(output),
                "--bootstrap-samples",
                "100",
            ]
        )
        == 0
    )
    summary = list(csv.DictReader((output / "case_budget_summary.csv").open()))
    assert {row["case_id"] for row in summary} == {"oh", "ch3"}
    assert (output / "figures" / "01_rate_amplitude_packetization.png").is_file()
    assert (output / "figures" / "02_relaxation_budget_proxy.png").is_file()
    assert (output / "figures" / "03_relaxation_budget_coverage.png").is_file()
