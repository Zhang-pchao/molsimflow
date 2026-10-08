import csv
import json
from pathlib import Path

from molsimflow.postprocess.tpcl_renewal_rewiring import main


def test_renewal_rewiring_writes_block_summaries_and_figures(tmp_path: Path):
    source = tmp_path / "topology.csv"
    fields = (
        "case_id",
        "primary_event_id",
        "time_block_200ps",
        "lag_frames",
        "lag_ps",
        "metric",
        "topology_did",
        "finite",
    )
    rows = []
    for case_id in ("oh", "ch3"):
        for block in range(3):
            for lag_frames, lag_ps in ((1, 0.5), (2, 1.0), (3, 1.5)):
                for event_id in range(2):
                    rows.extend(
                        (
                            {
                                "case_id": case_id,
                                "primary_event_id": block * 10 + event_id,
                                "time_block_200ps": block,
                                "lag_frames": lag_frames,
                                "lag_ps": lag_ps,
                                "metric": "node_survival_fraction",
                                "topology_did": "-0.20",
                                "finite": "1",
                            },
                            {
                                "case_id": case_id,
                                "primary_event_id": block * 10 + event_id,
                                "time_block_200ps": block,
                                "lag_frames": lag_frames,
                                "lag_ps": lag_ps,
                                "metric": "retained_node_edge_survival_fraction",
                                "topology_did": "-0.05",
                                "finite": "1",
                            },
                        )
                    )
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "output"
    assert (
        main(
            [
                "--event-topology-did",
                str(source),
                "--output-dir",
                str(output),
                "--bootstrap-samples",
                "100",
            ]
        )
        == 0
    )
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert result["case_count"] == 2
    central = list(csv.DictReader((output / "central_lag_summary.csv").open()))
    assert {row["case_id"] for row in central} == {"oh", "ch3"}
    assert (output / "figures" / "01_node_vs_retained_edge_excess.png").is_file()
    assert (output / "figures" / "02_node_vs_retained_edge_block_scatter.png").is_file()
    assert (output / "figures" / "03_renewal_rewiring_lag_profiles.png").is_file()
