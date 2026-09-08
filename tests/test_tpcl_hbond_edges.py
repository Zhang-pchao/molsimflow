import csv
import gzip

import numpy as np

from molsimflow.postprocess.local_water_order import SelectedFrame
from molsimflow.postprocess.tpcl_hbond_edges import (
    TpclNode,
    analyze_hbond_frame,
    iter_tpcl_node_frames,
    load_event_steps,
)


def _node(oxygen_id, donor, acceptor):
    return TpclNode(
        oxygen_id=oxygen_id,
        time_ns=1.0,
        arc_index=oxygen_id,
        theta_deg=float(oxygen_id),
        normal_distance_A=0.0,
        tangential_offset_A=0.0,
        surface_distance_A=2.0,
        hbond_donor_count=donor,
        hbond_acceptor_count=acceptor,
        hbond_degree=1,
        hbond_internal_tpcl_degree=1,
    )


def test_analyze_hbond_frame_preserves_directed_edges_and_metric_parity():
    frame = SelectedFrame(
        step=2000000,
        bounds=np.asarray([[0.0, 20.0], [0.0, 20.0], [0.0, 20.0]]),
        surface=np.asarray([[10.0, 10.0, 10.0]]),
        water_oxygen_ids=np.asarray([10, 20]),
        water_oxygen=np.asarray([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]]),
        candidate_oxygen_ids=np.asarray([10, 20]),
        candidate_oxygen=np.asarray([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]]),
        hydrogen=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.8, 0.0, 0.0],
                [2.8, 1.0, 0.0],
            ]
        ),
    )
    rows, summary = analyze_hbond_frame(
        frame,
        {10: _node(10, 1, 1), 20: _node(20, 1, 1)},
        oh_cutoff_A=1.25,
        oo_cutoff_A=3.5,
        hbond_angle_deg=30.0,
    )
    assert [(row["donor_id"], row["acceptor_id"]) for row in rows] == [
        (10, 20),
        (20, 10),
    ]
    assert {row["edge_scope"] for row in rows} == {"induced_tpcl"}
    assert summary["induced_undirected_edge_count"] == 1
    assert summary["sample_metric_parity_pass"] is True


def test_load_event_steps_and_stream_selected_sample_frames(tmp_path):
    events = tmp_path / "events.csv"
    with events.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "event_id"])
        writer.writeheader()
        writer.writerows(
            [
                {"step": 20, "event_id": 1},
                {"step": 20, "event_id": 2},
            ]
        )
    assert load_event_steps(events) == {20}

    samples = tmp_path / "samples.csv.gz"
    fields = [
        "step",
        "time_ns",
        "oxygen_id",
        "arc_index",
        "theta_deg",
        "normal_distance_A",
        "tangential_offset_A",
        "surface_distance_A",
        "hbond_donor_count",
        "hbond_acceptor_count",
        "hbond_degree",
        "hbond_internal_tpcl_degree",
    ]
    with gzip.open(samples, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in (10, 20, 30):
            writer.writerow(
                {
                    "step": step,
                    "time_ns": step / 10.0,
                    "oxygen_id": step,
                    "arc_index": 0,
                    "theta_deg": 0.0,
                    "normal_distance_A": 0.0,
                    "tangential_offset_A": 0.0,
                    "surface_distance_A": 2.0,
                    "hbond_donor_count": 0,
                    "hbond_acceptor_count": 0,
                    "hbond_degree": 0,
                    "hbond_internal_tpcl_degree": 0,
                }
            )
    groups = list(iter_tpcl_node_frames(samples, {20}))
    assert len(groups) == 1
    assert groups[0][0] == 20
    assert set(groups[0][1]) == {20}
