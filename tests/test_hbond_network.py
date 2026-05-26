import csv

from molsimflow.postprocess.hbond_network import (
    HbondNetworkConfig,
    analyze_hbond_network,
    build_frame_network_summary,
    classify_hbond_type,
    load_hbond_edge_rows,
    summarize_hbond_lifetimes,
)


def _write_hbond_edges(path):
    rows = [
        {
            "frame": 0,
            "time": 0.0,
            "donor_id": 1,
            "acceptor_id": 2,
            "hbond_type": "water_water",
            "donor_s_A": -1.0,
            "acceptor_s_A": 1.0,
            "surface_gap_A": 2.0,
        },
        {
            "frame": 0,
            "time": 0.0,
            "donor_id": 2,
            "acceptor_id": 3,
            "hbond_type": "water_h3o",
            "donor_s_A": 1.0,
            "acceptor_s_A": 1.2,
            "surface_gap_A": 2.0,
        },
        {
            "frame": 1,
            "time": 0.1,
            "donor_id": 1,
            "acceptor_id": 2,
            "hbond_type": "water_water",
            "donor_s_A": -1.0,
            "acceptor_s_A": 1.0,
            "surface_gap_A": 4.0,
        },
        {
            "frame": 1,
            "time": 0.1,
            "donor_id": 4,
            "acceptor_id": 5,
            "hbond_type": "water_cl",
            "donor_s_A": -0.8,
            "acceptor_s_A": -0.7,
            "surface_gap_A": 4.0,
        },
        {
            "frame": 2,
            "time": 0.2,
            "donor_id": 1,
            "acceptor_id": 3,
            "hbond_type": "water_water",
            "donor_s_A": -1.0,
            "acceptor_s_A": 1.2,
            "surface_gap_A": 6.0,
        },
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_classify_hbond_type_from_species():
    assert classify_hbond_type("h2o", "h2o") == "water_water"
    assert classify_hbond_type("h3o", "h2o") == "h3o_water"
    assert classify_hbond_type("h2o", "oh") == "water_oh"
    assert classify_hbond_type("unknown", "h2o") is None


def test_hbond_network_frame_and_lifetime_summaries(tmp_path):
    edge_path = tmp_path / "hbond_edges.csv"
    _write_hbond_edges(edge_path)

    edges = load_hbond_edge_rows(edge_path, donor_s_column="donor_s_A", acceptor_s_column="acceptor_s_A", gap_column="surface_gap_A")
    frame_summary = build_frame_network_summary(
        edges,
        case_label="caseA",
        config=HbondNetworkConfig(bridge_s_min_A=-1.0, bridge_s_max_A=1.0, side_thickness_A=0.2),
    )
    lifetimes = summarize_hbond_lifetimes(edges)

    assert frame_summary[0]["n_nodes"] == 3
    assert frame_summary[0]["n_hbond_total"] == 2
    assert frame_summary[0]["largest_hbond_component_fraction"] == 1.0
    assert frame_summary[0]["hbond_network_spanning_axial"] == 1
    assert frame_summary[1]["largest_hbond_component_fraction"] == 0.5

    lifetime_by_type = {row["hbond_type"]: row for row in lifetimes}
    assert lifetime_by_type["water_water"]["n_unique_runs"] == 2
    assert lifetime_by_type["water_water"]["dt_per_processed_frame_ps"] == 100.0


def test_analyze_hbond_network_writes_outputs(tmp_path):
    edge_path = tmp_path / "hbond_edges.csv"
    _write_hbond_edges(edge_path)

    outputs = analyze_hbond_network(
        input_csv=edge_path,
        output_dir=tmp_path / "hbond",
        case_label="caseA",
        config=HbondNetworkConfig(
            bridge_s_min_A=-1.0,
            bridge_s_max_A=1.0,
            side_thickness_A=0.2,
            gap_bin_width_A=2.0,
        ),
        donor_s_column="donor_s_A",
        acceptor_s_column="acceptor_s_A",
        gap_column="surface_gap_A",
    )

    assert outputs["edge_table"].exists()
    assert outputs["frame_summary"].exists()
    assert outputs["lifetime_summary"].exists()
    assert outputs["gap_summary"].exists()
    assert outputs["state_statistics"].exists()

    with outputs["frame_summary"].open() as handle:
        frames = list(csv.DictReader(handle))
    assert frames[0]["case_label"] == "caseA"
    assert frames[0]["n_hbond_water_water"] == "1"

    with outputs["state_statistics"].open() as handle:
        stats = {row["metric"]: row["value"] for row in csv.DictReader(handle)}
    assert stats["n_edge_rows"] == "5"
    assert stats["n_frames"] == "3"
