import csv

from molsimflow.postprocess.contact_graph import (
    ContactGraphConfig,
    analyze_contact_graph,
    build_contact_graph_summary,
    load_contact_edge_rows,
)


def _write_contact_edges(path):
    rows = [
        {
            "frame": 0,
            "time": 0.0,
            "source_id": 1,
            "target_id": 2,
            "edge_type": "water_water",
            "source_role": "water",
            "target_role": "water",
            "source_region": "bridge",
            "target_region": "bridge",
            "source_s_A": -1.0,
            "target_s_A": 0.0,
            "surface_gap_A": 2.0,
        },
        {
            "frame": 0,
            "time": 0.0,
            "source_id": 2,
            "target_id": 3,
            "edge_type": "water_ion",
            "source_role": "water",
            "target_role": "ion",
            "source_region": "bridge",
            "target_region": "bridge",
            "source_s_A": 0.0,
            "target_s_A": 1.0,
            "surface_gap_A": 2.0,
        },
        {
            "frame": 1,
            "time": 0.1,
            "source_id": 1,
            "target_id": 2,
            "edge_type": "water_water",
            "source_role": "water",
            "target_role": "water",
            "source_region": "bridge",
            "target_region": "bridge",
            "source_s_A": -1.0,
            "target_s_A": -0.5,
            "surface_gap_A": 4.0,
        },
        {
            "frame": 1,
            "time": 0.1,
            "source_id": 4,
            "target_id": 5,
            "edge_type": "water_surface",
            "source_role": "water",
            "target_role": "surface",
            "source_region": "bridge",
            "target_region": "surface",
            "source_s_A": 0.5,
            "target_s_A": 0.7,
            "surface_gap_A": 4.0,
        },
        {
            "frame": 2,
            "time": 0.2,
            "source_id": 1,
            "target_id": 2,
            "edge_type": "water_water",
            "source_role": "water",
            "target_role": "water",
            "source_region": "bridge",
            "target_region": "bridge",
            "source_s_A": -1.0,
            "target_s_A": 0.0,
            "surface_gap_A": 6.0,
        },
        {
            "frame": 2,
            "time": 0.2,
            "source_id": 2,
            "target_id": 3,
            "edge_type": "water_water",
            "source_role": "water",
            "target_role": "water",
            "source_region": "bridge",
            "target_region": "bridge",
            "source_s_A": 0.0,
            "target_s_A": 1.0,
            "surface_gap_A": 6.0,
        },
        {
            "frame": 2,
            "time": 0.2,
            "source_id": 3,
            "target_id": 1,
            "edge_type": "water_water",
            "source_role": "water",
            "target_role": "water",
            "source_region": "bridge",
            "target_region": "bridge",
            "source_s_A": 1.0,
            "target_s_A": -1.0,
            "surface_gap_A": 6.0,
        },
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_contact_graph_frame_summary(tmp_path):
    edge_path = tmp_path / "contact_edges.csv"
    _write_contact_edges(edge_path)

    edges = load_contact_edge_rows(
        edge_path,
        source_s_column="source_s_A",
        target_s_column="target_s_A",
        gap_column="surface_gap_A",
    )
    summary = build_contact_graph_summary(
        edges,
        case_label="caseA",
        config=ContactGraphConfig(bridge_s_min_A=-1.0, bridge_s_max_A=1.0, side_thickness_A=0.2),
    )

    assert summary[0]["n_nodes"] == 3
    assert summary[0]["n_edges"] == 2
    assert summary[0]["n_components"] == 1
    assert summary[0]["largest_component_fraction_bridge"] == 1.0
    assert summary[0]["bridge_spanning_flag"] == 1
    assert summary[0]["ion_mediated_edge_fraction"] == 0.5
    assert summary[0]["articulation_water_count"] == 1
    assert summary[1]["n_components"] == 2
    assert summary[2]["cycle_rank"] == 1


def test_analyze_contact_graph_writes_outputs(tmp_path):
    edge_path = tmp_path / "contact_edges.csv"
    _write_contact_edges(edge_path)

    outputs = analyze_contact_graph(
        input_csv=edge_path,
        output_dir=tmp_path / "contact_graph",
        case_label="caseA",
        config=ContactGraphConfig(
            bridge_s_min_A=-1.0,
            bridge_s_max_A=1.0,
            side_thickness_A=0.2,
            gap_bin_width_A=2.0,
        ),
        source_s_column="source_s_A",
        target_s_column="target_s_A",
        gap_column="surface_gap_A",
    )

    assert outputs["edge_table"].exists()
    assert outputs["frame_summary"].exists()
    assert outputs["gap_summary"].exists()
    assert outputs["state_statistics"].exists()

    with outputs["frame_summary"].open() as handle:
        frames = list(csv.DictReader(handle))
    assert frames[0]["case_label"] == "caseA"
    assert frames[2]["cycle_rank"] == "1"

    with outputs["state_statistics"].open() as handle:
        stats = {row["metric"]: row["value"] for row in csv.DictReader(handle)}
    assert stats["n_edge_rows"] == "7"
    assert stats["n_frames"] == "3"
