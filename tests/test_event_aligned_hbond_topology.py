import csv
import gzip

from molsimflow.postprocess.event_aligned_hbond_topology import run_analysis


def _write(path, rows, delimiter=","):
    opener = gzip.open if path.suffix == ".gz" else path.open
    kwargs = {"newline": "", "encoding": "utf-8"}
    if path.suffix == ".gz":
        handle = opener(path, "wt", **kwargs)
    else:
        handle = opener("w", **kwargs)
    with handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def test_event_topology_separates_node_loss_from_retained_edge_loss(tmp_path):
    windows = []
    frames = []
    nodes = []
    edges = []
    events = []
    for event_id, block in ((1, 0), (2, 1)):
        events.append(
            {"case_id": "x", "primary_event_id": event_id, "time_block_200ps": block}
        )
        for relative in range(-4, 5):
            step = event_id * 100 + relative + 4
            windows.append(
                {"event_id": event_id, "step": step, "relative_frame": relative}
            )
            frames.append({"step": step})
            present = (1, 2)
            if event_id == 2 and relative >= 1:
                present = (1,)
            for oxygen_id in present:
                nodes.append({"step": step, "oxygen_id": oxygen_id, "arc_index": oxygen_id})
            keep_edge = relative <= -1
            if keep_edge:
                edges.append(
                    {
                        "step": step,
                        "donor_id": 1,
                        "acceptor_id": 2,
                        "edge_scope": "induced_tpcl",
                    }
                )

    edge_path = tmp_path / "edges.csv.gz"
    frame_path = tmp_path / "frames.csv"
    node_path = tmp_path / "nodes.csv.gz"
    window_path = tmp_path / "windows.csv"
    event_path = tmp_path / "events.csv"
    _write(edge_path, edges)
    _write(frame_path, frames)
    _write(node_path, nodes)
    _write(window_path, windows)
    _write(event_path, events)
    sources = tmp_path / "sources.tsv"
    _write(
        sources,
        [
            {
                "case_id": "x",
                "edge_table": edge_path,
                "frame_table": frame_path,
                "node_table": node_path,
                "event_window_table": window_path,
            }
        ],
        delimiter="\t",
    )

    output = tmp_path / "output"
    summary = run_analysis(
        sources,
        event_path,
        output,
        bootstrap_draws=100,
        seed=7,
    )
    assert summary["event_count"] == 2
    assert summary["event_metric_row_count"] == 42
    with (output / "event_topology_did.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    node_did = [
        float(row["topology_did"])
        for row in rows
        if row["primary_event_id"] == "2"
        and row["lag_frames"] == "2"
        and row["metric"] == "node_survival_fraction"
    ]
    retained_edge = [
        row
        for row in rows
        if row["primary_event_id"] == "2"
        and row["lag_frames"] == "2"
        and row["metric"] == "retained_node_edge_survival_fraction"
    ]
    assert node_did == [-0.5]
    assert retained_edge[0]["finite"] == "0"
    assert retained_edge[0]["topology_did"] == "nan"
