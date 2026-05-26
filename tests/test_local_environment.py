import csv

from molsimflow.postprocess.local_environment import (
    LocalEnvironmentConfig,
    analyze_local_environment,
    build_class_environment_summary,
    build_frame_environment_summary,
    load_local_environment_rows,
    parse_class_order,
)


def _write_local_environment(path):
    rows = [
        {"frame": 0, "time_ns": 0.0, "entity_id": 1, "environment_class": "tetrahedral", "q": 0.8, "lsi": 0.2},
        {"frame": 0, "time_ns": 0.0, "entity_id": 2, "environment_class": "interfacial", "q": 0.4, "lsi": 0.6},
        {"frame": 1, "time_ns": 0.1, "entity_id": 1, "environment_class": "interfacial", "q": 0.5, "lsi": 0.5},
        {"frame": 1, "time_ns": 0.1, "entity_id": 2, "environment_class": "interfacial", "q": 0.3, "lsi": 0.7},
        {"frame": 2, "time_ns": 0.2, "entity_id": 1, "environment_class": "distorted", "q": 0.2, "lsi": 0.9},
        {"frame": 2, "time_ns": 0.2, "entity_id": 2, "environment_class": "interfacial", "q": 0.3, "lsi": 0.8},
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_local_environment_summaries(tmp_path):
    input_path = tmp_path / "local_environment.csv"
    _write_local_environment(input_path)

    rows, _fieldnames, _selected = load_local_environment_rows(
        input_path,
        time_column="time_ns",
    )
    class_order = parse_class_order("tetrahedral,interfacial,distorted")
    frame_summary = build_frame_environment_summary(rows, class_order, feature_columns=("q", "lsi"))
    class_summary = build_class_environment_summary(rows, class_order, feature_columns=("q", "lsi"))

    assert frame_summary[0]["class_count__tetrahedral"] == 1
    assert frame_summary[1]["class_fraction__interfacial"] == 1.0
    by_class = {row["environment_class"]: row for row in class_summary}
    assert by_class["interfacial"]["count"] == 4
    assert by_class["interfacial"]["n_entities"] == 2


def test_analyze_local_environment_writes_outputs(tmp_path):
    input_path = tmp_path / "local_environment.csv"
    _write_local_environment(input_path)

    outputs = analyze_local_environment(
        input_csv=input_path,
        output_dir=tmp_path / "local_environment",
        config=LocalEnvironmentConfig(
            class_order=parse_class_order("tetrahedral,interfacial,distorted"),
        ),
        time_column="time_ns",
        feature_columns=("q", "lsi"),
    )

    assert outputs["sample_table"].exists()
    assert outputs["frame_summary"].exists()
    assert outputs["class_summary"].exists()
    assert outputs["transition_counts"].exists()
    assert outputs["transition_probabilities"].exists()

    with outputs["transition_counts"].open() as handle:
        counts = list(csv.DictReader(handle))
    by_transition = {(row["from_species"], row["to_species"]): int(row["count"]) for row in counts}

    assert by_transition[("tetrahedral", "interfacial")] == 1
    assert by_transition[("interfacial", "interfacial")] == 2
    assert by_transition[("interfacial", "distorted")] == 1

    with outputs["state_statistics"].open() as handle:
        stats = {row["metric"]: row["value"] for row in csv.DictReader(handle)}
    assert stats["n_samples"] == "6"
    assert stats["n_entities"] == "2"
