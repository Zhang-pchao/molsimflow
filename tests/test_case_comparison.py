import csv
import math

from molsimflow.postprocess.case_comparison import (
    CasePairSpec,
    DescriptorTableSpec,
    analyze_case_scorecard,
    build_case_scorecard,
    compute_case_deltas,
    compute_correlations,
    load_case_manifest,
    load_descriptor_manifest,
)


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_case_scorecard_join_delta_and_correlation(tmp_path):
    cases = tmp_path / "cases.csv"
    _write_csv(
        cases,
        ["case_label", "group"],
        [
            {"case_label": "A", "group": "same"},
            {"case_label": "B", "group": "same"},
            {"case_label": "C", "group": "same"},
        ],
    )
    barriers = tmp_path / "barriers.csv"
    _write_csv(
        barriers,
        ["case_label", "barrier_kjmol"],
        [
            {"case_label": "A", "barrier_kjmol": "1.0"},
            {"case_label": "B", "barrier_kjmol": "2.0"},
            {"case_label": "C", "barrier_kjmol": "4.0"},
        ],
    )
    bridge = tmp_path / "bridge.csv"
    _write_csv(
        bridge,
        ["case_label", "bridge_waters"],
        [
            {"case_label": "A", "bridge_waters": "6.0"},
            {"case_label": "B", "bridge_waters": "4.0"},
            {"case_label": "C", "bridge_waters": "0.0"},
        ],
    )

    case_rows = load_case_manifest(cases)
    scorecard, manifest_rows, descriptor_columns = build_case_scorecard(
        case_rows,
        [
            DescriptorTableSpec("barrier", barriers, columns=("barrier_kjmol",)),
            DescriptorTableSpec("bridge", bridge, columns=("bridge_waters",)),
        ],
    )

    assert descriptor_columns == ["barrier__barrier_kjmol", "bridge__bridge_waters"]
    assert len(manifest_rows) == 3
    assert scorecard[0]["barrier__barrier_kjmol"] == 1.0
    assert scorecard[2]["bridge__bridge_waters"] == 0.0

    deltas = compute_case_deltas(
        scorecard,
        [CasePairSpec("A", "C", "C_minus_A")],
        ["barrier__barrier_kjmol", "bridge__bridge_waters"],
    )
    barrier_delta = next(row for row in deltas if row["descriptor"] == "barrier__barrier_kjmol")
    bridge_delta = next(row for row in deltas if row["descriptor"] == "bridge__bridge_waters")
    assert math.isclose(barrier_delta["delta_target_minus_reference"], 3.0)
    assert math.isclose(bridge_delta["delta_target_minus_reference"], -6.0)

    correlations = compute_correlations(
        scorecard,
        target_column="barrier__barrier_kjmol",
        descriptor_columns=["bridge__bridge_waters"],
    )
    assert len(correlations) == 1
    assert math.isclose(correlations[0]["pearson_r"], -1.0)
    assert math.isclose(correlations[0]["spearman_r"], -1.0)


def test_descriptor_manifest_relative_paths_and_file_outputs(tmp_path):
    cases = tmp_path / "cases.csv"
    descriptors = tmp_path / "descriptors.csv"
    descriptor_manifest = tmp_path / "descriptor_manifest.csv"
    _write_csv(
        cases,
        ["case_label"],
        [
            {"case_label": "A"},
            {"case_label": "B"},
        ],
    )
    _write_csv(
        descriptors,
        ["case_label", "metric_a", "metric_b"],
        [
            {"case_label": "A", "metric_a": "1", "metric_b": "5"},
            {"case_label": "B", "metric_a": "3", "metric_b": "9"},
        ],
    )
    _write_csv(
        descriptor_manifest,
        ["name", "path", "case_column", "columns"],
        [
            {
                "name": "metrics",
                "path": descriptors.name,
                "case_column": "case_label",
                "columns": "metric_a,metric_b",
            }
        ],
    )

    specs = load_descriptor_manifest(descriptor_manifest)
    outputs = analyze_case_scorecard(
        case_manifest=cases,
        descriptor_specs=specs,
        output_dir=tmp_path / "out",
        pair_specs=[CasePairSpec("A", "B")],
        target_column="metrics__metric_b",
        correlate_columns=["metrics__metric_a"],
    )

    assert outputs["scorecard"].exists()
    assert outputs["delta"].exists()
    assert outputs["correlation"].exists()
    assert outputs["manifest"].exists()

    with outputs["delta"].open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["case_pair_label"] == "B_minus_A"
    assert rows[0]["descriptor"] == "metrics__metric_a"
    assert math.isclose(float(rows[0]["delta_target_minus_reference"]), 2.0)
