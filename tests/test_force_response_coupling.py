import csv
import math

import pytest

from molsimflow.postprocess.force_response_coupling import run_analysis


def _write(path, rows, delimiter=","):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def test_force_response_join_preserves_missing_support_and_primary_test(tmp_path):
    selection = []
    mechanics = []
    outcomes = []
    pairs = [
        ("a1", "a", 1, 1.0),
        ("a2", "a", 2, 2.0),
        ("a3", "a", 3, 3.0),
        ("b1", "b", 4, 1.0),
        ("b2", "b", 5, 2.0),
        ("b3", "b", 6, 3.0),
    ]
    for pair_id, case_id, event_id, force in pairs:
        selection.append(
            {
                "pair_id": pair_id,
                "case_id": case_id,
                "primary_event_id": event_id,
                "source_time_block_200ps": event_id,
                "response_affected_arc_fraction": force / 4.0,
            }
        )
        for metric, contrast, value in (
            ("radial", "did_transition_minus_pre", force),
            ("tangential", "did_transition_minus_pre", -force),
        ):
            mechanics.append(
                {
                    "pair_id": pair_id,
                    "case_id": case_id,
                    "patch_radius_A": 6.0,
                    "metric": metric,
                    "contrast": contrast,
                    "value": value,
                }
            )
        if event_id == 6:
            continue
        for target, ratio in enumerate((force / 2.0, force, force * 1.5), start=10):
            outcomes.append(
                {
                    "case_id": case_id,
                    "primary_event_id": event_id,
                    "time_block_200ps": event_id,
                    "target_arc_index": target,
                    "distance_bin": "far",
                    "window": "fast",
                    "mobilization_ratio": ratio,
                    "mobilized": ratio >= 1.0,
                    "secondary_event_detected": target == 10 and ratio >= 1.0,
                }
            )
    selection_path = tmp_path / "selection.tsv"
    mechanics_path = tmp_path / "mechanics.csv"
    outcomes_path = tmp_path / "outcomes.csv"
    _write(selection_path, selection, delimiter="\t")
    _write(mechanics_path, mechanics)
    _write(outcomes_path, outcomes)
    output = tmp_path / "output"
    summary = run_analysis(
        mechanics_path,
        selection_path,
        outcomes_path,
        output,
        mechanical_coordinates=(
            ("radial_transition_did", "radial", "did_transition_minus_pre"),
            ("tangential_transition_did", "tangential", "did_transition_minus_pre"),
        ),
        primary_mechanical_label="radial_transition_did",
        permutation_draws=100,
        bootstrap_draws=100,
        seed=7,
    )
    assert summary["selected_mechanical_pair_count"] == 6
    assert summary["joint_support_pair_count"] == 5
    assert summary["excluded_no_outcome_count"] == 1
    with (output / "support_exclusions.csv").open(newline="", encoding="utf-8") as handle:
        excluded = list(csv.DictReader(handle))
    assert [(row["pair_id"], row["primary_event_id"]) for row in excluded] == [("b3", "6")]
    with (output / "force_response_associations.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        associations = list(csv.DictReader(handle))
    primary = [row for row in associations if row["is_primary"] == "1"]
    assert len(primary) == 1
    assert primary[0]["pair_count"] == "5"
    assert float(primary[0]["rank_correlation"]) > 0.0

    for target, ratio in enumerate((0.1, 0.2, 0.3), start=10):
        outcomes.append(
            {
                "case_id": "b",
                "primary_event_id": 6,
                "time_block_200ps": 6,
                "target_arc_index": target,
                "distance_bin": "far",
                "window": "fast",
                "mobilization_ratio": ratio,
                "mobilized": False,
                "secondary_event_detected": False,
            }
        )
    _write(outcomes_path, outcomes)
    complete_output = tmp_path / "complete_output"
    complete_summary = run_analysis(
        mechanics_path,
        selection_path,
        outcomes_path,
        complete_output,
        mechanical_coordinates=(
            ("radial_transition_did", "radial", "did_transition_minus_pre"),
            ("tangential_transition_did", "tangential", "did_transition_minus_pre"),
        ),
        primary_mechanical_label="radial_transition_did",
        permutation_draws=100,
        bootstrap_draws=100,
        seed=7,
    )
    assert complete_summary["joint_support_pair_count"] == 6
    assert complete_summary["excluded_no_outcome_count"] == 0
    with (complete_output / "support_exclusions.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        assert list(csv.DictReader(handle)) == []
    with (complete_output / "pair_force_response_table.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        complete_pairs = list(csv.DictReader(handle))
    unsupported_conversion = next(row for row in complete_pairs if row["pair_id"] == "b3")
    assert math.isnan(float(unsupported_conversion["far_fast_conversion_fraction"]))

    mismatched_mechanics = [
        {**row, "case_id": "wrong"} if row["pair_id"] == "a1" else row
        for row in mechanics
    ]
    _write(mechanics_path, mismatched_mechanics)
    with pytest.raises(ValueError, match="mechanics/selection case mismatch"):
        run_analysis(
            mechanics_path,
            selection_path,
            outcomes_path,
            tmp_path / "mismatch_output",
            mechanical_coordinates=(
                ("radial_transition_did", "radial", "did_transition_minus_pre"),
                ("tangential_transition_did", "tangential", "did_transition_minus_pre"),
            ),
            primary_mechanical_label="radial_transition_did",
            permutation_draws=100,
            bootstrap_draws=100,
            seed=7,
        )
