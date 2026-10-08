import csv
import gzip
import json

import numpy as np
import pytest

from molsimflow.postprocess.tpcl_water_turnover_prediction import (
    InputColumns,
    MembershipFrame,
    TurnoverConfig,
    _bh_adjust,
    _permuted_features,
    _weighted_auc,
    deduplicate_risk_rows,
    enrich_turnover_features,
    iter_membership_frames,
    patch_membership,
    run_analysis,
)


def _write_csv(path, rows, *, delimiter=","):
    opener = gzip.open if path.suffix == ".gz" else open
    kwargs = {"newline": "", "encoding": "utf-8"}
    if path.suffix == ".gz":
        kwargs["mode"] = "wt"
    else:
        kwargs["mode"] = "w"
    with opener(path, **kwargs) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def test_periodic_patch_membership():
    frame = MembershipFrame(
        step=10,
        by_arc={0: frozenset({1}), 1: frozenset({2}), 3: frozenset({4})},
    )
    assert patch_membership(frame, 0, arc_count=4, arc_half_width=1) == {1, 2, 4}


def test_deduplicate_rejects_conflicting_anchor_labels():
    columns = InputColumns()
    rows = [
        {
            "case_id": "a",
            "primary_arc_index": "0",
            "sample_step": "10",
            "sample_pre_step": "0",
            "sample_time_block_200ps": "0",
            "is_event": "0",
            "risk_set_weight": "1",
            "static": "0",
        },
        {
            "case_id": "a",
            "primary_arc_index": "0",
            "sample_step": "10",
            "sample_pre_step": "0",
            "sample_time_block_200ps": "0",
            "is_event": "1",
            "risk_set_weight": "1",
            "static": "0",
        },
    ]
    with pytest.raises(ValueError, match="inconsistent is_event"):
        deduplicate_risk_rows(rows, columns, ("static",))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("is_event", "2", "labels must be zero or one"),
        ("risk_set_weight", "0", "weights must be finite and positive"),
        ("risk_set_weight", "nan", "weights must be finite and positive"),
    ],
)
def test_deduplicate_rejects_invalid_label_or_weight(field, value, message):
    row = {
        "case_id": "case_a",
        "primary_arc_index": "0",
        "sample_step": "10",
        "sample_pre_step": "0",
        "sample_time_block_200ps": "0",
        "is_event": "1",
        "risk_set_weight": "1",
        "static": "0",
    }
    row[field] = value

    with pytest.raises(ValueError, match=message):
        deduplicate_risk_rows([row], InputColumns(), ("static",))


def test_membership_reader_rejects_duplicate_member_id(tmp_path):
    membership_path = tmp_path / "membership.csv"
    _write_csv(
        membership_path,
        [
            {"step": 0, "oxygen_id": 7, "arc_index": 0},
            {"step": 0, "oxygen_id": 7, "arc_index": 1},
        ],
    )

    with pytest.raises(ValueError, match="duplicate member 7 at step 0"):
        list(iter_membership_frames(membership_path, InputColumns(), arc_count=4))


def test_feature_join_rejects_nonuniform_membership_steps(tmp_path):
    membership_path = tmp_path / "membership.csv"
    _write_csv(
        membership_path,
        [
            {"step": 0, "oxygen_id": 1, "arc_index": 0},
            {"step": 10, "oxygen_id": 1, "arc_index": 0},
            {"step": 25, "oxygen_id": 1, "arc_index": 0},
        ],
    )
    columns = InputColumns()
    rows = [
        {
            "case_id": "case_a",
            "primary_arc_index": 0,
            "sample_step": 25,
            "sample_pre_step": 25,
        }
    ]
    config = TurnoverConfig(
        history_lags_ps=(1.0,),
        frame_interval_ps=1.0,
        arc_count=4,
        arc_half_width=0,
    )

    with pytest.raises(ValueError, match="nonuniform membership step stride"):
        enrich_turnover_features(
            rows,
            {"case_a": membership_path},
            columns,
            config,
        )


def test_feature_join_marks_missing_history_incomplete(tmp_path):
    membership_path = tmp_path / "membership.csv"
    _write_csv(
        membership_path,
        [
            {"step": 0, "oxygen_id": 1, "arc_index": 0},
            {"step": 10, "oxygen_id": 2, "arc_index": 0},
        ],
    )
    columns = InputColumns()
    rows = [
        {
            "case_id": "case_a",
            "primary_arc_index": 0,
            "sample_step": 10,
            "sample_pre_step": 10,
        }
    ]
    config = TurnoverConfig(
        history_lags_ps=(2.0,),
        frame_interval_ps=1.0,
        arc_count=4,
        arc_half_width=0,
    )

    enriched, coverage = enrich_turnover_features(
        rows,
        {"case_a": membership_path},
        columns,
        config,
    )

    assert enriched[0]["turnover_history_complete"] == 0
    assert coverage[0]["complete_history_anchor_count"] == 0
    assert coverage[0]["incomplete_history_anchor_count"] == 1


def test_feature_join_uses_only_pre_event_anchor_history(tmp_path):
    membership_path = tmp_path / "membership.csv"
    _write_csv(
        membership_path,
        [
            {"step": 0, "oxygen_id": 1, "arc_index": 0},
            {"step": 10, "oxygen_id": 1, "arc_index": 0},
            {"step": 20, "oxygen_id": 2, "arc_index": 0},
            {"step": 30, "oxygen_id": 2, "arc_index": 0},
        ],
    )
    columns = InputColumns()
    rows = [
        {
            "case_id": "case_a",
            "primary_arc_index": 0,
            "sample_step": 30,
            "sample_pre_step": 20,
        }
    ]
    config = TurnoverConfig(
        history_lags_ps=(1.0,),
        frame_interval_ps=1.0,
        arc_count=4,
        arc_half_width=0,
    )

    enriched, _ = enrich_turnover_features(
        rows,
        {"case_a": membership_path},
        columns,
        config,
    )

    assert enriched[0]["turnover_history_complete"] == 1
    assert enriched[0]["membership_node_survival_fraction_1ps"] == 0.0
    assert enriched[0]["membership_gross_turnover_fraction_1ps"] == 1.0


def test_turnover_permutation_preserves_vectors_within_case_and_block():
    rows = [
        {
            "case_id": case,
            "sample_time_block_200ps": block,
            "primary_arc_index": index,
            "sample_step": index,
            "is_event": index % 2,
            "baseline": index + 100,
            "turnover_a": index,
            "turnover_b": index + 10,
        }
        for index, (case, block) in enumerate(
            (("a", 0), ("a", 0), ("a", 1), ("a", 1), ("b", 0), ("b", 0))
        )
    ]
    columns = InputColumns()
    permuted = _permuted_features(
        rows,
        ("turnover_a", "turnover_b"),
        columns,
        np.random.default_rng(3),
    )

    for original, shuffled in zip(rows, permuted):
        for field in (
            "case_id",
            "sample_time_block_200ps",
            "primary_arc_index",
            "sample_step",
            "is_event",
            "baseline",
        ):
            assert shuffled[field] == original[field]
    for group in (("a", 0), ("a", 1), ("b", 0)):
        original_vectors = sorted(
            (row["turnover_a"], row["turnover_b"])
            for row in rows
            if (row["case_id"], row["sample_time_block_200ps"]) == group
        )
        shuffled_vectors = sorted(
            (row["turnover_a"], row["turnover_b"])
            for row in permuted
            if (row["case_id"], row["sample_time_block_200ps"]) == group
        )
        assert shuffled_vectors == original_vectors


def test_weighted_auc_and_bh_reference_values():
    target = np.asarray([0, 1, 0, 1])
    weight = np.asarray([1.0, 2.0, 3.0, 4.0])

    assert _weighted_auc(target, np.asarray([0.5, 0.5, 0.5, 0.5]), weight) == 0.5
    assert _weighted_auc(target, np.asarray([0.1, 0.8, 0.2, 0.9]), weight) == 1.0
    assert _weighted_auc(target, np.asarray([0.9, 0.2, 0.8, 0.1]), weight) == 0.0
    assert _bh_adjust([0.01, 0.04, 0.03, 0.2]) == pytest.approx(
        [0.04, 0.05333333333333334, 0.05333333333333334, 0.2]
    )


def test_run_analysis_rejects_nonempty_output_directory(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "existing.txt").write_text("preserve me\n")

    with pytest.raises(ValueError, match="refusing non-empty output directory"):
        run_analysis(
            tmp_path / "missing-cases.tsv",
            tmp_path / "missing-risk.csv",
            output,
        )

    assert (output / "existing.txt").read_text() == "preserve me\n"


def test_streamed_turnover_prediction(tmp_path):
    case_rows = []
    risk_rows = []
    for case_index, case_id in enumerate(("case_a", "case_b", "case_c")):
        membership_path = tmp_path / f"{case_id}.csv.gz"
        membership_rows = []
        states = {}
        for frame in range(80):
            for arc in range(4):
                state = ((frame + arc + case_index) // 2) % 2
                states[(frame, arc)] = state
                offset = 50 if state else 0
                for local_id in range(5):
                    membership_rows.append(
                        {
                            "step": frame * 10,
                            "oxygen_id": arc * 100 + offset + local_id,
                            "arc_index": arc,
                        }
                    )
        _write_csv(membership_path, membership_rows)
        case_rows.append({"case_id": case_id, "membership_table": membership_path})
        for frame in range(4, 80):
            pre_frame = frame - 1
            for arc in (0, 1):
                event = int(states[(pre_frame, arc)] != states[(pre_frame - 1, arc)])
                risk_rows.append(
                    {
                        "case_id": case_id,
                        "primary_arc_index": arc,
                        "sample_step": frame * 10,
                        "sample_pre_step": pre_frame * 10,
                        "sample_time_block_200ps": frame // 10,
                        "is_event": event,
                        "risk_set_weight": 1.0,
                        "static": float((frame + 2 * arc) % 5),
                    }
                )

    duplicate = dict(risk_rows[0])
    duplicate["risk_set_weight"] = 0.5
    risk_rows.append(duplicate)
    cases_table = tmp_path / "cases.tsv"
    risk_table = tmp_path / "risk.csv"
    _write_csv(cases_table, case_rows, delimiter="\t")
    _write_csv(risk_table, risk_rows)

    config = TurnoverConfig(
        history_lags_ps=(1.0, 2.0),
        frame_interval_ps=1.0,
        arc_count=4,
        arc_half_width=0,
        fold_count=4,
        embargo_blocks=0,
        penalty=0.05,
        bootstrap_samples=32,
        null_samples=8,
        random_seed=17,
    )
    output = tmp_path / "output"
    summary = run_analysis(
        cases_table,
        risk_table,
        output,
        baseline_features=("static",),
        config=config,
    )

    assert summary["status"] == "PASS"
    assert summary["case_count"] == 3
    assert summary["input_risk_row_count"] == len(risk_rows)
    assert summary["unique_anchor_count"] == len(risk_rows) - 1
    assert summary["complete_history_anchor_count"] == len(risk_rows) - 1
    assert summary["qualified_primary_count"] == (
        summary["qualified_within_case_count"] + summary["qualified_leave_one_case_out_count"]
    )
    for name in (
        "turnover_enriched_risk_sets.csv",
        "membership_coverage.csv",
        "prediction_scores.csv",
        "prediction_evidence.csv",
        "turnover_null_controls.csv",
        "out_of_fold_predictions.csv",
        "summary.json",
        "manifest.json",
        "REPORT.md",
    ):
        assert (output / name).is_file()

    with (output / "prediction_scores.csv").open() as handle:
        scores = list(csv.DictReader(handle))
    within_a = {
        row["model"]: float(row["weighted_log_loss"])
        for row in scores
        if row["evaluation"] == "within_case" and row["held_case"] == "case_a"
    }
    assert within_a["M2_turnover_history"] < within_a["M1_occupancy_history"]

    with (output / "turnover_enriched_risk_sets.csv").open() as handle:
        enriched = list(csv.DictReader(handle))
    first = next(
        row
        for row in enriched
        if row["case_id"] == "case_a"
        and row["primary_arc_index"] == "0"
        and row["sample_step"] == "40"
    )
    assert float(first["risk_set_weight"]) == 1.5
    assert int(first["aggregated_source_row_count"]) == 2
    assert 0.0 <= float(first["membership_gross_turnover_fraction_1ps"]) <= 1.0
    assert np.isfinite(float(first["membership_cumulative_turnover_fraction_2ps"]))
    assert json.loads((output / "summary.json").read_text())["random_seed"] == 17
