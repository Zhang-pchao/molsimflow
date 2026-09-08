import csv
import json

from molsimflow.postprocess.tpcl_dynamic_state import LagWindow, run_analysis


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_dynamic_state_separates_rate_and_cross_arc_excess(tmp_path):
    events = []
    event_id = 0
    for block in range(10):
        for arc, offset in ((0, 20.0), (1, 21.0)):
            event_id += 1
            time_ps = block * 200.0 + offset
            events.append(
                {
                    "event_id": event_id,
                    "arc_index": arc,
                    "transition_step": int(time_ps),
                    "transition_time_ns": time_ps / 1000.0,
                    "end_step": int(time_ps) - 1,
                    "post_end_step": int(time_ps) + 1,
                    "quality_status": "candidate",
                    "jump_distance_A": 1.0,
                    "localization_noise_A": 0.1,
                }
            )
    event_path = tmp_path / "events.csv"
    _write_csv(event_path, events)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "case_id": "case",
                "arc_count": 8,
                "first_time_ns": 0.0,
                "last_time_ns": 2.0,
                "admitted_event_rows": len(events),
                "event_cluster_count": 10,
            }
        ),
        encoding="utf-8",
    )
    sources = tmp_path / "sources.tsv"
    sources.write_text(
        "case_id\tpropagation_summary\tevents\n"
        f"case\t{summary_path}\t{event_path}\n",
        encoding="utf-8",
    )

    result = run_analysis(
        sources,
        tmp_path / "out",
        windows=(LagWindow("fast", 0.0, 5.0), LagWindow("slow", 5.0, 50.0)),
        block_ps=200.0,
        null_samples=100,
        bootstrap_samples=100,
        event_status="candidate",
        random_seed=7,
    )

    assert result["status"] == "PASS"
    with (tmp_path / "out" / "yielding_by_case.csv").open() as handle:
        yielding = next(csv.DictReader(handle))
    assert float(yielding["yielding_rate_per_arc_ns"]) == 1.25
    with (tmp_path / "out" / "cooperativity_by_case.csv").open() as handle:
        rows = {row["window"]: row for row in csv.DictReader(handle)}
    assert int(rows["fast"]["observed_cross_arc_pairs"]) == 10
    assert float(rows["fast"]["cooperative_excess_fraction"]) > 0.0
    assert rows["fast"]["informative_null_count"] == "False"
    assert 0.0 <= float(rows["fast"]["bh_q_primary_family"]) <= 1.0
    assert (tmp_path / "out" / "manifest.json").is_file()
